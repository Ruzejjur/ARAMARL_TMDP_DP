import numpy as np
from numpy.random import choice
import math
import multiprocessing
import os
import copy
from typing import cast, Optional
from tqdm.auto import tqdm
import hashlib
import logging
from pathlib import Path 
import filelock

from .base import LearningAgent
from engine_DP import CoinGame 

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
ValueFunction = np.ndarray
ActionDetails = np.ndarray

def _reset_sim_env_to_state(env_snapshot: CoinGame, obs: State):
    """
    Resets a simulation environment instance to a specific state ID.

    It works by reversing the state encoding
    process defined in the `CoinGame.get_state()` method, using radix decoding
    to extract player positions and coin statuses from the single integer `obs`.

    The function directly modifies the attributes of the provided `env_snapshot`
    object in place.

    Args:
        env_snapshot (CoinGame): The environment instance to be modified. This
            should be a deterministic copy used only for simulation.
        obs (State): The integer ID of the target state to set.
    """
    # Start by resetting the environment to a clean, default initial state.
    # This ensures that any attributes not explicitly set by the decoding
    # process (like step_count) are in a known, valid state.
    env_snapshot.reset()
    
    # --- Radix Decoding ---
    # Define the bases for the mixed-radix number system used in state encoding.
    # `base_pos` is the number of possible positions for a single player (N*N).
    # `base_coll` is the number of states for a single coin collection flag (True/False).
    base_pos, base_coll = env_snapshot.grid_size**2, 2
    state_copy = obs
    
    # Decode the state integer piece by piece, in the reverse order of encoding.
    p1_collected_c1 = bool(state_copy % base_coll); state_copy //= base_coll # Player 1 collected coin 1
    p1_collected_c0 = bool(state_copy % base_coll); state_copy //= base_coll # Player 1 collected coin 0
    p0_collected_c1 = bool(state_copy % base_coll); state_copy //= base_coll # Player 0 collected coin 1
    p0_collected_c0 = bool(state_copy % base_coll); state_copy //= base_coll # Player 0 collected coin 0
    p1_flat = state_copy % base_pos; state_copy //= base_pos      # Player 1 flattened position
    p0_flat = state_copy                                          # Player 0 flattened position
    
    # Convert the 1D flattened player positions back into 2D [row, col] coordinates.
    env_snapshot.player_0_pos = np.array([p0_flat % env_snapshot.grid_size, p0_flat // env_snapshot.grid_size])
    env_snapshot.player_1_pos = np.array([p1_flat % env_snapshot.grid_size, p1_flat // env_snapshot.grid_size])
    
    # Set the coin collection flags for both players.
    env_snapshot.player_0_collected_coin0, env_snapshot.player_0_collected_coin1 = p0_collected_c0, p0_collected_c1
    env_snapshot.player_1_collected_coin0, env_snapshot.player_1_collected_coin1 = p1_collected_c0, p1_collected_c1
    
    # Update the global availability of the coins based on the collection flags.
    # A coin is unavailable if either player has collected it.
    env_snapshot.coin0_available = not (p0_collected_c0 or p1_collected_c0)
    env_snapshot.coin1_available = not (p0_collected_c1 or p1_collected_c1)

def _compute_lookup_for_chunk(args: tuple) -> list:
    """
    Top-level worker function for parallel lookup table generation.

    This function is designed to be executed in a separate, independent process
    by a `multiprocessing.Pool`. It receives a "chunk" (a subset) of all
    states and is responsible for computing the next-state and reward lookups
    for each state in its assigned chunk.

    Args:
        args (tuple): A tuple containing all necessary information for the worker:
            - states_chunk (list[int]): The subset of state indices to process.
            - env_params (dict): A serializable configuration dictionary used to
              create a local, deterministic `CoinGame` instance.
            - player_id (int): The ID of the agent (0 or 1) for whom the rewards
              are being calculated.
            - self_actions (np.ndarray): Details of the agent's actions.
            - opp_actions (np.ndarray): Details of the opponent's actions.

    Returns:
        list[tuple]: A list where each element is a tuple `(s, s_prime_row, r_row)`,
                     representing the computed lookup table rows for a single state `s`.
    """
    # Initialization
    # Unpack the arguments tuple for clarity.
    states_chunk, env_params, player_id, self_actions, opp_actions = args
    
    # Create a private, deterministic environment instance for this worker process.
    env_snapshot = CoinGame(**env_params)
    
    # This list will store the results for all states processed by this worker.
    results_chunk = []

    # Iterate through each state assigned to this worker's chunk.
    for s in states_chunk:
        # Pre-allocate arrays to store the results for the current state `s`.
        s_prime_row = np.zeros((len(self_actions), len(opp_actions)), dtype=int)
        r_row = np.zeros((len(self_actions), len(opp_actions)), dtype=float)

        # Wrap the simulation in a try-except block to handle
        # invalid or unreachable states that might cause errors during decoding.
        try:
            # Set the local environment to the current state `s`.
            _reset_sim_env_to_state(env_snapshot, s)
            
            # Iterate through every possible pair of EXECUTED actions to find the outcome.
            for a_self_exec in range(len(self_actions)):
                for a_opp_exec in range(len(opp_actions)):
                    # Store the state before the step to ensure we can revert to it.
                    original_state = env_snapshot.get_state()
                    
                    # Ensure the action tuple is in the correct (p0, p1) order
                    # for the environment's step function.
                    action_pair = (a_self_exec, a_opp_exec) if player_id == 0 else (a_opp_exec, a_self_exec)
                    
                    # Simulate one deterministic step to get the outcome.
                    s_prime, rewards_vec, *_ = env_snapshot.step(action_pair)
                    
                    # Store the resulting next state and the reward for our agent.
                    s_prime_row[a_self_exec, a_opp_exec] = s_prime
                    r_row[a_self_exec, a_opp_exec] = rewards_vec[player_id]
                    
                    # Reset the environment to its pre-step state
                    # to ensure the next simulated action pair starts from the same `s`.
                    _reset_sim_env_to_state(env_snapshot, original_state)
        except (IndexError, ValueError):
            # If a state is invalid, simply skip it. The result rows will remain
            # as arrays of zeros, which is a safe default.
            pass 
        
        # Append the completed lookup rows for state `s` to this worker's results list.
        results_chunk.append((s, s_prime_row, r_row))
            
    # Return the completed list of results for this entire chunk back to the main process.
    return results_chunk


class _BaseLevelK_TMDP_DP_Agent(LearningAgent):
    """
    Base class for Level-K Threatened Markov Decision Process Dynamic Programming (TMDP_DP) agents.

    This implementation differs from the standard DP agent by using a value function
    V(s, b), which represents the value of being in state 's' *given* that the
    opponent has committed to taking intended action 'b'. This allows for a more
    explicit form of best-response reasoning.

    It handles the common, expensive setup procedures like pre-computing the
    environment's deterministic transition and reward dynamics.

    Attributes:
        k (int): The cognitive level of the agent. A Level-k agent models its
                 opponent as a Level-(k-1) agent.
        n_states (int): The total number of states in the environment.
        epsilon (float): The exploration rate for epsilon-greedy policies.
        gamma (float): The discount factor for future rewards.
        player_id (int): The agent's identifier (0, 1).
        action_space (np.ndarray): The set of actions available to this agent.
        opponent_action_space (np.ndarray): The set of actions available to the opponent.
        lower_level_k_epsilon(float): The exploration rate embeded into lower levels of the k-level hierarchy.
        env_snapshot (CoinGame): A deep copy of the environment, set to be deterministic.
                                 Used for simulating outcomes to build the model.
        V (ValueFunction): The agent's value function table, representing
                           V(s, b), where 's' is the state and 'b' is the opponent's action.
        s_prime_lookup (np.ndarray): A pre-computed table mapping
                                     (s, a_self_executed, a_opp_executed) -> s'.
        r_lookup (np.ndarray): A pre-computed table mapping
                              (s, a_self_executed, a_opp_executed) -> r.
        self_action_details (ActionDetails): Details (move, push) of the agent's
                                             combined actions.
        opponent_action_details (ActionDetails): Details (move, push) of the
                                                 opponent's combined actions.
        num_self_actions (int): The number of actions available to this agent.
        num_opponent_actions (int): The number of actions available to the opponent.
        opponent (object): The agent's internal model of its opponent. This is
                           defined and initialized in subclasses.
        dirichlet_counts (np.ndarray): Dirichlet counts for modeling a Level-0
                                       opponent's policy. Initialized in subclasses that require it (k=1).
        _opponent_policy_cache (dict): A cache to store opponent policies for future states
                                       to prevent redundant calculations within a single time step.
    """
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray, lower_level_k_epsilon:float,
                 n_states: int, epsilon: float, gamma: float, initial_V_value: float, player_id: int, env):
        
        # --- Core Agent Parameters ---
        self.k = k
        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.initial_V_value = initial_V_value
        self.player_id = player_id
        self.action_space = action_space
        self.opponent_action_space = opponent_action_space
        self.lower_level_k_epsilon = lower_level_k_epsilon
        
        # --- Environment Snapshot for Model Computation ---
        # Create a deterministic copy of the environment. This allows
        # simulation of the outcome of any action pair from any state without
        # affecting the actual game environment.
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.player_0_execution_prob = 1.0
        self.env_snapshot.player_1_execution_prob = 1.0
        
        # ---Action Setup ---
        if self.env_snapshot.enable_push:
            self.self_action_details = self.env_snapshot.combined_actions
            self.opponent_action_details = self.env_snapshot.combined_actions
        else: 
            self.self_action_details = self.env_snapshot.combined_actions[:4,:]
            self.opponent_action_details = self.env_snapshot.combined_actions[:4, :]
            
        self.num_self_actions = len(action_space)
        self.num_opponent_actions = len(opponent_action_space)
        self.self_available_move_actions_num = len(env.available_move_actions)
        self.opponent_available_move_actions_num = len(env.available_move_actions)

        # --- Value Function and Model Initialization ---
        # V(s, opponent_action): The value of being in state 's' *after* 
        # the opponent has committed to taking 'opponent_action'.
        self.V = self._setup_value_function(self.initial_V_value)
            
        # --- Lookup tables computation ---
        # Pre-compute and save (s, a_self_executed, a_opp_executed) -> (s', r) lookup tables.
        # These tensors store the outcomes for every state and *executed* action pair
        # mapped to resulting s'. 
        # If it was precomputed in the past retrieve the tables from the cache
        
        # Determine the project's root directory
        #    __file__ is the path to the current script (e.g., .../agents/mdp_dp.py)
        #    .parent gives the directory of the script (.../agents/)
        #    .parent again gives the project root (.../TMDP_DP/)
        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent
        
        # Define the cache directory relative to the project root
        cache_dir = PROJECT_ROOT / "results" / "agent_lookup_cache"
        
        # Create a unique signature based on environment params and player_id
        env_params_for_hash = {
            'grid_size': env.grid_size,
            'enable_push': env.enable_push,
            'push_distance': env.push_distance,
            'rewards': env.get_reward_config() 
        }
        
        # Convert dict to a canonical string representation (sorted keys)
        params_string = repr(env_params_for_hash)
        
        # Create a unique hash for this configuration + player_id
        hasher = hashlib.sha256()
        hasher.update(params_string.encode('utf-8'))
        hasher.update(str(player_id).encode('utf-8'))
        config_hash = hasher.hexdigest()
        
        cache_filename = cache_dir / f"lookups_{config_hash}.npz"
        lock_filename = cache_dir / f"lookups_{config_hash}.npz.lock"

        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the file lock for the specific file name
        lock = filelock.FileLock(lock_filename)
        
        with lock:
            # Now that we have the lock, we must check AGAIN if the file exists.
            # Another process might have created it while we were waiting for the lock.
            if cache_filename.exists():
                # Cache HIT: The file was created by another process. We can just load it.
                logging.info(f"CACHE HIT for Player {player_id}. Loading from {cache_filename}")
                with np.load(cache_filename) as data:
                    self.s_prime_lookup = data['s_prime_lookup']
                    self.r_lookup = data['r_lookup']
            else:
                # Cache MISS: We are the first and only process here. Time to do the work.
                logging.info(f"CACHE MISS for Player {player_id}. Pre-computing lookups...")
                self.s_prime_lookup, self.r_lookup  = self._precompute_lookups()
                
                logging.info(f"Saving computed lookups to cache: {cache_filename}")
                np.savez_compressed(cache_filename, s_prime_lookup=self.s_prime_lookup, r_lookup=self.r_lookup)
        
        # --- Opponent Model Initialization (handled by subclasses) ---
        self.opponent = None
        self.dirichlet_counts = None
        self._opponent_policy_cache = {}
        
        # --- Probability of action execution tensor (handled by subclasses) ---
        self.prob_exec_tensor = None
        
        
    def _setup_value_function(self, initial_V_value: float) -> ValueFunction:
        """
        Initializes the value function V(s, opponent_action).

        The value of all terminal states is set to 0, as no further rewards
        can be obtained from them.

        Args:
            initial_V_value (float): The initial value for all non-terminal states.

        Returns:
            np.ndarray: The initialized value function table of shape
                        (n_states, num_opponent_actions).
        """
        
        V = np.ones([self.n_states, len(self.opponent_action_space)])*initial_V_value
        
        # tqdm shows progress bar.
        for s in tqdm(range(self.n_states), desc="Initializing value function."):
            if self._is_terminal_state(s):
                V[s,:] = 0
                
        return V
            
    def _is_terminal_state(self, obs:State) -> bool:
        """
        Checks if a given state is terminal.

        A state is terminal if either player has collected both of their coins
        (a win) or if all four coins have been claimed by any combination of
        players (a draw or a win). This logic is coupled to the environment's
        specific state encoding scheme.
        (Implementation is identical to the _BaseLevelK_MDP_DP_Agent).
        """
        
        # Grid size N, base for coin collection status
        _, base_coll = self.env_snapshot.grid_size**2, 2
        
        # Radix decoding of the state integer to get coin collection status
        state_copy = obs
        p1_collected_c1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        p1_collected_c0 = bool(state_copy % base_coll)
        state_copy //= base_coll
        p0_collected_c1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        p0_collected_c0 = bool(state_copy % base_coll)

        # Check for win conditions
        p0_wins = p0_collected_c0 and p0_collected_c1
        p1_wins = p1_collected_c0 and p1_collected_c1
        
        # Check for draw condition
        coins_gone = (p0_collected_c0 or p1_collected_c0) and (p0_collected_c1 or p1_collected_c1)
        is_draw = coins_gone and not p0_wins and not p1_wins
        
        # NOTE: The DP agent cannot account for the episode ending due to max_steps,
        # as this is not encoded in the state itself. This is a known limitation.
        return p0_wins or p1_wins or is_draw
        
    def _precompute_lookups(self) -> tuple:
        """
        Builds lookup tables for next states (s') and rewards (r) in parallel.

        This is a one-time, computationally expensive process that maps every
        (state, executed_action_self, executed_action_opponent) tuple to its
        deterministic outcome. To significantly accelerate this, the workload is
        parallelized across all available CPU cores using Python's `multiprocessing`
        module.

        The overall process is as follows:
        1. The full set of states is divided into smaller, non-overlapping "chunks".
        2. A pool of worker processes is created (typically one per CPU core).
        3. Each worker is assigned one chunk of states and a lightweight copy of the
        environment's configuration.
        4. Workers compute their assigned portion of the lookup tables in parallel.
        5. The main process collects the results from all workers and assembles them
        into the final, complete lookup tables.

        This parallel approach dramatically reduces the wall-clock time required for
        agent initialization, especially when the lookup tables are not yet cached.

        Returns:
            A tuple containing:
            - s_prime_lookup (np.ndarray): Table of next states, with shape
            (n_states, num_self_actions, num_opponent_actions).
            - r_lookup (np.ndarray): Table of rewards for the current agent, with shape
            (n_states, num_self_actions, num_opponent_actions).
        """
        desc_str = f"Pre-computing lookups for Level-{self.k} TMDP DP Agent (Player {"DM" if self.player_id == 0 else "Adv"})"
        
        # Pre-allocate the final lookup tables with zeros.
        s_prime_lookup = np.zeros((self.n_states, self.num_self_actions, self.num_opponent_actions), dtype=int)
        r_lookup = np.zeros((self.n_states, self.num_self_actions, self.num_opponent_actions), dtype=float)

        # Prepare Arguments for Worker Processes
        env_params = {
            'max_steps': self.env_snapshot.max_steps,
            'grid_size': self.env_snapshot.grid_size,
            'enable_push': self.env_snapshot.enable_push,
            'push_distance': self.env_snapshot.push_distance,
            'rewards': self.env_snapshot.get_reward_config(),
            'action_execution_probabilities': [1.0, 1.0] # Deterministic for simulation
        }
        
        # Divide the Workload into Chunks
        # Determine the optimal number of worker processes based on available CPU cores.
        if hasattr(os, "sched_getaffinity"):
            # We are on a Linux-like system that supports affinity.
            # Get the number of CPUs we are allowed to use by the OS/scheduler.
            num_of_available_cores = len(os.sched_getaffinity(0))  # pyright: ignore[reportAttributeAccessIssue]
            # NOTE: Suppresing platform specific warning
            
            # If our affinity is less than the total machine CPUs, it means we are in a
            # restricted environment (like a SLURM job). We must respect that limit.
            # Otherwise, we have access to the whole machine, so we apply the "good citizen"
            # rule and leave a couple of cores free.
            if num_of_available_cores < multiprocessing.cpu_count():
                num_processes = num_of_available_cores
            else:
                num_processes = max(1, multiprocessing.cpu_count() - 2)
        else:
            # We are on another OS (e.g., Windows/macOS) that doesn't have sched_getaffinity.
            # Fall back to the safe "good citizen" approach for shared systems
            num_processes = max(1, multiprocessing.cpu_count() - 2)
            
        # Calculate the number of states each worker should handle. `math.ceil` ensures
        # all states are covered, even if `n_states` isn't perfectly divisible.
        chunk_size = math.ceil(self.n_states / num_processes)
        all_states = range(self.n_states)
        
        # Split the full range of states into a list of smaller, non-overlapping lists (chunks).
        state_chunks = [list(all_states[i:i + chunk_size]) for i in range(0, self.n_states, chunk_size)]
        
        # Prepare the final list of tasks, where each task is a tuple of arguments
        # for the top-level `_compute_lookup_for_chunk` worker function.
        tasks = [(chunk, env_params, self.player_id, self.self_action_details, self.opponent_action_details) for chunk in state_chunks]

        # Execute Tasks in Parallel 
        with multiprocessing.Pool(processes=num_processes) as pool:
            # `pool.imap_unordered` distributes the tasks to the workers. It returns an
            # iterator that yields results as they are completed, which can be more
            # memory-efficient and slightly faster than waiting for all tasks to finish.
            # `tqdm` provides a progress bar to monitor the completion of chunks.
            results_from_chunks = list(tqdm(pool.imap_unordered(_compute_lookup_for_chunk, tasks), total=len(tasks), desc=desc_str))

        # Assemble Results
        # Iterate through the list of results returned by each worker. Each result
        # is itself a list of tuples `(state, s_prime_row, r_row)`.
        for chunk in results_from_chunks:
            for s, s_prime_row, r_row in chunk:
                # Place the computed rows into the correct slice of the final lookup tables.
                s_prime_lookup[s] = s_prime_row
                r_lookup[s] = r_row
                
        return s_prime_lookup, r_lookup
    
    def update_epsilon(self, new_epsilon_agent: float, new_epsilon_lower_k_level: Optional[float]):
        """
        Updates the exploration rate for this agent and its recursive opponent model.
        This ensures that if the exploration schedule changes, the change
        propagates down the entire cognitive hierarchy.
        (Implementation is identical to the _BaseLevelK_MDP_DP_Agent).
        
        Args:
            new_epsilon_agent (float): The new exploration rate for agent.
            new_epsilon_lower_k_level (float): The new exploration rate for agents internal model of the opponent reasoning (k-1 levels)
        """
        
        self.epsilon = new_epsilon_agent
        if self.k > 1 and self.opponent:
            new_epsilon_lower_k_level = cast(float,new_epsilon_lower_k_level)
            self.opponent.update_epsilon(new_epsilon_lower_k_level, new_epsilon_lower_k_level)
    
class LevelK_TMDP_DP_Agent_Stationary(_BaseLevelK_TMDP_DP_Agent):
    """
    A Level-K DP agent that assumes a stationary environment.

    "Stationary" means the probabilities of an action succeeding or failing
    are fixed and do not depend on the current state. This agent pre-computes
    a single, state-independent transition probability tensor. It models its
    opponent recursively as a Level-(k-1) stationary agent.

    - A Level-1 agent models its opponent as Level-0 (uniformly random policy).
    - A Level-k agent (k>1) models its opponent as a Level-(k-1) agent.
    
    Attributes: 
        Check parent
    """

    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray, lower_level_k_epsilon:float,
                 n_states: int, epsilon: float, gamma: float, initial_V_value: float, player_id: int, env):
        if k < 1:
            raise ValueError("Level k must be a positive integer.")

        # Call the base class constructor to handle all common setup.
        super().__init__(k, action_space, opponent_action_space, lower_level_k_epsilon, n_states, 
                         epsilon, gamma, initial_V_value, player_id, env)
        
        # Pre-calculate the state-independent execution probability tensor:
        # P(a_self_exec, a_opp_exec | a_self_intend, a_opp_intend)
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)
        
        # --- Opponent Model Initialization ---
        if self.k == 1:
            # Base Case (k=1): Model opponent as Level-0 (random policy).
            # We use Dirichlet counts to learn this policy from observations.
            # Initially, we assume a uniform prior.
            self.dirichlet_counts = np.ones((self.n_states, len(self.opponent_action_space)))
        elif self.k > 1:
            # Recursive Step (k>1): The opponent is a Level-(k-1) version of this
            # same class, with the player roles reversed.
            self.opponent = self.__class__( 
                k=self.k - 1,
                action_space=self.opponent_action_space,
                opponent_action_space=self.action_space,
                lower_level_k_epsilon=self.lower_level_k_epsilon,
                n_states=self.n_states,
                epsilon=self.lower_level_k_epsilon,
                gamma=self.gamma,
                initial_V_value = self.initial_V_value,
                player_id=1 - self.player_id,
                env=env
            )
            
    def _clear_policy_cache_recursive(self):
        """
        Clears the policy cache for this agent and all agents
        down its cognitive hierarchy. This is essential to call before each
        new `act` or `update` to ensure policies are not stale.
        """

        self._opponent_policy_cache.clear()
        if self.k > 1 and self.opponent:
            # Recursively call the clear method on the opponent model
            self.opponent._clear_policy_cache_recursive()
            
    def get_opponent_policy(self, obs: State) -> Policy:
        """
        Estimates the opponent's policy (action probabilities) for a given state.

        - For a k=1 agent, this is derived from the learned Dirichlet counts,
          approximating a random (Level-0) opponent.
        - For a k>1 agent, this is the epsilon-greedy policy of its internal
          Level-(k-1) opponent model.
          
        Args:
            obs: The current state.

        Returns:
            np.ndarray: A probability distribution over the opponent's actions.
        """
        
        # If the opponents policy was already calculated for the current observation, return it.
        if obs in self._opponent_policy_cache:
            return self._opponent_policy_cache[obs]
        
        # Initialize policy variable
        policy = None
        
        # --- Level-1 Agent: Opponent is Level-0 (Learned Random Policy) ---
        if self.k == 1:
            # Check if atributes is still initalized to None
            assert self.dirichlet_counts is not None, "dirichlet_counts should be initialized for a Level-1 agent."
            
            # Normalize the counts for the given state to get a probability distribution.
            dir_sum = np.sum(self.dirichlet_counts[obs])
            if dir_sum != 0:
                policy = self.dirichlet_counts[obs] / dir_sum
            else: 
                # Handle intialization of dirichlet_counts to 0
                policy = np.ones(self.num_opponent_actions) / self.num_opponent_actions
        
        # --- Level-k > 1 Agent: Opponent is Level-(k-1) ---
        else:
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            
            policy = self.opponent.get_policy(obs)
            
        self._opponent_policy_cache[obs] = policy
            
        return policy
            
    def _calculate_expected_future_values(self, s_prime_array: np.ndarray) -> np.ndarray:
        """
        Efficiently calculates the expected future value for an array of next states.

        Calculates the expected value E[V(s')] for an array of next states.

        For each next state s', it computes E_{p(b'|s')}[V(s', b')], which is the
        value of s' averaged over the opponent's predicted next action b'. This
        uses an optimization to compute the value only for reachable states.

        Args:
            s_prime_array: An array of potential next states (s').

        Returns:
            An array of the same shape as s_prime_array, containing the
            calculated expected future values.
        """
        # Find unique next states to avoid redundant calculations
        unique_s_primes, inverse_indices = np.unique(s_prime_array, return_inverse=True)
        
        # Calculate the expected value V(s') for each unique next state s'.
        # V(s') = sum over b' [ p(b'|s') * V(s', b') ]
        expected_V_map = {
            s_prime: np.dot(self.V[s_prime, :], self.get_opponent_policy(s_prime))
            for s_prime in unique_s_primes
        }
        
        # Map the computed values back to the original array shape
        expected_V_flat = np.array([expected_V_map.get(s, 0) for s in unique_s_primes])
        return expected_V_flat[inverse_indices].reshape(s_prime_array.shape)

    def optim_act(self, obs: State) -> Action:
        """
        Selects the optimal action by calculating the Q-value for each of the
        agent's actions and choosing the best one. This is done using efficient
        vectorized operations.
        """
        
        # Get pre-computed outcomes (next states and rewards) for all executed
        # action pairs from the current state 'obs'.
        rewards_executed = self.r_lookup[obs, :, :]
        s_primes_executed = self.s_prime_lookup[obs, :, :]
        
        # Calculate the expected future values for all possible next states.
        future_V_executed = self._calculate_expected_future_values(s_primes_executed)

        # Using the transition model, calculate the expected reward for each
        # *intended* action pair by marginalizing over the *executed* outcomes.
        # 'ijkl,kl->ij' sums over the last two axes (executed actions).
        expected_self_rewards_intended = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, rewards_executed, optimize=True)
        
        # Similarly, calculate the expected future value for each *intended* action pair.
        weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, future_V_executed, optimize=True)
        
        # Get the opponent's predicted policy (action probabilities) for the current state.
        opponent_policy_in_obs = self.get_opponent_policy(obs)
        
        # Calculate the total Q-value for each of our actions by marginalizing
        # over the opponent's predicted policy.
        # Q(s,a) = Σ_b [ p(b|s) * E[R(s,a,b) + γ * V(s')] ]
        total_action_values = np.dot(expected_self_rewards_intended, opponent_policy_in_obs) + \
                              self.gamma * np.dot(weighted_sum_future_V, opponent_policy_in_obs)
        
        # Choose the best action, breaking ties randomly.
        max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))
        return np.random.choice(max_indices)
        
    def get_policy(self, obs: State) -> Policy:
        """
        Calculates agent's own epsilon-greedy policy distribution for a
        given state.
        
        Args:
            obs: The current state.

        Returns:
            np.ndarray: A probability distribution over the agent's actions.
        """
        
        optimal_action_idx = self.optim_act(obs)
        num_actions = len(self.action_space)
        
        # Assign base exploration probability to all actions.
        policy = np.full(num_actions, self.epsilon / num_actions)
        # Add the exploitation probability to the optimal action.
        policy[optimal_action_idx] += 1.0 - self.epsilon
        
        return policy / np.sum(policy) # Normalize to ensure it's a valid distribution

    def act(self, obs:State, env=None) -> Action:
        """
        Selects an action based on agent's own policy.
        """
        # Clear the policy cache at the start of the decision-making process.
        self._clear_policy_cache_recursive()
        
        policy = self.get_policy(obs)
        return choice(self.action_space, p=policy)

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards = None):
        """
        Updates the agent's value function V(s,b) and its internal opponent model
        based on a single transition (s, a, b, s').
        
        Args:
            obs: The current state.
            actions: list of agents and opponents actions in state obs
            new_obs: The next state after transition.
        """
        
        # Clear the policy cache at the start of the update process.
        self._clear_policy_cache_recursive()

        # Identify the opponent's action based on player ID
        opponent_action_taken = actions[1] if self.player_id == 0 else actions[0]
        
         # --- Update Opponent Model ---
        if self.k > 1:
            # If k>1, recursively call update on the internal Level-(k-1) model
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            self.opponent.update(obs, actions, new_obs, None)
        else:
            # If k=1, update the Dirichlet counts for the observed opponent action.
            assert self.dirichlet_counts is not None, "dirichlet_counts should be initialized for a Level-1 agent."
            self.dirichlet_counts[obs, opponent_action_taken] += 1
        
        # If the state was already terminal, its value is fixed at 0, so no update needed.
        if self._is_terminal_state(obs):
            return
            
        # --- Vectorized V(obs, opponent_action_taken) Update ---
        
        # Get outcomes from the lookup tables for the current state.
        rewards_executed = self.r_lookup[obs, :, :]
        s_primes_executed = self.s_prime_lookup[obs, :, :]
        
         # Calculate expected future values for all possible resulting states.
        future_V_executed = self._calculate_expected_future_values(s_primes_executed)
        
        # Get the transition probabilities, conditioned on the opponents action.
        # NOTE: This is different from opponents resulting movement. It represents opponents *INDENDET* action.
        prob_exec_tensor_fixed_opp_action = self.prob_exec_tensor[:, opponent_action_taken, :, :]
        
        # Calculate the Q-values for each of our actions 'a', given s and opponent action 'b'.
        # Q(s, a) = E_{b~pi_opp}[ R(s,a,b) + gamma * V(s') ]
        expected_rewards_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, rewards_executed, optimize=True)
        expected_future_V_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, future_V_executed, optimize=True)
        
        # The Bellman update: V(s, b) is the max Q-value over agent's actions.
        q_values_for_actions = expected_rewards_per_action + self.gamma * expected_future_V_per_action
        self.V[obs, opponent_action_taken] = np.max(q_values_for_actions)
        
    def _calculate_execution_probabilities(self, env) -> np.ndarray:
        """
        Calculates the state-independent transition tensor P(executed | intended).

        This 4D tensor gives the probability of a specific pair of *executed*
        actions (a_self_exec, a_opp_exec) occurring, given a pair of *intended*
        actions (a_self_intend, a_opp_intend). This model is state-independent
        as it only relies on the fixed action execution probabilities from the env.
        (Implementation is identical to the LevelK_MDP_DP_Agent_Stationary).

        Returns:
            A 4D np.ndarray with dimensions corresponding to:
            (intended_self, intended_opp, executed_self, executed_opp)
        """
        
        if self.player_id == 0: 
            self_exec_prob, opp_exec_prob = env.player_0_execution_prob, env.player_1_execution_prob
        else:
            self_exec_prob, opp_exec_prob = env.player_1_execution_prob, env.player_0_execution_prob
         
        # Extract move and push actions from the detailes
        self_moves, self_pushes = self.self_action_details[:, 0], self.self_action_details[:, 1]
        opp_moves, opp_pushes = self.opponent_action_details[:, 0], self.opponent_action_details[:, 1]

        # --- Calculate P(a_self_exec | a_self_intend) ---
        prob_self = np.zeros((self.num_self_actions, self.num_self_actions))
        # Broadcasting `[:, None]` allows comparing every element to every other element.
        move_match = (self_moves[:, None] == self_moves[None, :])
        push_match = (self_pushes[:, None] == self_pushes[None, :])
        # If intended and executed actions match (both move and push), prob is `self_exec_prob`.
        prob_self[move_match & push_match] = self_exec_prob
        if self.self_available_move_actions_num > 1:
            # If move differs but push matches, it's a failed move. The probability
            # is distributed uniformly among all other possible moves.
            # NOTE: Push is deterministic so the probabilities of push not matching stays 0
            prob_self[~move_match & push_match] = (1.0 - self_exec_prob) / (self.self_available_move_actions_num - 1)
        
        # --- Calculate P(a_opp_exec | a_opp_intend) ---
        prob_opp = np.zeros((self.num_opponent_actions, self.num_opponent_actions))
        move_match = (opp_moves[:, None] == opp_moves[None, :])
        push_match = (opp_pushes[:, None] == opp_pushes[None, :])
        prob_opp[move_match & push_match] = opp_exec_prob
        if self.opponent_available_move_actions_num > 1:
            prob_opp[~move_match & push_match] = (1.0 - opp_exec_prob) / (self.opponent_available_move_actions_num - 1)
            
        # --- Combine into the 4D tensor ---
        # Using outer product via broadcasting to combine the two probability matrices.
        # P(a_s_exec, a_o_exec | a_s_int, a_o_int) = P(a_s_exec | a_s_int) * P(a_o_exec | a_o_int)
        return prob_self[:, np.newaxis, :, np.newaxis] * prob_opp[np.newaxis, :, np.newaxis, :]


class LevelK_TMDP_DP_Agent_NonStationary(LevelK_TMDP_DP_Agent_Stationary):
    """
    A Level-K DP agent for non-stationary environments.

    This agent is designed for environments where the action execution
    probabilities (e.g., `env.player_0_execution_prob`) can change over time.
    It adapts by recalculating the transition probability tensor before every
    action selection.
    (Implementation is identical to the LevelK_MDP_DP_Agent_NonStationary).
    
    Attributes: 
        Check parent
    """
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray, lower_level_k_epsilon:float,
                 n_states: int, epsilon: float, gamma: float, initial_V_value:float, player_id: int, env):
        
        # Call the parent class constructor to handle all common setup.
        super().__init__(k, action_space, opponent_action_space, lower_level_k_epsilon, n_states, epsilon, gamma, initial_V_value, player_id, env)

    def recalculate_transition_model(self, env):
        """
        Recursively recalculates the transition probability tensor for this agent
        and all agents down its cognitive hierarchy, using the current
        probabilities from the environment.
        """
        # Recalculate this agent's own transition tensor.
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)
        
        # Recursively trigger recalculation for the opponent model.
        if self.k > 1 and self.opponent:
            # Check if is intialized
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            opponent_model = cast(LevelK_TMDP_DP_Agent_NonStationary, self.opponent)
            
            opponent_model.recalculate_transition_model(env)

    def act(self, obs: State, env=None) -> Action:
        """
        Selects an action for the non-stationary case.

        Crucially, it first updates the transition model based on the current
        environment's stochasticity before calculating the optimal action.
        """
        # Get the latest transition probabilities from the environment.
        self.recalculate_transition_model(env)
        
        # Use the parent's `act` method with the updated model.
        return super().act(obs, env)


class LevelK_TMDP_DP_Agent_Dynamic(LevelK_TMDP_DP_Agent_Stationary):
    """
    A Level-K DP agent that learns the environment's transition model online.

    This agent does not assume a fixed transition model. Instead, it learns
    a state-dependent model P(a_self_exec, a_opp_exec | a_self_exec, a_opp_exec , state) by maintaining
    Dirichlet counts for transition outcomes. This makes it suitable for
    environments with unknown transition dynamics.
    (Implementation is identical to the LevelK_MDP_DP_Agent_Dynamic).
    
    Attributes: 
        Check parent
    """
    
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray, lower_level_k_epsilon: float,
                 n_states: int, epsilon: float, gamma: float, initial_V_value: float, player_id: int, env):

        # Call the parent class constructor to handle all common setup.
        super().__init__(k, action_space, opponent_action_space, lower_level_k_epsilon, n_states, 
                         epsilon, gamma, initial_V_value, player_id, env)
        
        # --- Dynamic-Specific Initialization ---
        # `prob_exec_tensor` is not fixed; it will be calculated on-the-fly for each state.

        # 5D tensor to store transition counts (weights) for learning the model.
        # Dims: (state, intended_self, intended_opp, executed_self, executed_opp)
        # We initialize with ones (uniform prior) for Bayesian updating.
        self.prob_exec_tensor = None 
        self.transition_model_weights = np.ones(
            (self.n_states, self.num_self_actions, self.num_opponent_actions, self.num_self_actions, self.num_opponent_actions)
        )

    def _get_probabilities_for_state(self, obs: State) -> np.ndarray:
        """
        Calculates P(a_self_exec, a_opp_exec | a_self_exec, a_opp_exec) for a given state using learned counts.
        
        Returns:
            A 4D np.ndarray with dimensions corresponding to:
            (intended_self, intended_opp, executed_self, executed_opp)
        """
        
        # Get the learned counts for the specific state 'obs'.
        weights_for_obs = self.transition_model_weights[obs, :, :, :, :]
        # Sum over the executed action axes to get total counts for each intended pair.
        # `keepdims=True` ensures the result can be broadcast for division.
        total_counts = np.sum(weights_for_obs, axis=(2, 3), keepdims=True)
        
        # Normalize weights to get probabilities. Handles division by zero by outputting zero.
        prob_tensor_for_obs = np.divide(
            weights_for_obs, total_counts,
            out=np.zeros_like(weights_for_obs),
            where=total_counts != 0
        )
        return prob_tensor_for_obs
    
    def optim_act(self, obs: State) -> Action:
        """
        Selects optimal action using the dynamically learned transition model.
        """

        # Calculate the transition model for the current state based on learned counts.
        self.prob_exec_tensor = self._get_probabilities_for_state(obs)
        
        # call the parent's `optim_act` method, which will use fresh probability of execution tensor.
        return super().optim_act(obs)

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards = None):
        """
        Updates the agent's value function V(s,b) and its internal opponent model
        based on a single transition (s, a, b, s').
        
        Args:
            obs: The current state.
            actions: tuple of agents and opponents actions in state obs
            new_obs: The next state after transition.
        """
        
        # Determine the intended actions from the action pair.
        if self.player_id == 0: 
            intended_action_self, intended_action_opp = actions[0], actions[1]
        else: 
            intended_action_self, intended_action_opp = actions[1], actions[0]
        
        # Lookup which executed actions could have caused the transition s -> new_obs.
        possible_exec_actions_mask = (self.s_prime_lookup[obs] == new_obs)
        exec_self_indices, exec_opp_indices = np.where(possible_exec_actions_mask)
        
        # Update the Dirichlet counts for all plausible executed actions
        for a_self_exec, a_opp_exec in zip(exec_self_indices, exec_opp_indices):
            self.transition_model_weights[obs, intended_action_self, intended_action_opp, a_self_exec, a_opp_exec] += 1
        
        # Get the fresh, state-specific probability tensor *before* updating the value function.
        self.prob_exec_tensor = self._get_probabilities_for_state(obs)
        
        # Call the parent's update method. It will perform the Bellman update
        # using the newly calculated `prob_exec_tensor` and update the opponent model.
        super().update(obs, actions, new_obs, None)