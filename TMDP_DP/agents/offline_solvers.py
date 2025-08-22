import logging
import numpy as np
import math
import multiprocessing
import os
import copy
from tqdm.auto import tqdm
import hashlib
from pathlib import Path 
import filelock

from typing import Optional

from .tmdp_dp import LevelK_TMDP_DP_Agent_Stationary
from .heuristic import ManhattanAgent, ManhattanAgent_Passive
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

class MDP_DP_Agent_PerfectModel(LearningAgent):
    """
    Computes the optimal policy (a best response) against a known, fixed opponent
    policy using offline batch value iteration.

    This agent acts as an offline solver. It first computes the opponent's
    policy for every state. It then combines this policy with the environment's
    base transition dynamics to create a perfect transition model.
    Finally, it runs value iteration on V(s) to find the optimal value function
    and extracts the corresponding deterministic policy.
    """
    def __init__(self, action_space: np.ndarray, n_states: int, gamma: float, initial_V_value: float, player_id: int,
                 termination_criterion: float, value_iteration_max_num_of_iter: int, env, opponent: ManhattanAgent):

        if not isinstance(opponent, ManhattanAgent):
            raise TypeError(f"The provided opponent must be a ManhattanAgent class instance, but got {type(opponent).__name__} instead.")

        if env.enable_push is False and not isinstance(opponent, ManhattanAgent_Passive):
            raise TypeError(f"The provided opponent must be a ManhattanAgent_Passive class instance, but got {type(opponent).__name__} instead.")

        # --- Core Agent Parameters ---
        self.n_states = n_states
        self.gamma = gamma
        self.initial_V_value = initial_V_value
        self.player_id = player_id
        self.action_space = action_space
        self.opponent = copy.deepcopy(opponent) # Store a deep copy of the fixed opponent we are solving against.
        
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
        self.num_opponent_actions = len(action_space)
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
                
        logging.info("Pre-computing the perfect opponent policy table...")
        self.opponent_policy_table = self._precompute_opponent_policy()
        logging.info("Opponent policy table finished.")
        
        # Pre-calculate the state-independent execution probability tensor:
        # P(a_self_exec, a_opp_exec | a_self_intend)
        self.state_dependent_transition_tensor = self._calculate_execution_probabilities(env)
        
        # Value iteration termination critaria
        self.termination_criterion = termination_criterion
        self.value_iteration_max_num_of_iter = value_iteration_max_num_of_iter
        
        # --- Offline Solving Pipeline ---

        # This table will store the final optimal policy π*(s).
        self.optim_policy_table = np.zeros(self.n_states, dtype=int)

        # Run value iteration to find the optimal value function V(s).
        self.run_value_iteration(termination_criterion=self.termination_criterion, value_iteration_max_num_of_iter=self.value_iteration_max_num_of_iter)
        # Extract the final deterministic policy from the converged V-function.
        self._extract_optimal_policy()
        
        
    def _is_terminal_state(self, obs:State) -> bool:
        """
        Checks if a given state is terminal.

        A state is terminal if either player has collected both of their coins
        (a win) or if all four coins have been claimed by any combination of
        players (a draw or a win). This logic is coupled to the environment's
        specific state encoding scheme.
        """
        
        # Grid size N, base for coin collection status
        _, base_coll = self.env_snapshot.grid_size**2, 2
        
        # Radix decoding of the state integer to get coin collection status
        state_copy = obs
        c_r2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll)

        # Check for win conditions
        p0_wins = c_b1 and c_b2
        p1_wins = c_r1 and c_r2
        
        # Check for draw condition
        coins_gone = (c_b1 or c_r1) and (c_b2 or c_r2)
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
        desc_str = f"Pre-computing lookups for DP Agent perfect model (Player {"DM" if self.player_id == 0 else "Adv"})"
        
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
    
    def _reset_sim_env_to_state(self, obs: State):
        """
        Resets the internal simulation environment to a specific state index.
        This involves decoding the integer state into player positions and
        coin collection statuses.
        """
        
        self.env_snapshot.reset()
        # Grid size N, base for coin collection status
        base_pos, base_coll = self.env_snapshot.grid_size**2, 2
        state_copy = obs
        
        # Radix decode the state integer to get all environment components
        c_r2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        p1_flat = state_copy % base_pos
        state_copy //= base_pos
        p0_flat = state_copy
        
        # Set the simulation environment's attributes to match the decoded state
        self.env_snapshot.player_0_pos = np.array([p0_flat % self.env_snapshot.grid_size, p0_flat // self.env_snapshot.grid_size])
        self.env_snapshot.player_1_pos = np.array([p1_flat % self.env_snapshot.grid_size, p1_flat // self.env_snapshot.grid_size])
        self.env_snapshot.player_0_collected_coin0, self.env_snapshot.player_0_collected_coin1 = c_b1, c_b2
        self.env_snapshot.player_1_collected_coin0, self.env_snapshot.player_1_collected_coin1 = c_r1, c_r2
        self.env_snapshot.coin0_available = not (c_b1 or c_r1)
        self.env_snapshot.coin1_available = not (c_b2 or c_r2)

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
        
        V = np.ones([self.n_states])*initial_V_value
        
        # tqdm shows progress bar.
        for s in tqdm(range(self.n_states), desc="Initializing value function."):
            if self._is_terminal_state(s):
                V[s] = 0
        
        return V
    
    def _precompute_opponent_policy(self):
        """
        Computes and stores the opponent's action probability distribution for
        every state in the environment.

        It iterates through all states, queries the provided opponent agent for its
        action distribution, and stores it in a table.

        Returns:
            np.ndarray: A policy table of shape (n_states, num_opponent_actions).
        """
        policy_table = np.zeros((self.n_states, self.num_opponent_actions))
        for s in tqdm(range(self.n_states), desc="Pre-computing opponent policy table"):
            try:
                # The provided opponent's `act` method is expected to determine
                # its set of preferred actions. Here, we rely on the implementation
                # detail that it sets a `optimal_actions` attribute defining all
                # equaly optimal actions of the opponent.
                self.opponent.act(s, None)
                optimal_actions = self.opponent.optimal_actions_cache

                if optimal_actions is not None and len(optimal_actions) > 0:
                    # Assume the opponent chooses uniformly among its possible optimal actions.
                    policy_table[s, optimal_actions] = 1.0 / len(optimal_actions)
                else: 
                    # Handle cases where no specific action is determined (e.g., in a terminal state).
                    # Assume a uniform random policy as a fallback.
                    policy_table[s, :] = 1.0 / self.num_opponent_actions
            except (ValueError, IndexError): # Catch errors from invalid/unreachable states
                policy_table[s, :] = 1.0 / self.num_opponent_actions
        return policy_table
    
    def _calculate_execution_probabilities(self, env) -> np.ndarray:
        """
        Calculates the state-dependent transition tensor P(executed | intended, state).

        This is a two-step process:
        1. Calculate the state-independent P(executed | intended) tensor based on
           the environment's raw action success probabilities.
        2. Marginalize out the opponent's intended action by multiplying with their
           pre-computed, state-dependent policy P(opp_intended | state).

        Returns:
            A 4D np.ndarray with dimensions corresponding to:
            (state, intended_self, executed_self, executed_opp)
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
        
        probab_exec_tensor = prob_self[:, np.newaxis, :, np.newaxis] * prob_opp[np.newaxis, :, np.newaxis, :]
        
        # --- Calculate opponent action independent and state dependent probability execution tensor ---
        # \sum_{a_o_int}P(a_s_exec, a_o_exec | a_s_int, a_o_int)p(a_o_int| s)
        state_dependent_transition_tensor = np.einsum(
            'ijlk, sj -> silk', 
            probab_exec_tensor, 
            self.opponent_policy_table, 
            optimize=True
        )
        
        return state_dependent_transition_tensor
    
    def _extract_optimal_policy(self):
        """
        Extracts the optimal deterministic policy π*(s) after value iteration by performing one extra value iteration sweep
        extracting the agents action which maximises the value in each state.
        """
        
        logging.info("Extracting optimal policy...")
        for s in range(self.n_states):
            if not self._is_terminal_state(s):
                rewards_executed = self.r_lookup[s, :, :]
                s_primes_executed = self.s_prime_lookup[s, :, :]
                
                prob_tensor_for_state_s = self.state_dependent_transition_tensor[s, :, :, :]
                
                q_values_for_actions = np.einsum(
                    'ikl,kl->i', prob_tensor_for_state_s,
                    rewards_executed + self.gamma * self.V[s_primes_executed], optimize=True
                )
                self.optim_policy_table[s] = np.argmax(q_values_for_actions)
        logging.info("Optimal policy extracted.")


    def run_value_iteration(self, termination_criterion: float, value_iteration_max_num_of_iter: int):
        """
        Performs offline, in-place (asynchronous) value iteration on the V(s)
        table until convergence.

        "In-place" means the V-table is updated directly in each sweep, which is
        more memory-efficient than creating a new copy of V for each iteration.

        Args:
            termination_criterion (float): The convergence threshold. Iteration stops when the
                           maximum change in the value function is less than termination_criterion.
            value_iteration_max_num_of_iter (int): The maximum number of iterations to perform.
        """
        logging.info("Starting offline in-place value iteration on V(s)...")
        
        progress_bar = tqdm(range(value_iteration_max_num_of_iter), desc="Value Iteration", unit="it")
        
        for i in progress_bar:
            delta = 0  # Tracks the maximum change in V(s) during a sweep.

            # Loop through all states to update their values.
            for s in range(self.n_states):
                
                if self._is_terminal_state(s):
                    continue  # The value of terminal states is always 0.
                
                # Store the value of V(s) before the update
                v_s_old = self.V[s]
                
                # Get pre-computed outcomes from the lookup tables for state 's'.
                rewards_executed = self.r_lookup[s, :, :]
                s_primes_executed = self.s_prime_lookup[s, :, :]
                
                # Extract probability of transition for the current state s
                prob_tensor_for_state_s = self.state_dependent_transition_tensor[s, :, :, :]
                
                q_values_for_actions = np.einsum('ikl,kl->i', prob_tensor_for_state_s ,rewards_executed + self.gamma * self.V[s_primes_executed], optimize=True)
                
                # The new value is the maximum Q-value over our possible actions.
                v_s_new = np.max(q_values_for_actions)
                
                # --- Update V in-place and track the change ---
                self.V[s] = v_s_new
                delta = max(delta, np.abs(v_s_new - v_s_old))
                    
            # Update the progress bar with the current delta.       
            progress_bar.set_postfix(delta=f"{delta:.6f}/{termination_criterion:.4f}", refresh=True)
            
            # --- Convergence Check and Policy extraction ---
            if delta < termination_criterion:
                logging.info(f"\nValue iteration converged after {i + 1} iterations.")
                # Final policy extraction pass
                return
            
        logging.info(f"Value iteration did not converge after {value_iteration_max_num_of_iter} iterations.")

    def act(self, obs: State, env=None) -> Action:
        """
        Returns the pre-computed optimal action for the given state.

        Args:
            obs: The current state observation.
            env: The environment (not used in this offline agent).

        Returns:
            The optimal action for the given state.
        """
        
        return self.optim_policy_table[obs]

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards: Optional[tuple[Reward, Reward]]):
        """
        This agent is an offline solver and does not learn during online interaction.
        This method is implemented to satisfy the agent interface but performs no action.
        """
        pass
    
    def update_epsilon(self, new_epsilon_agent: float, new_epsilon_lower_k_level: Optional[float]):
        """
        This agent is an offline solver, therfore it does not utilise exploration-strategy.
        """
        pass
        


class TMDP_DP_Agent_PerfectModel(LevelK_TMDP_DP_Agent_Stationary):
    """
    Computes the optimal policy (a best response) against a known, fixed opponent
    policy using offline value iteration.

    This agent acts as an offline solver. It utilises direct policy of a
    specific, non-learning opponent (e.g., a hard-coded heuristst builds a perfect agent). Then, it
    runs value iteration until convergence to find the optimal value function
    against that opponent's fixed policy. Finally, it extracts a deterministic
    optimal policy.

    It inherits from `LevelK_TMDP_DP_Agent_Stationary` to reuse the efficient,
    vectorized model-handling and Bellman update logic.

    Attributes:
        opponent (object): A deep copy of the fixed opponent agent this agent is
                           solving a best response for.
        opponent_policy_table (np.ndarray): A pre-computed table of shape
                                           (n_states, num_opponent_actions)
                                           storing the opponent's fixed policy.
        optim_policy_table (np.ndarray): A table of shape (n_states,) storing the
                                         final, optimal action for each state.
                                         
        For other atribute description check parent.
    """
    def __init__(self, action_space: np.ndarray, opponent_action_space: np.ndarray, n_states: int, gamma: float, initial_V_value: float, player_id: int,
                 termination_criterion: float, value_iteration_max_num_of_iter: int, env, opponent: ManhattanAgent):
        # Initialize as a Level-1 DP Agent to leverage its pre-computation,
        # lookup tables, and vectorized calculation methods.
        # k=1 is sufficient as we are pre-solving the optimal policy.
        # epsilon=0 is used because the final policy will be purely deterministic
        # and there is no exploration during online interaction.
        
        if not isinstance(opponent, ManhattanAgent):
            raise TypeError(f"The provided opponent must be a ManhattanAgent class instance, but got {type(opponent).__name__} instead.")
        
        if env.enable_push is False and not isinstance(opponent, ManhattanAgent_Passive):
            raise TypeError(f"The provided opponent must be a ManhattanAgent_Passive class instance, but got {type(opponent).__name__} instead.")
        
        super().__init__(k=1, action_space=action_space, opponent_action_space=opponent_action_space, lower_level_k_epsilon=0,
                         n_states=n_states, epsilon=0, gamma=gamma, initial_V_value=initial_V_value,
                         player_id=player_id, env=env)

        # Agent specific initialisation
        self.termination_criterion = termination_criterion
        self.value_iteration_max_num_of_iter = value_iteration_max_num_of_iter
        
        # Store a deep copy of the fixed opponent we are solving against.
        self.opponent = copy.deepcopy(opponent)

        # --- Offline Solving Pipeline ---
        # Pre-compute the opponent's full, fixed policy for every state.
        logging.info("Pre-computing the perfect opponent policy table...")
        self.opponent_policy_table = self._precompute_opponent_policy()
        logging.info("Opponent policy table finished.")

        # This table will store the final optimal policy π*(s).
        self.optim_policy_table = np.zeros(self.n_states, dtype=int)

        # Run value iteration to find the optimal value function V*(s, b).
        self.run_value_iteration(termination_criterion=self.termination_criterion, value_iteration_max_num_of_iter=self.value_iteration_max_num_of_iter)
        # Extract the final deterministic policy from the converged V-function.
        self._extract_optimal_policy()
        

    def _precompute_opponent_policy(self):
        """
        Computes and stores the opponent's action probability distribution for
        every state in the environment.

        It iterates through all states, queries the provided opponent agent for its
        action distribution, and stores it in a table.

        Returns:
            np.ndarray: A policy table of shape (n_states, num_opponent_actions).
        """
        policy_table = np.zeros((self.n_states, self.num_opponent_actions))
        for s in tqdm(range(self.n_states), desc="Pre-computing opponent policy table"):
            try:
                # The provided opponent's `act` method is expected to determine
                # its set of preferred actions. Here, we rely on the implementation
                # detail that it sets a `optimal_actions` attribute defining all
                # equaly optimal actions of the opponent.
                self.opponent.act(s, None)
                optimal_actions = self.opponent.optimal_actions_cache

                if optimal_actions is not None and len(optimal_actions) > 0:
                    # Assume the opponent chooses uniformly among its possible optimal actions.
                    policy_table[s, optimal_actions] = 1.0 / len(optimal_actions)
                else: 
                    # Handle cases where no specific action is determined (e.g., in a terminal state).
                    # Assume a uniform random policy as a fallback.
                    policy_table[s, :] = 1.0 / self.num_opponent_actions
            except (ValueError, IndexError): # Catch errors from invalid/unreachable states
                policy_table[s, :] = 1.0 / self.num_opponent_actions
        return policy_table

    def get_opponent_policy(self, obs: State):
        """
        Overrides the parent method. Instead of learning or using a recursive
        model, it simply returns the pre-computed policy for the fixed opponent
        from the lookup table.

        Args:
            obs: The current state.

        Returns:
            An array representing the opponent's action probability distribution.
        """
        return self.opponent_policy_table[obs]

    def run_value_iteration(self, termination_criterion: float, value_iteration_max_num_of_iter: int):
        """
        Performs offline, in-place (asynchronous) value iteration on the V(s, b)
        table until convergence.

        "In-place" means the V-table is updated directly in each sweep, which is
        more memory-efficient than creating a new copy of V for each iteration.

        Args:
            termination_criterion (float): The convergence threshold. Iteration stops when the
                           maximum change in the value function is less than termination_criterion.
            value_iteration_max_num_of_iter (int): The maximum number of iterations to perform.
        """
        logging.info("Starting offline in-place value iteration on V(s, b)...")
        
        progress_bar = tqdm(range(value_iteration_max_num_of_iter), desc="Value Iteration", unit="it")
        
        for i in progress_bar:
            delta = 0  # Tracks the maximum change in V(s,b) during a sweep.

            # Loop through all states to update their values.
            for s in range(self.n_states):
                
                if self._is_terminal_state(s):
                    continue  # The value of terminal states is always 0.
                
                # Get pre-computed outcomes from the lookup tables for state 's'.
                rewards_executed = self.r_lookup[s, :, :]
                s_primes_executed = self.s_prime_lookup[s, :, :]
                
                # Calculate expected future values for all possible resulting states.
                # Note: This step uses the perfect opponent model via get_opponent_policy.
                future_V_values_executed = self._calculate_expected_future_values(s_primes_executed)
                
                # Loop through all possible opponent actions `b` to update V(s,b).
                for opponent_action in range(self.num_opponent_actions):
                    
                    # Store the value of V(s, opponent_action) before the update
                    v_s_b_old = self.V[s, opponent_action]
                    
                    # Get transition probabilities conditioned on the opponent's INTENDED action.
                    prob_exec_tensor_fixed_opp_action = self.prob_exec_tensor[:, opponent_action, :, :]
                    
                    # Calculate expected rewards and future values for each of agents INTENDED actions.
                    expected_rewards_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, rewards_executed, optimize=True)
                    expected_future_V_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, future_V_values_executed, optimize=True)
                    
                    # Q(s, a | b) = E[R(s,a,b)] + gamma * E[V(s')]
                    q_values_for_actions = expected_rewards_per_action + self.gamma * expected_future_V_per_action
                    
                    # The new value is the maximum Q-value over our possible actions.
                    v_s_b_new = np.max(q_values_for_actions)

                    # --- Update V in-place and track the change ---
                    self.V[s, opponent_action] = v_s_b_new
                    delta = max(delta, np.abs(v_s_b_new - v_s_b_old))
                    
            # Update the progress bar with the current delta.       
            progress_bar.set_postfix(delta=f"{delta:.6f}/{termination_criterion:.4f}", refresh=True)
            
            # --- Convergence Check ---
            if delta < termination_criterion:
                logging.info(f"\nValue iteration converged after {i + 1} iterations.")
                return

        logging.info(f"Value iteration did not converge after {value_iteration_max_num_of_iter} iterations.")


    def _extract_optimal_policy(self):
        """
        Extracts the optimal deterministic policy π*(s) after value iteration.

        For each state, it calculates the best action for the agent, assuming the
        opponent plays according to its pre-computed fixed policy. This reuses
        the parent's `optim_act` method, which performs exactly this calculation.
        """
        
        logging.info("Extracting optimal policy...")
        for s in range(self.n_states):
            # The parent's optim_act function finds the action 'a' that maximizes
            # the expected Q-value, marginalized over the opponent's policy at state 's'.
            # This is exactly what we need to determine the best response at each state.
            self.optim_policy_table[s] = self.optim_act(s)
        logging.info("Optimal policy extracted.")

    def act(self, obs: State, env=None) -> Action:
        """
        Returns the pre-computed optimal action for the given state.

        Args:
            obs: The current state observation.
            env: The environment (not used in this offline agent).

        Returns:
            The optimal action for the given state.
        """
        
        return self.optim_policy_table[obs]

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards: Optional[tuple[Reward, Reward]] = None):
        """
        This agent is an offline solver and does not learn during online interaction.
        This method is implemented to satisfy the agent interface but performs no action.
        """
        pass

    def update_epsilon(self, new_epsilon_agent: float, new_epsilon_lower_k_level: Optional[float]):
        """
        This agent is an offline solver, therfore it does not utilise exploration-strategy.
        """
        pass
        