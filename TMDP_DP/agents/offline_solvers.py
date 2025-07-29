import logging
import numpy as np
import copy
from tqdm.notebook import tqdm

from typing import Optional

from .tmdp_dp import LevelK_TMDP_DP_Agent_Stationary
from .heuristic import ManhattanAgent
from .base import LearningAgent

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
ValueFunction = np.ndarray
ActionDetails = np.ndarray


class DPAgent_PerfectModel(LearningAgent):
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
        # Determine action spaces and details based on player_id

        self.self_action_details = self.env_snapshot.combined_actions
        self.opponent_action_details = self.env_snapshot.combined_actions
        self.num_self_actions = len(env.combined_actions)
        self.num_opponent_actions = len(env.combined_actions)
        self.self_available_move_actions_num = len(env.available_move_actions)
        self.opponent_available_move_actions_num = len(env.available_move_actions)
        
        # --- Value Function and Model Initialization ---
        # V(s, opponent_action): The value of being in state 's' *after* 
        # the opponent has committed to taking 'opponent_action'.
        self.V = self._setup_value_function(self.initial_V_value)
        
        # Pre-compute (s, a_self_executed, a_opp_executed) -> (s', r) lookup tables.
        # These tensors store the outcomes for every state and *executed* action pair
        # mapped to resulting s'. 
        self.s_prime_lookup, self.r_lookup  = self._precompute_lookups()
        
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
        Builds lookup tables for next states (s') and rewards (r).
        
        This is a one-time, expensive computation that maps every
        (state, executed_action_self, executed_action_opponent) tuple
        to its deterministic outcome. This avoids repeated simulation calls
        during the learning process, significantly speeding up calculations.

        Returns:
            A tuple containing:
            - s_prime_lookup (np.ndarray): Table of next states.
            - r_lookup (np.ndarray): Table of rewards for the current agent.
        """
        
        desc_str = f"Pre-computing lookups for DP Agent perfect model (Player {"DM" if self.player_id == 0 else "Adv"})"
        
        s_prime_lookup = np.zeros((self.n_states, self.num_self_actions, self.num_opponent_actions), dtype=int)
        r_lookup = np.zeros((self.n_states, self.num_self_actions, self.num_opponent_actions), dtype=float)

        for s in tqdm(range(self.n_states), desc=desc_str):
            try:
                # Set the deterministic simulation environment to state 's'
                self._reset_sim_env_to_state(s)
            except (IndexError, ValueError):
                continue # Skip unreachable/invalid states
            
            for a_self_exec in range(self.num_self_actions):
                for a_opp_exec in range(self.num_opponent_actions):
                    # Save the environment's state to restore it after the step
                    current_env_state = self.env_snapshot.get_state()
                    
                    # The environment engine expects actions in [0, 1] order.
                    if self.player_id == 0:
                        action_pair = (a_self_exec, a_opp_exec)
                    else:
                        action_pair = (a_opp_exec, a_self_exec)

                    # Simulate one step with the specified executed action pair
                    s_prime, rewards_vec, _ = self.env_snapshot.step(action_pair)
                    
                    # Store the resulting next state and the reward for this agent
                    s_prime_lookup[s, a_self_exec, a_opp_exec] = s_prime
                    r_lookup[s, a_self_exec, a_opp_exec] = rewards_vec[self.player_id]
                    
                    # Restore environment to its original state for the next action pair
                    self._reset_sim_env_to_state(current_env_state)
                    
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
                    'ilk,lk->i', prob_tensor_for_state_s,
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
                
                q_values_for_actions = np.einsum('ikl,lk->i', prob_tensor_for_state_s ,rewards_executed + self.gamma * self.V[s_primes_executed], optimize=True)
                
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
        


class TMDP_DPAgent_PerfectModel(LevelK_TMDP_DP_Agent_Stationary):
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
        