import numpy as np
from numpy.random import choice
import copy
from typing import Optional, Tuple, cast
from tqdm.notebook import tqdm


from .base import Agent

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
ValueFunction = np.ndarray
ActionDetails = np.ndarray

class _BaseLevelKDPAgent(Agent):
    """
    Base class for Level-K Dynamic Programming agents.

    This class handles the common, expensive setup procedures required by all
    Level-K DP agents, such as pre-computing the environment's transition
    and reward dynamics. Subclasses should implement the specific opponent
    modeling logic.

    Attributes:
        k (int): The cognitive level of the agent.
        n_states (int): The total number of states in the environment.
        epsilon (float): The initial exploration rate for epsilon-greedy policies.
        gamma (float): The discount factor for future rewards.
        player_id (int): The agent's ID (0 or 1).
        opponent_action_space (np.ndarray): The set of actions available to the opponent.
        env_snapshot (CoinGame): A deep copy of the environment for deterministic simulation.
        V (ValueFunction): The agent's value function table, V(s, b), where b is the opponent's action.
        s_prime_lookup (np.ndarray): Pre-computed table mapping (s, a_self_executed, a_opp_executed) -> s'.
        r_lookup (np.ndarray): Pre-computed table mapping (s, a_self_executed, a_opp_executed) -> r.
        self_action_details (ActionDetails): Details of the agent's combined (move, push) actions.
        opponent_action_details (ActionDetails): Details of the opponent's combined (move, push) actions.
        num_self_actions (int): The number of actions available to this agent.
        num_opponent_actions (int): The number of actions available to the opponent.
        opponent (Optional[Agent]): The agent's internal model of its opponent.
        dirichlet_counts (Optional[np.ndarray]): Dirichlet counts for modeling policy of a Level-0 opponent.
    """
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray,
                 n_states: int, epsilon: float, gamma: float, player_id: int, env):
        
        super().__init__(action_space)
        
        # --- Core Agent Parameters ---
        self.k = k
        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.player_id = player_id
        self.opponent_action_space = opponent_action_space
        
        # --- Environment Snapshot for Model Computation ---
        # Create a deterministic copy of the environment for simulation.
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.blue_player_execution_prob = 1.0
        self.env_snapshot.red_player_execution_prob = 1.0
        
        # ---Action Setup ---
        # Determine action spaces and details based on player_id
        if self.player_id == 0: # This agent is the Blue Player
            self.self_action_details = self.env_snapshot.combined_actions_blue
            self.opponent_action_details = self.env_snapshot.combined_actions_red
            self.num_self_actions = len(env.combined_actions_blue)
            self.num_opponent_actions = len(env.combined_actions_red)
            self.self_available_move_actions_num = len(env.available_move_actions_DM)
            self.opponent_available_move_actions_num = len(env.available_move_actions_Adv)
        else: # This agent is the Red Player
            self.self_action_details = self.env_snapshot.combined_actions_red
            self.opponent_action_details = self.env_snapshot.combined_actions_blue
            self.num_self_actions = len(env.combined_actions_red)
            self.num_opponent_actions = len(env.combined_actions_blue)
            self.self_available_move_actions_num = len(env.available_move_actions_Adv)
            self.opponent_available_move_actions_num = len(env.available_move_actions_DM)
            
        # --- Value Function and Model Initialization ---
        # V(s, opponent_action), where opponent_action is the opponent's selected action.
        self.V = self._setup_value_function(0)
            
        # Pre-compute (s, a_self_executed, a_opp_executed) -> (s', r) lookup tables.
        # These tensors store the outcomes for every state and EXECUTED action pair.
        self.s_prime_lookup, self.r_lookup  = self._precompute_lookups()
        
        # --- Opponent Model Initialization (handled by subclasses) ---
        self.opponent: Optional[Agent] = None
        self.dirichlet_counts: Optional[np.ndarray] = None
        
    def _setup_value_function(self, initial_value: float) -> ValueFunction:
        """Initializes the value function V(s, opponent_action).
        
        Sets the value of all terminal states to 0.

        Args:
            initial_value (float): The initial value for non-terminal states.

        Returns:
            np.ndarray: The initialized value function table.
        """
        
        V = np.ones([self.n_states, len(self.opponent_action_space)])*initial_value
        
        # tqdm shows progress during this potentially long setup.
        for s in tqdm(range(self.n_states), desc="Initializing value function."):
            if self._is_terminal_state(s):
                V[s,:] = 0
                
        return V
            
    def _is_terminal_state(self, obs:State) -> bool:
        """Checks if a given state is terminal (i.e., both coins are collected).
        
        Note: This logic is coupled to the environment's state encoding.
        A future refactor could move this into the environment class itself.
        """
        # We can use the environment snapshot to get the grid parameters
        _, base_coll = self.env_snapshot.N**2, 2
        
        # Implement radix decoding of state only for coins
        state_copy = obs
        c_r2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll)

        # Check for win conditions
        blue_wins = c_b1 and c_b2
        red_wins = c_r1 and c_r2
        
        # Check for draw condition
        coins_gone = (c_b1 or c_r1) and (c_b2 or c_r2)
        is_draw = coins_gone and not blue_wins and not red_wins
        
        # A state is terminal if a player has won or if it's a draw.
        # The max_steps condition cannot be encoded in the state, so the DP agent
        # cannot account for it, which is a limitation of this state representation.
        return blue_wins or red_wins or is_draw

    def _precompute_lookups(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds lookup tables for next states (s') and rewards (r).
        
        This is a one-time, expensive computation that maps every
        (state, executed_action_self, executed_action_opponent) tuple
        to its deterministic outcome, avoiding repeated simulation calls
        during training.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 
            s_prime_lookup and r_lookup tables.
        """
        
        desc_str = f"Pre-computing lookups for Level-{self.k} DP Agent (Player {"DM" if self.player_id == 0 else "Adv"})"
        
        s_prime_lookup = np.zeros((self.n_states, self.num_self_actions, self.num_opponent_actions), dtype=int)
        r_lookup = np.zeros((self.n_states, self.num_self_actions, self.num_opponent_actions), dtype=float)

        for s in tqdm(range(self.n_states), desc=desc_str):
            try:
                self._reset_sim_env_to_state(s)
            except (IndexError, ValueError):
                continue # Skip unreachable/invalid states

            for a_self_exec in range(self.num_self_actions):
                for a_opp_exec in range(self.num_opponent_actions):
                    # Save state before stepping
                    current_env_state = self.env_snapshot.get_state()
                    
                    # Create the action pair in the correct [blue, red] order for the engine
                    if self.player_id == 0: # I am the Blue Player (DM)
                        action_pair = (a_self_exec, a_opp_exec)
                    else: # I am the Red Player (Adv)
                        action_pair = (a_opp_exec, a_self_exec)

                    # Simulate one step with the correctly ordered action pair
                    s_prime, rewards_vec, _ = self.env_snapshot.step(action_pair)
                    
                    # Store the resulting state and reward
                    s_prime_lookup[s, a_self_exec, a_opp_exec] = s_prime
                    r_lookup[s, a_self_exec, a_opp_exec] = rewards_vec[self.player_id]
                    
                    # Restore environment to the original state for the next action pair
                    self._reset_sim_env_to_state(current_env_state)
                    
        return s_prime_lookup, r_lookup
    
    def _reset_sim_env_to_state(self, obs: State):
        """Resets the internal simulation environment to a specific state."""
        
        self.env_snapshot.reset()
        base_pos, base_coll = self.env_snapshot.N**2, 2
        state_copy = obs
        
        # Radix decode the state
        c_r2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll)
        state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll)
        state_copy //= base_coll
        p2_flat = state_copy % base_pos
        state_copy //= base_pos
        p1_flat = state_copy
        
        self.env_snapshot.blue_player = np.array([p1_flat % self.env_snapshot.N, p1_flat // self.env_snapshot.N])
        self.env_snapshot.red_player = np.array([p2_flat % self.env_snapshot.N, p2_flat // self.env_snapshot.N])
        self.env_snapshot.blue_collected_coin1, self.env_snapshot.blue_collected_coin2 = c_b1, c_b2
        self.env_snapshot.red_collected_coin1, self.env_snapshot.red_collected_coin2 = c_r1, c_r2
        self.env_snapshot.coin1_available = not (c_b1 or c_r1)
        self.env_snapshot.coin2_available = not (c_b2 or c_r2)
    
    def update_epsilon(self, new_epsilon: float):
        """Updates the exploration rate for this agent and its recursive opponent model."""
        
        self.epsilon = new_epsilon
        if self.k > 1 and self.opponent:
            self.opponent.update_epsilon(new_epsilon)
    
class LevelKDPAgent_Stationary(_BaseLevelKDPAgent):
    """
    A Level-K DP agent that assumes a stationary environment.

    This agent pre-computes a fixed transition probability tensor based on the
    environment's initial action execution probabilities. It models its opponent
    recursively as a Level-(k-1) stationary agent.
    """

    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray,
                 n_states: int, epsilon: float, gamma: float, player_id: int, env):
        if k < 1:
            raise ValueError("Level k must be a positive integer.")

        # Call the base class constructor to handle all common setup.
        super().__init__(k, action_space, opponent_action_space, n_states, 
                         epsilon, gamma, player_id, env)
        
        # Pre-calculate the state-independent execution probability tensor P(a_self_exec, a_opp_exec | intended_action_self, intended_action_opp)
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)
        
        # --- Opponent Model Initialization ---
        if self.k == 1:
            # Base Case (k=1): Model opponent as Level-0 (random) using Dirichlet counts.
            self.dirichlet_counts = np.ones((self.n_states, len(self.opponent_action_space)))
        elif self.k > 1:
            # Recursive Step (k>1): Opponent is a Level-(k-1) version of this same class.
            self.opponent = self.__class__( 
                k=self.k - 1,
                action_space=self.opponent_action_space,
                opponent_action_space=self.action_space,
                n_states=self.n_states,
                epsilon=self.epsilon,
                gamma=self.gamma,
                player_id=1 - self.player_id,
                env=env
            )
            
    def get_opponent_policy(self, obs: State) -> Policy:
        """
        Estimates the opponent's policy (probability distribution over actions) for a given state.
        """
        
        # Level-1 models opponent policy from Dirichlet counts
        if self.k == 1:
            # Check if atributes is still initalized to None
            assert self.dirichlet_counts is not None, "dirichlet_counts should be initialized for a Level-1 agent."
            
            dir_sum = np.sum(self.dirichlet_counts[obs])
            if dir_sum == 0:
                # Return a uniform policy if the state has not been seen
                return np.ones(self.num_opponent_actions) / self.num_opponent_actions
            return self.dirichlet_counts[obs] / dir_sum
        else: # k > 1
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            
            # Level-k recursively asks its internal Level-(k-1) model for its policy
            opponent_model = cast(LevelKDPAgent_Stationary, self.opponent)
            opponent_opt_act = opponent_model.optim_act(obs)
            
            opponent_policy = np.zeros(self.num_opponent_actions)
            
            if self.num_opponent_actions > 1:
                
                # Construct an epsilon-greedy policy for the opponent
                
                prob_non_optimal = self.epsilon / self.num_opponent_actions
                opponent_policy[:] = prob_non_optimal
                opponent_policy[opponent_opt_act] += 1.0 - self.epsilon
            else:
                opponent_policy[opponent_opt_act] = 1.0
            return opponent_policy
            
    def _calculate_expected_future_values(self, s_prime_array: np.ndarray) -> np.ndarray:
        """
        Efficiently calculates E_{p(b'|s')}[V(s', b')] for an array of next states.
        """
        # Find unique next states to avoid redundant calculations
        unique_s_primes, inverse_indices = np.unique(s_prime_array, return_inverse=True)
        
        # Calculate the expected value for each unique next state
        expected_V_map = {
            s_prime: np.dot(self.V[s_prime, :], self.get_opponent_policy(s_prime))
            for s_prime in unique_s_primes
        }
        
        # Map the computed values back to the original array shape using the inverse indices
        expected_V_flat = np.array([expected_V_map.get(s, 0) for s in unique_s_primes])
        return expected_V_flat[inverse_indices].reshape(s_prime_array.shape)

    def optim_act(self, obs: State) -> Action:
        """Selects the optimal action based on the pre-computed model using vectorized operations."""
        # Get outcomes for the current state from the pre-computed lookup tables
        rewards_executed = self.r_lookup[obs, :, :]
        s_primes_executed = self.s_prime_lookup[obs, :, :]
        
        # Calculate expected future values for all possible executed outcomes
        future_V_executed = self._calculate_expected_future_values(s_primes_executed)

        # Expected immediate rewards for each INTENDED action pair, averaged over execution stochasticity.
        # einsum computes the dot product over the last two axes (executed actions).
        expected_self_rewards_intended = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, rewards_executed)
        
        # Expected future values for each INTENDED action pair.
        weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, future_V_executed)
        
        # Get opponent's predicted policy for the current state
        opponent_policy_in_obs = self.get_opponent_policy(obs)
        
        # Calculate total value for each of our actions, marginalized over the opponent's policy
        total_action_values = np.dot(expected_self_rewards_intended, opponent_policy_in_obs) + \
                              self.gamma * np.dot(weighted_sum_future_V, opponent_policy_in_obs)
        
        # Choose the best action, breaking ties randomly
        max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))
        return np.random.choice(max_indices)
        
    def get_policy(self, obs: State) -> Policy:
        """Calculates this agent's own epsilon-greedy policy distribution."""
        
        # Calculate optimal action in state obs
        optimal_action_idx = self.optim_act(obs)
        num_actions = len(self.action_space)
        
        policy = np.full(num_actions, self.epsilon / num_actions)
        policy[optimal_action_idx] += 1.0 - self.epsilon
        return policy / np.sum(policy)

    def act(self, obs:State, env=None) -> Action:
        """Selects an action based on the epsilon-greedy policy."""
        policy = self.get_policy(obs)
        return choice(self.action_space, p=policy)

    def update(self, obs: State, actions: list[Action], 
               rewards, new_obs: State):
        """Updates the agent's value function V(s,b) and its internal opponent model."""
        
        # Determine opponent's observed action
        opponent_action_taken = actions[1] if self.player_id == 0 else actions[0]
        
        # Update opponent model (recursively for k>1, or Dirichlet for k=1)
        if self.k > 1:
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            self.opponent.update(obs, actions, rewards, new_obs)
        else:
            assert self.dirichlet_counts is not None, "dirichlet_counts should be initialized for a Level-1 agent."
            self.dirichlet_counts[obs, opponent_action_taken] += 1
        
        # If terminal state is reached do not update the value  
        if self._is_terminal_state(obs):
            return
            
        # --- Vectorized V(obs, opponent_action) Update ---
        
        # Get outcomes for the current state from the lookup tables
        rewards_executed = self.r_lookup[obs, :, :]
        s_primes_executed = self.s_prime_lookup[obs, :, :]
        
        # Calculate expected future values for all outcomes
        future_V_executed = self._calculate_expected_future_values(s_primes_executed)
        
        # Extract transition probabilities conditioned on the opponent's TAKEN action
        prob_exec_tensor_fixed_opp_action = self.prob_exec_tensor[:, opponent_action_taken, :, :]
        
        # Calculate expected rewards and future values for each of our INTENDED actions
        expected_rewards_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, rewards_executed)
        expected_future_V_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, future_V_executed)
        
        # Bellman update for the value function
        q_values_for_actions = expected_rewards_per_action + self.gamma * expected_future_V_per_action
        self.V[obs, opponent_action_taken] = np.max(q_values_for_actions)
        
    def _calculate_execution_probabilities(self, env) -> np.ndarray:
        """Calculates the state-independent transition tensor P(executed | intended)."""
        
        if self.player_id == 0:
            self_exec_prob, opp_exec_prob = env.blue_player_execution_prob, env.red_player_execution_prob
        else:
            self_exec_prob, opp_exec_prob = env.red_player_execution_prob, env.blue_player_execution_prob
        
        # Extract move and push actions from the detailes
        self_moves, self_pushes = self.self_action_details[:, 0], self.self_action_details[:, 1]
        opp_moves, opp_pushes = self.opponent_action_details[:, 0], self.opponent_action_details[:, 1]

        # Probability of our executed move given our intended move.
        prob_self = np.zeros((self.num_self_actions, self.num_self_actions))
        move_match = (self_moves[:, None] == self_moves[None, :])
        push_match = (self_pushes[:, None] == self_pushes[None, :])
        prob_self[move_match & push_match] = self_exec_prob
        if self.self_available_move_actions_num > 1:
            prob_self[~move_match & push_match] = (1.0 - self_exec_prob) / (self.self_available_move_actions_num - 1)
        
        # Probability of opponent's executed move given their intended move.
        prob_opp = np.zeros((self.num_opponent_actions, self.num_opponent_actions))
        move_match = (opp_moves[:, None] == opp_moves[None, :])
        push_match = (opp_pushes[:, None] == opp_pushes[None, :])
        prob_opp[move_match & push_match] = opp_exec_prob
        if self.opponent_available_move_actions_num > 1:
            prob_opp[~move_match & push_match] = (1.0 - opp_exec_prob) / (self.opponent_available_move_actions_num - 1)
            
        # Combine into a 4D tensor: P(a_self_exec, a_opp_exec | a_self_intend, a_opp_intend)
        # using the outer product.
        return prob_self[:, np.newaxis, :, np.newaxis] * prob_opp[np.newaxis, :, np.newaxis, :]


class LevelKDPAgent_NonStationary(LevelKDPAgent_Stationary):
    """
    A Level-K DP agent for non-stationary environments where action probabilities can change.
    
    This agent recalculates the execution probability tensor before every action selection,
    making it adaptable to environments with dynamic transition stochasticity.
    """
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray, n_states: int, epsilon: float, gamma: float, player_id: int, env):
        
        # Initializes everything except the opponent model from the base class.
        super().__init__(k, action_space, opponent_action_space, n_states, epsilon, gamma, player_id, env)

        self.prob_exec_tensor = self._calculate_execution_probabilities(env) # Initial calculation

        # Initalize k-level hierarchy
        if self.k == 1:
            self.dirichlet_counts = np.ones((self.n_states, len(self.opponent_action_space)))
        elif self.k > 1:
            self.opponent = LevelKDPAgent_NonStationary(
                k=self.k - 1,
                action_space=self.opponent_action_space,
                opponent_action_space=self.action_space,
                n_states=self.n_states,
                epsilon=self.epsilon,
                gamma=self.gamma,
                player_id=1 - self.player_id,
                env=env
            )

    def recalculate_transition_model(self, env):
        """
        Recursively recalculates the transition probability tensor for this agent
        and all agents in its opponent model hierarchy.
        """
        # Calculate current execution probabilities
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)
        
        # Recursively update transition models of lover level-k estimates
        if self.k > 1 and self.opponent:
            # Check if is intialized
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            opponent_model = cast(LevelKDPAgent_NonStationary, self.opponent)
            
            opponent_model.recalculate_transition_model(env)

    def act(self, obs: State, env=None) -> Action:
        """
        Action selection for the non-stationary case.
        First, it updates the transition model based on the current environment state.
        """
        # Recalculate transition model 
        self.recalculate_transition_model(env)
        
        return super().act(obs, env)


class LevelKDPAgent_Dynamic(LevelKDPAgent_Stationary):
    """
    A Level-K DP agent that learns the environment's transition model online.

    Instead of using a fixed transition probability tensor, this agent maintains
    Dirichlet counts for P(executed | intended, state) and updates them after each
    step, making it suitable for environments with unknown transition model.
    """
    
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray,
                 n_states: int, epsilon: float, gamma: float, player_id: int, env):

        # Initialize from the base class, which sets up lookups, etc.
        super().__init__(k, action_space, opponent_action_space, n_states, 
                         epsilon, gamma, player_id, env)
        
        # --- Dynamic-Specific Initialization ---
        # This tensor will be recalculated for the current state at each step.
        
        self.prob_exec_tensor = None # This agent learns the transition model, so no fixed tensor
        self.transition_model_weights = np.ones(
            (self.n_states, self.num_self_actions, self.num_opponent_actions, self.num_self_actions, self.num_opponent_actions)
        )

        # Initialize k-level hierarchy
        if self.k == 1:
            self.dirichlet_counts = np.ones((self.n_states, len(self.opponent_action_space)))
        elif self.k > 1:
            self.opponent = LevelKDPAgent_Dynamic(
                k=self.k - 1,
                action_space=self.opponent_action_space,
                opponent_action_space=self.action_space, 
                n_states=self.n_states,
                epsilon=self.epsilon,
                gamma=self.gamma,
                player_id=1 - self.player_id,
                env=env
            )

    def _get_probabilities_for_state(self, obs: State) -> np.ndarray:
        """Calculates P(executed | intended) from learned counts for a given state."""
        
        weights_for_obs = self.transition_model_weights[obs, :, :, :, :]
        # Sum over the last two axes (executed actions) to get the total counts for each intended pair.
        total_counts = np.sum(weights_for_obs, axis=(2, 3), keepdims=True)
        
        # Normalize weights to get probabilities, handling division by zero
        prob_tensor_for_obs = np.divide(
            weights_for_obs, total_counts,
            out=np.zeros_like(weights_for_obs),
            where=total_counts != 0
        )
        return prob_tensor_for_obs
    
    def optim_act(self, obs: State) -> Action:
        """Selects optimal action using the dynamically learned transition model."""
        
        # Update the transition model for the current state before making a decision.
        self.prob_exec_tensor = self._get_probabilities_for_state(obs)
        
        # Call the parent's (Stationary) efficient optim_act method
        return super().optim_act(obs)

    def update(self, obs: State, actions: list[Action], 
               rewards, new_obs: State):
        """Updates the agent's value function V(s,b) and its internal opponent model."""
        
        # Correctly assign intended actions based on player_id
        if self.player_id == 0:  # This agent is the Blue Player (DM)
            intended_action_self, intended_action_opp = actions[0], actions[1]
        else:  # This agent is the Red Player (ADV)
            intended_action_self, intended_action_opp = actions[1], actions[0]
        
        # Fast lookup to find which executed actions could have caused the transition s -> new_obs.
        possible_exec_actions_mask = (self.s_prime_lookup[obs] == new_obs)
        exec_self_indices, exec_opp_indices = np.where(possible_exec_actions_mask)
        
        # Update the Dirichlet counts for all plausible executed actions
        for a_self_exec, a_opp_exec in zip(exec_self_indices, exec_opp_indices):
            self.transition_model_weights[obs, intended_action_self, intended_action_opp, a_self_exec, a_opp_exec] += 1
        
        # Get the fresh probability tensor for this state before updating the value function
        self.prob_exec_tensor = self._get_probabilities_for_state(obs)
        
        # Call the parent's update method to handle the Bellman update and opponent model
        super().update(obs, actions, None, new_obs)