import numpy as np
from numpy.random import choice
from tqdm.auto import tqdm

from typing import cast, Optional

from .utils import softmax
from .base import LearningAgent

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
QFunction = np.ndarray

class LevelKQAgent(LearningAgent):
    """
    A Level-K Q-learning agent for multi-agent environments.

    This agent uses a cognitive hierarchy model. It reasons about the opponent's
    actions to make better decisions. The Q-function is three-dimensional,
    representing the value of a state-action pair, conditioned on the
    opponent's intended action: `Q(s, a_self, a_opponent)`.

    - k=1 (Base Case): A Level-1 agent models its opponent as Level-0 (random).
      It uses a Dirichlet distribution to learn the opponent's policy `P(b|s)`
      from observed actions.
    - k>1 (Recursive Step): A Level-k agent models its opponent as a
      Level-(k-1) agent. It recursively builds this model to predict the
      opponent's policy.

    Attributes:
        k (int): The cognitive level of the agent. A Level-k agent models its
                 opponent as a Level-(k-1) agent.
        n_states (int): The total number of states in the environment.
        learning_rate (float): The learning rate (alpha) for Q-function updates.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        gamma (float): The discount factor for future rewards.
        action_space (np.ndarray): The set of actions available to this agent.
        opponent_action_space (np.ndarray): The set of actions available to the opponent.
        lower_level_k_epsilon(float): The exploration rate embeded into lower levels of the k-level hierarchy.
        grid_size (int): The size of one dimension of the square grid.
        Q (np.ndarray): The agent's Q-table, with shape
                        (n_states, num_self_actions, num_opponent_actions).
        opponent (LevelKQAgent): The agent's internal model of its opponent.
                                 This is None for a Level-1 agent.
        dirichlet_counts (np.ndarray): Dirichlet counts for modeling a Level-0
                                       opponent's policy. Used only by Level-1 agents.
    """
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray,lower_level_k_epsilon: float,
                 n_states: int, grid_size: int, learning_rate: float, epsilon: float, gamma: float, initial_Q_value: float, player_id: int):
        if k < 1:
            raise ValueError("Level k must be a positive integer.")

        # --- Core Agent Parameters ---
        self.k = k
        self.action_space = action_space
        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.initial_Q_value = initial_Q_value
        self.opponent_action_space = opponent_action_space
        self.lower_level_k_epsilon=lower_level_k_epsilon
        self.grid_size = grid_size
        self.player_id = player_id
        

        # Q-function Q(s, a_self, a_opponent) setup
        self.Q = self._setup_Q(self.initial_Q_value)

        # --- Level-Specific Opponent Model Initialization ---
        self.opponent = None
        self.dirichlet_counts = None
        
        # --- Opponent Model Initialization ---
        if self.k == 1:
            # Base Case (k=1): Model opponent as Level-0 (random policy).
            # We use Dirichlet counts to learn this policy from observations.
            # Initially, we assume a uniform prior.
            self.dirichlet_counts = np.ones((self.n_states, len(self.opponent_action_space)))
        elif self.k > 1:
            # Recursive Step (k>1): The opponent is a Level-(k-1) version of this
            # same class, with the player roles reversed.
            self.opponent = LevelKQAgent(
                k=self.k - 1,
                action_space=self.opponent_action_space, # Swapping actions space and opponent action space
                opponent_action_space=self.action_space,
                lower_level_k_epsilon=self.lower_level_k_epsilon,
                n_states=self.n_states,
                learning_rate=self.alpha, # Using same parameters for the modeled opponent
                epsilon=self.epsilon,
                gamma=self.gamma,
                initial_Q_value = self.initial_Q_value,
                grid_size=grid_size,
                player_id=1-self.player_id
            )
            
    def _setup_Q(self,initial_Q_value: float) -> QFunction:
        """
        Initializes the Q-table `Q(s, a_self, a_opponent)`.

        The value of all terminal states is set to 0, as no further rewards
        can be obtained from them.

        Args:
            initial_Q_value (float): The initial value for all non-terminal states.

        Returns:
            np.ndarray: The initialized Q-table of shape (n_states, num_self_actions, num_opponent_actions).
        """
        
        Q = np.ones([self.n_states, len(self.action_space), len(self.opponent_action_space)])*initial_Q_value
        
        # tqdm shows progress bar.
        for s in tqdm(range(self.n_states), desc="Initializing value function."):
            if self._is_terminal_state(s):
                Q[s,:,:] = 0.0
                
        return Q

    def _is_terminal_state(self, obs:State) -> bool:
        """
        Checks if a given state is terminal.

        A state is terminal if either player has collected both of their coins
        (a win) or if all four coins have been claimed by any combination of
        players (a draw or a win). This logic is coupled to the environment's
        specific state encoding scheme.
        """
        
        # Grid size N, base for coin collection status
        _, base_coll = self.grid_size**2, 2
        
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
        
        # NOTE: The Q agent cannot account for the episode ending due to max_steps,
        # as this is not encoded in the state itself. This is a known limitation.
        return p0_wins or p1_wins or is_draw

    def get_opponent_policy(self, obs:State) -> Policy:
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
        
        # --- Level-1 Agent: Opponent is Level-0 (Learned Random Policy) ---
        if self.k == 1:
            # Check if atributes is still initalized to None
            assert self.dirichlet_counts is not None, "dirichlet_counts should be initialized for a Level-1 agent."
            
            # Normalize counts to get a probability distribution.
            dir_sum = np.sum(self.dirichlet_counts[obs])
            if dir_sum == 0:
                # Handle intialization of dirichlet_counts to 0
                return np.ones(len(self.opponent_action_space)) / len(self.opponent_action_space)
            return self.dirichlet_counts[obs] / dir_sum
        
        # --- Level-k > 1 Agent: Opponent is Level-(k-1) ---
        else:
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            return self.opponent.get_policy(obs)

    def get_policy(self, obs: State) -> Policy:
        """
        Calculates this agent's own epsilon-greedy policy distribution.

        The policy is derived from the expected Q-values, which are calculated
        by marginalizing over the opponent's predicted policy.

        Args:
            obs (int): The current state observation.

        Returns:
            np.ndarray: A probability distribution over the agent's own actions.
        """
        
        # Predict the opponent's policy for the current state
        opponent_policy = self.get_opponent_policy(obs)
        
        # Calculate E[Q(s,a)] = Σ_b [ P(b|s) * Q(s,a,b) ]
        # This gives the expected Q-value for each of agents actions.
        expected_q_values = np.dot(self.Q[obs], opponent_policy)
        
        # Determine agents optimal action
        optimal_action_idx = np.argmax(expected_q_values)

        # Construct the epsilon-greedy policy distribution
        num_actions = len(self.action_space)
        policy = np.full(num_actions, self.epsilon / num_actions)
        policy[optimal_action_idx] += (1.0 - self.epsilon)
        
        return policy

    def act(self, obs:State, env=None) -> int:
        """
        Selects an action by sampling from the epsilon-greedy policy.

        Args:
            obs (int): The current state observation.
            env: The environment (not used in this agent).

        Returns:
            int: The chosen action.
        """
        policy = self.get_policy(obs)
        return choice(self.action_space, p=policy)

    def update(self, obs: int, actions: tuple[Action, Action], new_obs: int, rewards: Optional[tuple[Reward, Reward]]):
        """
        Updates the Q-function and the internal opponent model after a transition.

        Args:
            obs (int): The state before the action.
            actions (list): A list [self_action, opponent_action].
            rewards (list): A list [self_reward, opponent_reward].
            new_obs (int): The state after the action.
        """
        
        if rewards is None:
            raise ValueError("LevelKQAgent requires a rewards tuple for its update method.")

        
        if self.player_id == 0:
            self_action, opponent_action = actions
            self_reward, _ = rewards
        else: 
            self_action, opponent_action = actions[::-1]
            self_reward, _ = rewards[::-1]
        
        
        # --- Update Opponent Model ---
        if self.k > 1:
            assert self.opponent is not None, "Opponent model must be set for Level > 1 agents."
            # Recursively call update on the internal Level-(k-1) model.
            # Note the reversed order for actions and rewards.
            self.opponent.update(obs, actions, new_obs, rewards)
        else: # k == 1
            assert self.dirichlet_counts is not None, "dirichlet_counts should be initialized for a Level-1 agent."
            # Update the Dirichlet counts for the observed opponent action.
            self.dirichlet_counts[obs, opponent_action] += 1
        
        # If the state was already terminal, its value is fixed at 0, so no update needed.
        if self._is_terminal_state(obs):
            return
        
        # --- Update Agent's Q-Function ---
        
        # Calculate the value of the next state V(s') for the Bellman update.
        # V(s') = max_a' E[Q(s',a')] = max_a' Σ_b' [ P(b'|s') * Q(s',a',b') ]
        opponent_policy_new = self.get_opponent_policy(new_obs)
        expected_q_new = np.dot(self.Q[new_obs], opponent_policy_new)
        max_q_new = np.max(expected_q_new)
        
       # Standard Q-learning update rule.
        current_q = self.Q[obs, self_action, opponent_action]
        self.Q[obs, self_action, opponent_action] = (1 - self.alpha) * current_q + self.alpha * (self_reward + self.gamma * max_q_new)
        
    def update_epsilon(self, new_epsilon_agent: float, new_epsilon_lower_k_level: Optional[float]):
        """
        Updates the exploration rate for this agent and its recursive opponent model.
        This ensures that if the exploration schedule changes, the change
        propagates down the entire cognitive hierarchy.
        
        Args:
            new_epsilon_agent (float): The new exploration rate for agent.
            new_epsilon_lower_k_level (float): The new exploration rate for agents internal model of the opponent reasoning (k-1 levels)
        """
        
        self.epsilon = new_epsilon_agent
        if self.k > 1 and self.opponent:
            new_epsilon_lower_k_level = cast(float,new_epsilon_lower_k_level)
            self.opponent.update_epsilon(new_epsilon_lower_k_level, new_epsilon_lower_k_level)


class LevelKQAgentSoftmax(LevelKQAgent):
    """
    A Level-K Q-learning agent that uses a softmax policy for action selection
    instead of epsilon-greedy. It inherits from `LevelKQAgent`.

    Attributes:
        beta (float): The temperature parameter for the softmax calculation.
                      Higher beta leads to more deterministic (greedy) actions.
    """
    def __init__(self, k: int, action_space: np.ndarray, opponent_action_space: np.ndarray, lower_level_k_epsilon: float,
                 n_states: int, grid_size: int, learning_rate: float, epsilon: float, gamma: float, initial_Q_value: float, beta: float, player_id: int):
        
        # Call the parent class constructor to handle all common setup.
        super().__init__(k, action_space, opponent_action_space, lower_level_k_epsilon, n_states, grid_size, learning_rate, epsilon, gamma, initial_Q_value, player_id)
        
        self.beta = beta

        # IMPORTANT: Override the opponent model to also be a Softmax agent.
        # This ensures the cognitive hierarchy is consistent.
        if self.k > 1:
            self.opponent = LevelKQAgentSoftmax( # Recursive call to the Softmax class
                k=self.k - 1,
                action_space=self.opponent_action_space,
                opponent_action_space=self.action_space,
                lower_level_k_epsilon=self.lower_level_k_epsilon,
                n_states=self.n_states,
                grid_size=self.grid_size,  
                learning_rate=self.alpha,
                epsilon=self.epsilon,
                gamma=self.gamma,
                initial_Q_value=self.initial_Q_value,
                beta=self.beta,
                player_id=1-self.player_id
            )

    def get_policy(self, obs):
        """
        Overrides the parent method to calculate a softmax policy distribution.

        Args:
            obs (int): The current state observation.

        Returns:
            np.ndarray: A softmax probability distribution over the agent's actions.
            
        """
        # Predict the opponent's policy.
        opponent_policy = self.get_opponent_policy(obs)
        
        # Calculate expected Q-values by marginalizing over the opponent's policy.
        expected_q_values = np.dot(self.Q[obs], opponent_policy)
        
        # Return the softmax distribution over the expected Q-values.
        return softmax(expected_q_values, self.beta)