import numpy as np
from numpy.random import choice
from tqdm.notebook import tqdm

from typing import Optional

from .utils import softmax
from .base import LearningAgent

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
QFunction = np.ndarray

class IndQLearningAgent(LearningAgent):
    """
    An independent Q-learning (IQL) agent.

    This agent treats other players as a static part of the environment. It
    learns a Q-function `Q(s, a)` that maps state-action pairs to expected
    future rewards, ignoring the opponent's influence on the state dynamics.
    It is intended to serve as a baseline.

    Attributes:
        n_states (int): The total number of states in the environment.
        grid_size (int): The size of one dimension of the square grid.
        learning_rate (float): The learning rate (learning_rate) for Q-function updates.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        gamma (float): The discount factor for future rewards.
        action_space (np.ndarray): The set of actions available to this agent.
        Q (np.ndarray): The agent's Q-table, with shape (n_states, num_actions).
    """

    def __init__(self, action_space: np.ndarray, n_states: int, grid_size: int, learning_rate: float,
                 epsilon: float, gamma: float, player_id: int):
        
        self.action_space = action_space
        self.n_states = n_states
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.player_id = player_id

        # This is the Q-function Q(s, a)
        self.Q = self._setup_Q(-10)
        
    def _setup_Q(self,initial_value: int) -> QFunction:
        """
        Initializes the Q-table `Q(s, a_self, a_opponent)`.

        The value of all terminal states is set to 0, as no further rewards
        can be obtained from them.

        Args:
            initial_value (float): The initial value for all non-terminal states.

        Returns:
            np.ndarray: The initialized Q-table of shape (n_states, num_self_actions, num_opponent_actions).
        """
        
        Q = np.ones([self.n_states, len(self.action_space)])*initial_value
        
        # tqdm shows progress bar.
        for s in tqdm(range(self.n_states), desc="Initializing value function."):
            if self._is_terminal_state(s):
                Q[s,:] = 0
                
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

    def act(self, obs: State, env = None) -> Action:
        """
        Selects an action using an epsilon-greedy policy.

        With probability epsilon, it chooses a random action. Otherwise, it
        chooses the action with the highest Q-value for the current state.

        Args:
            obs (int): The current state observation.
            env: The environment (not used in this agent).

        Returns:
            int: The chosen action.
        """
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[np.argmax(self.Q[obs, :])]

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards: Optional[tuple[Reward, Reward]]):
        """
        Updates the Q-function using the vanilla Q-learning update rule.

        The update rule is:
        Q(s,a) <- (1-α)Q(s,a) + α(r + γ * max_a' Q(s',a'))

        Args:
            obs (int): The state before the action.
            actions (list): A list [self_action, opponent_action].
            rewards (list): A list [self_reward, opponent_reward].
            new_obs (int): The state after the action.
        """
        if rewards is None:
            raise ValueError("IndQLearningAgent requires a rewards tuple for its update method.")
        
        self_action = actions[0] if self.player_id == 0 else actions[1]
        self_reward = rewards[0] if self.player_id == 0 else rewards[1]

        self.Q[obs, self_action] = (1 - self.learning_rate)*self.Q[obs, self_action] + self.learning_rate*(self_reward + self.gamma*np.max(self.Q[new_obs, :]))
        
    
    def update_epsilon(self, new_epsilon: float):
        """
        Updates the agent's exploration rate (epsilon).

        Args:
            new_epsilon (float): The new exploration rate.
        """
        self.epsilon = new_epsilon
        
class IndQLearningAgentSoftmax(IndQLearningAgent):
    """
    An independent Q-learning agent that uses a softmax policy.

    This agent inherits from `IndQLearningAgent` but overrides the action
    selection method to choose actions probabilistically based on their
    Q-values, rather than using an epsilon-greedy approach.

    Attributes:
        beta (float): The temperature parameter for the softmax calculation.
                      Higher beta leads to more deterministic (greedy) actions.
    """
    
    def __init__(self, action_space: np.ndarray, n_states: int, grid_size: int, learning_rate: float,
                 epsilon: float, gamma: float, beta: float = 1.0, player_id: int = 0):
        # Call the parent constructor
        super().__init__(action_space, n_states, grid_size, learning_rate, epsilon, gamma, player_id)
        
        self.beta = beta
        
    def act(self, obs: int, env=None) -> Action:
        """
        Selects an action using a softmax policy.

        The probability of selecting each action is proportional to its
        exponentiated Q-value, scaled by the temperature parameter beta.

        Args:
            obs (int): The current state observation.
            env: The environment (not used in this agent).

        Returns:
            int: The chosen action.
        """
        # Calculate policy using softmax over Q-values and sample from it.
        policy = softmax(self.Q[obs, :], self.beta)
        return choice(self.action_space, p=policy)
