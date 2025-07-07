import numpy as np
from numpy.random import choice
from tqdm.notebook import tqdm

from .base import Agent

class LevelKQAgent(Agent):
    """
    A template for a level-k Q-learning agent.
    - k=1 (Base Case): A level-1 agent that models its opponent as a level-0 
      agent using a Dirichlet distribution to learn their 
      action probabilities, P(b|s).
    - k>1 (Recursive Step): A level-k agent that models its opponent as a 
      level-(k-1) agent. It recursively builds a model of the opponent to 
      predict their actions.
    """
    def __init__(self, k, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, grid_size):
        if k < 1:
            raise ValueError("Level k must be a positive integer.")
            
        Agent.__init__(self, action_space)

        # Core agent parameters
        self.k = k
        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        self.grid_size = grid_size

        # Q-function Q(s, a, b), where a is self action, b is opponent action
        self.Q = self._setup_Q(-10)

        # --- Level-Specific Opponent Model Initialization ---
        self.enemy = None
        self.Dir = None
        
        if self.k == 1:
            # BASE CASE: Level-1 models opponent with a Dirichlet distribution
            self.Dir = np.ones((self.n_states, len(self.enemy_action_space)))
        elif self.k > 1:
            # RECURSIVE STEP: Level-k models opponent as a Level-(k-1) agent
            self.enemy = LevelKQAgent(
                k=self.k - 1,
                action_space=self.enemy_action_space, # Swapping actions space and enemy action space
                enemy_action_space=self.action_space,
                n_states=self.n_states,
                learning_rate=self.alpha, # Using same parameters for the modeled enemy
                epsilon=self.epsilon,
                gamma=self.gamma,
                grid_size=grid_size
            )
            
    def _setup_Q(self,initial_value):
        """
        Initalizing the value function. Setting the values of terminal states to 0.
        """
        
        Q = np.ones([self.n_states, len(self.action_space), len(self.enemy_action_space)])*initial_value
        
        for s in tqdm(range(self.n_states), desc="Initializing value function."):
            if self._is_terminal_state(s):
                Q[s,:,:] = 0
                
        return Q

    def _is_terminal_state(self, obs):
        """
        Checks if a given state observation is terminal (e.g., both coins are gone).
        Returns True if the state is terminal, False otherwise.
        """
        # We can use the environment snapshot to get the grid parameters
        _, base_coll = self.grid_size**2, 2
        
        state_copy = obs
        c_r2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll)

        # The game is over if both coins have been collected by either player
        coin1_gone = c_b1 or c_r1
        coin2_gone = c_b2 or c_r2
        
        return coin1_gone and coin2_gone

    def get_opponent_policy(self, obs):
        """
        Calculates the predicted probability distribution of the opponent's next action.
        """
        if self.k == 1:
            # For L1, the opponent is L0, modeled by Dirichlet action counts.
            dir_sum = np.sum(self.Dir[obs])
            if dir_sum == 0:
                return np.ones(len(self.enemy_action_space)) / len(self.enemy_action_space)
            return self.Dir[obs] / dir_sum
        else: # k > 1
            # For L-k, the opponent is L-(k-1). We ask our internal model for its policy.
            return self.enemy.get_policy(obs)

    def get_policy(self, obs):
        """
        Calculates this agent's own policy (as a probability distribution) 
        for a given observation using an epsilon-greedy strategy.
        """
        # Predict the opponent's policy for the current state
        opponent_policy = self.get_opponent_policy(obs)
        
        # Calculate the expected Q-value for each of our actions, averaging over the opponent's policy
        expected_q_values = np.dot(self.Q[obs], opponent_policy)
        
        # Determine the optimal action from our perspective
        optimal_action_idx = np.argmax(expected_q_values)

        # Construct the epsilon-greedy policy distribution
        num_actions = len(self.action_space)
        policy = np.full(num_actions, self.epsilon / num_actions)
        policy[optimal_action_idx] += (1.0 - self.epsilon)
        
        return policy

    def act(self, obs, env):
        """Choose an action based on the calculated policy distribution."""
        policy = self.get_policy(obs)
        return choice(self.action_space, p=policy)

    def update(self, obs, actions, rewards, new_obs):
        """Updates the agent's Q-function and its internal opponent model."""
        a_dm, a_adv = actions
        r_dm, r_adv = rewards

        if self.k > 1:
            self.enemy.update(obs, [a_adv, a_dm], [r_adv, r_dm], new_obs)
        else: # k == 1
            self.Dir[obs, a_adv] += 1
        
        # If the state is terminal, no Bellman update should occur.
        if self._is_terminal_state(obs):
            return # Exit early
        
        # Calculate the value of the next state (max Q') for the Bellman update
        opponent_policy_new = self.get_opponent_policy(new_obs)
        expected_q_new = np.dot(self.Q[new_obs], opponent_policy_new)
        max_q_new = np.max(expected_q_new)
        
        # Update our own Q-function
        current_q = self.Q[obs, a_dm, a_adv]
        self.Q[obs, a_dm, a_adv] = (1 - self.alpha) * current_q + self.alpha * (r_dm + self.gamma * max_q_new)
        
    def update_epsilon(self, new_epsilon):
        """Updating the epsilon for the whole hierarchy."""
        
        self.epsilon = new_epsilon
        if self.k > 1 and self.enemy:
            self.enemy.update_epsilon(new_epsilon)


class LevelKQAgentSoftmax(LevelKQAgent):
    """
    A version of the Level-k Q-learning agent that uses a softmax policy
    for action selection instead of epsilon-greedy.
    """
    def __init__(self, k, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, beta=1.0):
        # Call the parent constructor
        super().__init__(k, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, grid_size)
        self.beta = beta

        # IMPORTANT: Override the opponent model to also be a Softmax agent
        if self.k > 1:
            self.enemy = LevelKQAgentSoftmax( # Recursive call to the Softmax class
                k=self.k - 1,
                action_space=self.enemy_action_space,
                enemy_action_space=self.action_space,
                n_states=self.n_states,
                learning_rate=self.alpha,
                epsilon=self.epsilon,
                gamma=self.gamma,
                beta=self.beta
            )

    def get_policy(self, obs):
        """
        Overrides the parent method to calculate a softmax policy distribution.
        """
        # Predict the opponent's policy (which will also be softmax if k > 1)
        opponent_policy = self.get_opponent_policy(obs)
        
        # Calculate expected Q-values
        expected_q_values = np.dot(self.Q[obs], opponent_policy)
        
        # Return the softmax distribution over the expected Q-values
        return softmax(expected_q_values, self.beta)