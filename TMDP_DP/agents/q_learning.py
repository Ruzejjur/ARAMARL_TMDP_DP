import numpy as np
from numpy.random import choice

from .base import Agent
from .utils import softmax

class IndQLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular form, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, opponent_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        
        # This is the Q-function Q(s, a)
        self.Q = -10*np.ones([self.n_states, len(self.action_space)])

    def act(self, obs, env):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[np.argmax(self.Q[obs, :])]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))
        
    
    def update_epsilon(self, new_epsilon):
        """Updating the epsilon for the whole hierarchy."""
        
        self.epsilon = new_epsilon
        
class IndQLearningAgentSoftmax(IndQLearningAgent):
    """ A vanilla Q-learning agent that applies softmax policy."""
    
    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, opponent_action_space=None, beta=1):
        IndQLearningAgent.__init__(self, action_space, n_states, learning_rate, epsilon, gamma, opponent_action_space)
        
        self.beta = beta
        
    def act(self, obs, env):
        
        return choice(self.action_space, p=softmax(self.Q[obs,:],self.beta))
