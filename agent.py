"""
This module implements several agents. An agent is characterized by two methods:
 * act : implements the policy, i.e., it returns agent's decisions to interact in a MDP or Markov Game.
 * update : the learning mechanism of the agent.
"""

import numpy as np
from numpy.random import choice

from engine import RMG


class Agent():
    """
    Parent abstract Agent.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        """
        This implements the policy, \pi : S -> A.
        obs is the observed state s
        """
        raise NotImplementedError()

    def update(self, obs, actions, rewards, new_obs):
        """
        This is after an interaction has ocurred, ie all agents have done their respective actions, observed their rewards and arrived at
        a new observation (state).
        For example, this is were a Q-learning agent would update her Q-function
        """
        pass


class DummyAgent(Agent):
    """
    A dummy and stubborn agent that always takes the first action, no matter what happens.
    """

    def act(self, obs=None):
        # obs is the state (in this case)

        return self.action_space[0]

    " This agent is so simple it doesn't even need to implement the update method! "


class RandomAgent(Agent):
    """
    An agent that with probability p chooses the first action
    """

    def __init__(self, action_space, p):
        Agent.__init__(self, action_space)
        self.p = p

    def act(self, obs=None):

        assert len(self.action_space) == 2
        return choice(self.action_space, p=[self.p, 1-self.p])

    " This agent is so simple it doesn't even need to implement the update method! "


class IndQLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])

    def act(self, obs=None):
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

class Exp3QLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])
        self.S = np.zeros([self.n_states, len(self.action_space)])
        self.p = np.ones([self.n_states, len(self.action_space)])/len(self.action_space)


    def act(self, obs=None):
        """An epsilon-greedy policy"""
        return choice(self.action_space, p=self.p[obs,:])

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))
        self.S[obs, a0] = self.S[obs, a0] + self.Q[obs, a0]/self.p[obs, a0]

        K = len(self.action_space)

        for i in self.action_space:
            self.p[obs, i] = (1-self.epsilon)/( np.exp((self.S[obs, :] - self.S[obs, i])*self.epsilon/K).sum() ) + self.epsilon/K



class PHCLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, delta, enemy_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])
        self.pi = 1/len(self.action_space)*np.ones([self.n_states, len(self.action_space)])

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        #print(self.pi[obs,:])
        #print(self.n_states)
        return choice(self.action_space, p=self.pi[obs,:])


    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))

        a = self.action_space[np.argmax(self.Q[obs, :])]
        self.pi[obs, :] -= self.delta*self.alpha / len(self.action_space)
        self.pi[obs, a] += ( self.delta*self.alpha + self.delta*self.alpha / len(self.action_space))
        self.pi[obs, :] = np.maximum(self.pi[obs, :], 0)
        self.pi[obs, :] /= self.pi[obs,:].sum()


class WoLFPHCLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, delta_w, delta_l, enemy_action_space=None):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta_w = delta_w
        self.delta_l = delta_l
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])
        self.pi = 1/len(self.action_space)*np.ones([self.n_states, len(self.action_space)])
        self.pi_ = 1/len(self.action_space)*np.ones([self.n_states, len(self.action_space)])
        self.C = np.zeros(self.n_states)

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        #print(self.pi[obs,:])
        #print(self.n_states)
        return choice(self.action_space, p=self.pi[obs,:])


    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))

        self.C[obs] += 1
        self.pi_[obs, :] += (self.pi[obs,:]-self.pi_[obs,:])/self.C[obs]
        a = self.action_space[np.argmax(self.Q[obs, :])]

        if np.dot(self.pi[obs, :], self.Q[obs,:]) > np.dot(self.pi_[obs, :], self.Q[obs,:]):
            delta = self.delta_w
        else:
            delta = self.delta_l

        self.pi[obs, :] -= delta*self.alpha / len(self.action_space)
        self.pi[obs, a] += ( delta*self.alpha + delta*self.alpha / len(self.action_space))
        self.pi[obs, :] = np.maximum(self.pi[obs, :], 0)
        self.pi[obs, :] /= self.pi[obs,:].sum()

class FPLearningAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions in a bayesian way.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        # This is the Q-function Q(s, a, b)
        self.Q = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir = np.ones( len(self.enemy_action_space) )

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[ np.argmax( np.dot( self.Q[obs], self.Dir/np.sum(self.Dir) ) ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        self.Dir[a1] += 1 # Update beliefs about adversary

        aux = np.max( np.dot( self.Q[new_obs], self.Dir/np.sum(self.Dir) ) )
        self.Q[obs, a0, a1] = (1 - self.alpha)*self.Q[obs, a0, a1] + self.alpha*(r0 + self.gamma*aux)

class FPQwForgetAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions in a bayesian way, plus a discount to ignore distant observations.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, forget=0.8):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        # This is the Q-function Q(s, a, b)
        self.Q = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir = np.ones( len(self.enemy_action_space) )
        self.forget = forget

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[ np.argmax( np.dot( self.Q[obs], self.Dir/np.sum(self.Dir) ) ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        self.Dir *= self.forget
        self.Dir[a1] += 1 # Update beliefs about adversary

        aux = np.max( np.dot( self.Q[new_obs], self.Dir/np.sum(self.Dir) ) )
        self.Q[obs, a0, a1] = (1 - self.alpha)*self.Q[obs, a0, a1] + self.alpha*(r0 + self.gamma*aux)

class Level2QAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alphaA = learning_rate
        self.alphaB = learning_rate
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        #self.gammaB = 0

        self.enemy_action_space = enemy_action_space

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])

        # This is the Q-function Q_B(s, b, a) (i.e, the adversary Q-function, as a point estimate)
        self.QB = np.zeros([self.n_states, len(self.enemy_action_space), len(self.action_space) ])
        #self.QB = np.zeros([self.n_states, len(self.enemy_action_space)])

        # Parameters of the Dirichlet distribution used to model the other agent's belief of our actions p_B(a)
        # Initialized using a uniform prior
        self.DirB = np.ones( len(self.action_space) )

    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            # We obtain opponent's next action using Q_B
            if np.random.rand() < self.epsilonB:
                b = choice(self.action_space)
            else:
                b = self.enemy_action_space[ np.argmax( np.dot( self.QB[obs], self.DirB/np.sum(self.DirB) ) ) ]  # Check and add uncertainty!!
                #b = self.action_space[np.argmax(self.QB[obs, :])]

            # Add epsilon-greedyness
            return self.action_space[ np.argmax( self.QA[obs, :, b ] ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        #self.DirB *= 1
        self.DirB[a] += 1 # Update beliefs about adversary's level 1 model

        # Update beliefs about adversary's Q function
        aux = np.max( np.dot( self.QB[new_obs], self.DirB/np.sum(self.DirB) ) )
        self.QB[obs, b, a] = (1 - self.alphaB)*self.QB[obs, b, a] + self.alphaB*(rB + self.gammaB*aux)
        #self.QB[obs, b] = (1 - self.alphaB)*self.QB[obs, b] + self.alphaB*(rB + self.gammaB*np.max(self.QB[new_obs, :]))

        # We obtain opponent's next action using Q_B
        if np.random.rand() < self.epsilonB:
            bb = choice(self.action_space)
        else:
            bb = self.enemy_action_space[ np.argmax( np.dot( self.QB[new_obs], self.DirB/np.sum(self.DirB) ) ) ]  # Check and add uncertainty!!
            #bb = self.action_space[np.argmax(self.QB[new_obs, :])]

        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA[new_obs, :, bb]))



class TFT(Agent):
    """
    An agent playing TFT
    """

    def __init__(self, action_space):
        Agent.__init__(self, action_space)

    def act(self, obs):

        if obs[0] == None: #MAAAL esto lo interpreta como vacío si (0,0)!!!
            return(self.action_space[0]) # First move is cooperate
        else:
            return(obs[1]) # Copy opponent's previous action



    " This agent is so simple it doesn't even need to implement the update method! "

class Mem1FPLearningAgent(Agent):
    """
    Extension of the FPLearningAgent to the case of having memory 1
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        # This is the Q-function Q(s, a, b)
        self.Q = np.zeros( [len(self.action_space),len(self.enemy_action_space),
            len(self.action_space), len(self.enemy_action_space)] )
        # Parameters of the Dirichlet distribution used to model the other agent, conditioned to the previous action
        # Initialized using a uniform prior
        self.Dir = np.ones( [len(self.action_space),
            len(self.enemy_action_space),len(self.enemy_action_space)] )

    def act(self, obs):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            if obs[0] == None:
                unif = np.ones(len(self.action_space))
                return self.action_space[ np.argmax( np.dot( self.Q[obs[0], obs[1],:,:],
                    unif/np.sum(unif) ) ) ]
            else:
                return self.action_space[ np.argmax( np.dot( self.Q[obs[0], obs[1],:,:],
                    self.Dir[obs[0], obs[1],:]/np.sum(self.Dir[obs[0], obs[1],:]) ) ) ]



    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        if obs[0] == None:
            unif = np.ones(len(self.action_space))
            aux = np.max( np.dot( self.Q[new_obs[0],new_obs[1],:,:], unif/np.sum(unif) ) )
        else:
            self.Dir[obs[0],obs[1],a1] += 1 # Update beliefs about adversary
            aux = np.max( np.dot( self.Q[new_obs[0],new_obs[1],:,:],
                self.Dir[new_obs[0],new_obs[1],:]/np.sum(self.Dir[new_obs[0],new_obs[1],:]) ) )

        self.Q[obs[0], obs[1], a0, a1] = ( (1 - self.alpha)*self.Q[obs[0], obs[1], a0, a1] +
            self.alpha*(r0 + self.gamma*aux) )


############################ KUHN POKER ########################################

class kuhnAgent2(Agent):
    """
    Stationary second agent in kuhn poker game. His action is parametrized using
    two parameters and depends on his card and the previous movement of his opponent.
    Cards: J=0, Q=1, K=2.
    Actions: pass=0, bet=1.
    """

    def __init__(self, action_space, zeta, eta):
        Agent.__init__(self, action_space)

        self.zeta = zeta
        self.eta = eta

    def act(self, card, enemy_action, obs=None):
        if card == 2:
            return 1
        if card == 0:
            if enemy_action == 0:
                return 1 if np.random.rand() < self.zeta else  0
            else:
                return 0
        else:
            if enemy_action == 0:
                return 0
            else:
                return 1 if np.random.rand() < self.eta else 0
