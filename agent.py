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
        self.Q = -10*np.ones([self.n_states, len(self.action_space)])

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
        
class IndQLearningAgentSoftmax(IndQLearningAgent):
    
    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None):
        IndQLearningAgent.__init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space)
        
    def act(self, obs=None):
        p = np.exp(self.Q[obs,:])
        p = p / np.sum(p)
        return choice(self.action_space, p=p)
    
    

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
            #print('obs ', obs)
            #print(self.Q[obs].shape)
            #print(self.Dir.shape)
            #print(np.dot( self.Q[obs], self.Dir/np.sum(self.Dir) ).shape)
            return self.action_space[ np.argmax( np.dot( self.Q[obs], self.Dir/np.sum(self.Dir) ) ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, a1 = actions
        r0, _ = rewards

        self.Dir[a1] += 1 # Update beliefs about adversary

        aux = np.max( np.dot( self.Q[new_obs], self.Dir/np.sum(self.Dir) ) )
        self.Q[obs, a0, a1] = (1 - self.alpha)*self.Q[obs, a0, a1] + self.alpha*(r0 + self.gamma*aux)
    
    def get_Q_function(self):
        """Returns the Q-function of the agent"""
        return self.Q
    
    def get_Dirichlet(self):
        """Returns the Dirichlet distribution of the agent"""
        return self.Dir

class FPLearningAgent_DP(Agent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions in a bayesian way.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, rewards, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        self.rewards = rewards
        # This is the value function V(s,b)
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir = np.ones(len(self.enemy_action_space))
        
        # Initialize best action from the value function update, which is used for the as action chosen in the

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            #print('obs ', obs)
            #print(self.Q[obs].shape)
            #print(self.Dir.shape)
            #print(np.dot( self.Q[obs], self.Dir/np.sum(self.Dir) ).shape)
            return self.action_space[ np.argmax( np.dot( self.V[obs], self.Dir/np.sum(self.Dir) ) ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        _, a1 = actions
        r0, _ = rewards

        self.Dir[a1] += 1 # Update beliefs about adversary

        aux = np.dot(self.V[new_obs], self.Dir/np.sum(self.Dir))
        self.V[obs, a1] = r0 + self.gamma*aux
    
    def get_V_function(self):
        """Returns the Q-function of the agent"""
        return self.Q
    
    def get_Dirichlet(self):
        """Returns the Dirichlet distribution of the agent"""
        return self.Dir

class FPQwForgetAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions in a bayesian way, plus a discount to ignore distant observations.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """
    # TODO: Check if this is correct
    # !!! So the level-0 agent is meant the adversary who is assumed to simply choose actions according to p(b)?
    # !!! I was under impresion that the level-0 is DM who keeps track of Q function dependent on Q(a,b) which utilizes p(b)
    # !!! as DM's belief about opponent's policy.
    # * Note: Seems to be actualy fine, because this is used wit n_states = 1, so this is the sateless variant.
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

###############################################
## Agents for the friend or foe environment
###############################################
class ExpSmoother(Agent):
    """
    An agent predicting its opponent actions using an exponential smoother.
    """

    def __init__(self, action_space, enemy_action_space, learning_rate):
        Agent.__init__(self, action_space)

        self.alpha = learning_rate
        self.action_space = action_space
        self.enemy_action_space = enemy_action_space
        # Initial forecast
        self.prob = np.ones( len(self.enemy_action_space) )*0.5

    def act(self, obs=None):
        """Just chooses the less probable place"""
        return self.action_space[np.argmin(self.prob)]


    def update(self, obs, actions, rewards, new_obs):
        """Update the exp smoother"""
        a0, a1 = actions
        OHE = np.array([[1,0],[0,1]])

        self.prob = self.alpha*self.prob + (1-self.alpha)*OHE[a1] # Update beliefs about DM

##
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

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = FPLearningAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            b = self.enemy.act(obs)
            #print(self.QA.shape)
            #print('b', b)
            #print(self.QA[obs, :, b ])
            # Add epsilon-greedyness
            return self.action_space[ np.argmax( self.QA[obs, :, b ] ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        
        # TODO: Check if this is correct
        # !!! There should be mean value of Q(s',a',b') with respect to p(b'|s'), this formulation implies that the adversary
        # !!! Is directly using the level-1 model to choose its actions (we don't need averaging in this case).
        # * Note: This is the update, after we observe (s,a,b,r,s'). We utilize only this information, so the mean value described above
        # *       should be implemented here, because we do not yet observe b'.
        bb = self.enemy.act(obs)
    
        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA[new_obs, :, bb]))

class Level2QAgent_fixed(Agent):
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

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = FPLearningAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            # Adversary's Q-function
            QB = self.enemy.get_Q_function()
            
            # Adversary's belief about DM's action (in form of weights)
            Dir_B = self.enemy.get_Dirichlet()
            # Normalize Dir_B
            Dir_B = Dir_B/np.sum(Dir_B)
            
            # TODO: Test for more states than 1
            # Calculate the mean value of adversarys Q-function with respect to Dir_B
            mean_values_QB_p_B_a = np.dot(QB[obs, :, :], Dir_B)
            
            # Calculate argmx b of the mean values of adversarys Q-function
            Adversary_best_action = np.argmax(mean_values_QB_p_B_a)
            
            # Calculate DM's belief about adversary's action using epsilon-greedy policy
            p_A_b = np.ones(len(self.enemy_action_space))
            p_A_b.fill(self.epsilonA/(len(self.enemy_action_space)-1))
            p_A_b[Adversary_best_action] = 1 - self.epsilonA
            
            # Calculate the mean value of DM's Q-function with respect to DM's belief about adversary's actions p_A_b
            mean_values_QA_p_A_b = np.tensordot(self.QA[obs, :, :], p_A_b, axes=([1], [0]))
            
            return Adversary_best_action

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        
        # Adversary's Q-function
        QB = self.enemy.get_Q_function()
        
        # Adversary's belief about DM's action (in form of weights)
        Dir_B = self.enemy.get_Dirichlet()
        # Normalize Dir_B
        Dir_B = Dir_B/np.sum(Dir_B)
        
        # TODO: Test for more states than 1
        # Calculate the mean value of adversarys Q-function with respect to Dir_B
        mean_values_QB_p_B_a = np.dot(QB[new_obs, :, :], Dir_B)
        
        # Calculate argmx b of the mean values of adversarys Q-function
        Adversary_best_action = np.argmax(mean_values_QB_p_B_a)
        
        # Calculate DM's belief about adversary's action using epsilon-greedy policy
        p_A_b = np.ones(len(self.enemy_action_space))
        p_A_b.fill(self.epsilonA/(len(self.enemy_action_space)-1))
        p_A_b[Adversary_best_action] = 1 - self.epsilonA
        
        # Calculate the mean value of DM's Q-function with respect to DM's belief about adversary's actions p_A_b
        mean_values_QA_p_A_b = np.tensordot(self.QA[new_obs, :, :], p_A_b, axes=([1], [0]))
        
        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(mean_values_QA_p_A_b))

class Level2QAgent_DP_fixed(Agent):
    """
    A value iteration agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular fashion, i.e., using a matrix V.
    """

    def __init__(self, action_space, enemy_action_space, n_states, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        #self.gammaB = 0

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = FPLearningAgent_DP(self.enemy_action_space, self.action_space, self.n_states,
                                        epsilon=self.epsilonB, gamma=self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            # Adversary's Q-function
            QB = self.enemy.get_Q_function()
            
            # Adversary's belief about DM's action (in form of weights)
            Dir_B = self.enemy.get_Dirichlet()
            # Normalize Dir_B
            Dir_B = Dir_B/np.sum(Dir_B)
            
            # TODO: Test for more states than 1
            # Calculate the mean value of adversarys Q-function with respect to Dir_B
            mean_values_QB_p_B_a = np.dot(QB[obs, :, :], Dir_B)
            
            # Calculate argmx b of the mean values of adversarys Q-function
            Adversary_best_action = np.argmax(mean_values_QB_p_B_a)
            
            # Calculate DM's belief about adversary's action using epsilon-greedy policy
            p_A_b = np.ones(len(self.enemy_action_space))
            p_A_b.fill(self.epsilonA/(len(self.enemy_action_space)-1))
            p_A_b[Adversary_best_action] = 1 - self.epsilonA
            
            # Calculate the mean value of DM's Q-function with respect to DM's belief about adversary's actions p_A_b
            mean_values_QA_p_A_b = np.tensordot(self.QA[obs, :, :], p_A_b, axes=([1], [0]))
            
            return Adversary_best_action

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        
        # Adversary's Q-function
        QB = self.enemy.get_Q_function()
        
        # Adversary's belief about DM's action (in form of weights)
        Dir_B = self.enemy.get_Dirichlet()
        # Normalize Dir_B
        Dir_B = Dir_B/np.sum(Dir_B)
        
        # TODO: Test for more states than 1
        # Calculate the mean value of adversarys Q-function with respect to Dir_B
        mean_values_QB_p_B_a = np.dot(QB[new_obs, :, :], Dir_B)
        
        # Calculate argmx b of the mean values of adversarys Q-function
        Adversary_best_action = np.argmax(mean_values_QB_p_B_a)
        
        # Calculate DM's belief about adversary's action using epsilon-greedy policy
        p_A_b = np.ones(len(self.enemy_action_space))
        p_A_b.fill(self.epsilonA/(len(self.enemy_action_space)-1))
        p_A_b[Adversary_best_action] = 1 - self.epsilonA
        
        # Calculate the mean value of DM's Q-function with respect to DM's belief about adversary's actions p_A_b
        mean_values_QA_p_A_b = np.tensordot(self.QA[new_obs, :, :], p_A_b, axes=([1], [0]))
        
        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(mean_values_QA_p_A_b))
        
class Level2QAgentSoftmax(Level2QAgent):
    """
    A Q-learning agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
        Level2QAgent.__init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma)
        
    def act(self, obs=None):
        b = self.enemy.act(obs)
        p = np.exp(self.QA[obs,:,b])
        p = p / np.sum(p)
        return choice(self.action_space, p=p)


##
class Level3QAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 2 agent.
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

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = Level2QAgent(self.enemy_action_space, self.action_space,
         self.n_states, self.alphaB, self.epsilonB, self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            b = self.enemy.act(obs)
            # Add epsilon-greedyness
            return self.action_space[ np.argmax( self.QA[obs, :, b ] ) ]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy.act(obs)

        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA[new_obs, :, bb]))


class Level3QAgentMixExp(Agent):
    """
    A Q-learning agent that treats the other player as a mixture of a
    level 2 agent and a level 1 agent, with different probabilities, that
    are updated dynamically using an exponential smoother.
    She learns from others' actions, estimating their Q function.
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
        self.prob_type = np.array([0.5, 0.5])
        #self.gammaB = 0

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = Level2QAgent(self.enemy_action_space, self.action_space,
         self.n_states, self.alphaB, 0.1, self.gammaB)

        self.enemy2 = FPLearningAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=0.1, gamma=self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA1 = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        self.QA2 = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])

        # To store enemies actions
        self.E1_action = 0
        self.E2_action = 0


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            self.E1_action = self.enemy.act(obs)
            self.E2_action = self.enemy2.act(obs)
            # Add epsilon-greedyness
        res1 = self.action_space[ np.argmax( self.QA1[obs, :, self.E1_action ] ) ]
        res2 = self.action_space[ np.argmax( self.QA2[obs, :, self.E2_action ] ) ]
        return choice( np.array([res1, res2]), p = self.prob_type )

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards
        lr_prob = 0.4

        ### Update p

        if self.E1_action == self.E2_action:
            pass
        else:
            if self.E1_action == b:
                self.prob_type = lr_prob*self.prob_type + (1-lr_prob)*np.array([1,0])
            else:
                self.prob_type = lr_prob*self.prob_type + (1-lr_prob)*np.array([0,1])

        self.enemy.update( obs, [b,a], [rB, rA], new_obs )
        self.enemy2.update( obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy.act(obs)
        bb2 = self.enemy2.act(obs)

        # Finally we update the supported agent's Q-function
        self.QA1[obs, a, b] = (1 - self.alphaA)*self.QA1[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA1[new_obs, :, bb]))
        self.QA2[obs, a, b] = (1 - self.alphaA)*self.QA2[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA2[new_obs, :, bb2]))

class Level3QAgentMixDir(Agent):
    """
    A Q-learning agent that treats the other player as a mixture of a
    level 2 agent and a level 1 agent, with different probabilities, that
    are updated dynamically in a Bayesian way.
    She learns from others' actions, estimating their Q function.
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

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy1 = Level2QAgent(self.enemy_action_space, self.action_space,
         self.n_states, self.alphaB, 0.0, self.gammaB)

        self.enemy2 = FPLearningAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=0.0, gamma=self.gammaB)

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA1 = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        self.QA2 = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])

        # To store enemies actions
        self.E1_action = 0
        self.E2_action = 0

        # Probabilities of types
        self.prob_type = np.array([1, 1]) #Dirichlet distribution


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            self.E1_action = self.enemy1.act(obs)
            self.E2_action = self.enemy2.act(obs)
            # Add epsilon-greedyness
        res1 = self.action_space[ np.argmax( self.QA1[obs, :, self.E1_action ] ) ]
        res2 = self.action_space[ np.argmax( self.QA2[obs, :, self.E2_action ] ) ]
        return choice( np.array([res1, res2]), p = self.prob_type/(np.sum(self.prob_type)) )

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards
        lr_prob = 0.4

        ### Update Dirichlets  self.Dir[a1] += 1 # Update beliefs about adversary

        if self.E1_action == self.E2_action:
            pass
        else:
            if self.E1_action == b:
                self.prob_type[0] += 1
            else:
                self.prob_type[1] += 1

        self.enemy1.update( obs, [b,a], [rB, rA], new_obs )
        self.enemy2.update( obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        bb = self.enemy1.act(obs)
        bb2 = self.enemy2.act(obs)

        # Finally we update the supported agent's Q-function
        self.QA1[obs, a, b] = (1 - self.alphaA)*self.QA1[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA1[new_obs, :, bb]))
        self.QA2[obs, a, b] = (1 - self.alphaA)*self.QA2[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(self.QA2[new_obs, :, bb2]))
