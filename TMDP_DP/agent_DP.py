"""
This module implements several agents. An agent is characterized by two methods:
 * act : implements the policy, i.e., it returns agent's decisions to interact in a MDP or Markov Game.
 * update : the learning mechanism of the agent.
"""

import numpy as np
from numpy.random import choice


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
        For example, this is were a Q-learning agent would update her Q-function.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    
    def get_Belief(self):
        """Returns the Dirichlet distribution of the agent"""
        return self.Dir

class Level0DPAgent(Agent):
    """
    A myopic agent which maximizes imidiate reward based on belief about other agents actions.
    """
    # TODO: Finish this implementation
    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma,reward_table, system_model):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.enemy_action_space = enemy_action_space
        self.reward_table = reward_table
        # Value iteration does not impement learning rate, leaving for coherence with definition of other agents
        self.learning_rate = None
        
        # This is the value function V(b)
        self.V = np.zeros(len(self.enemy_action_space))
        
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir = np.ones(len(self.enemy_action_space))

    def act(self, obs=None):
        """ Select the action that maximizes the imidiate reward. """
        # * No need for epsilon-greedy strategy in exploitation of the myopic agent as it does not facilitate learning        
        selected_action = np.argmax(np.tensordot(self.reward_table, self.Dir/np.sum(self.Dir), axes=([1], [0])))

        return selected_action

    def update(self, obs, actions, rewards, new_obs):
        """ Update weights of the agents belief about the other agents action selection. """
        _, b = actions
        
        self.Dir[b] += 1 # Update beliefs about adversary
    
    def get_Belief(self):
        """ Returns the Dirichlet distribution of the agents belief about other agents actions. """
        return self.Dir

class FPQwForgetAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions in a bayesian way, plus a discount to ignore distant observations.
    She represents Q-values in a tabular form, i.e., using a matrix Q.
    """
    # TODO: Check if this is correct
    # !!! So the level-0 agent the adversary who is assumed to simply choose actions according to p(b)?
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
        a0, _ = actions
        # TODO: Apply more advanced method for onehot encoding based on self.action_space
        OHE = np.array([[1,0],[0,1]]) # One hot encoding of actions

        self.prob = (1-self.alpha)*self.prob + self.alpha*OHE[a0] # Update beliefs about DM
        

##
class Level2QAgent(Agent):
    """
    A Q-learning agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
            Dir_B = self.enemy.get_Belief()
            # Normalize Dir_B
            Dir_B = Dir_B/np.sum(Dir_B)
            
            # TODO: Test for more states than 1
            # Calculate the mean value of adversarys Q-function with respect to Dir_B
            mean_values_QB_p_A_b = np.dot(QB[obs, :, :], Dir_B)
            
            # Calculate argmx a of the mean values of adversarys Q-function
            Agents_best_action = np.argmax(mean_values_QB_p_A_b)
            
            return Agents_best_action

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs )

        # We obtain opponent's next action using Q_B
        
        # Adversary's Q-function
        QB = self.enemy.get_Q_function()
        
        # Adversary's belief about DM's action (in form of weights)
        Dir_B = self.enemy.get_Belief()
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
        
class Level1DPAgent(Agent):
    """
    A value iteration agent that treats the other player as a level 0 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular form, i.e., using a matrix.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, reward_table, system_model):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alphaA = learning_rate
        self.alphaB = learning_rate
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        self.reward_table = reward_table
        self.system_model = system_model
        #self.gammaB = 0

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        ## Other agent
        self.enemy = Level0DPAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB, reward_table=self.reward_table, system_model=None)

        # This is the value function V(s,a)
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        
        # This is the inner term of value function for calculation of value in various contexts: 
        #   1. Update of the value function.
        #   2. Calculation of the value function for the action selection.
        #   3. Constraction of epsilon greedy for upper level agent

    def extract_epsilon_greedy(self): 
        """Extracts the epsilon-greedy policy from the level-0 agent"""
        other_agent_optimal_action = self.enemy.act(None)
        
        eps_greedy = np.ones(len(self.enemy_action_space)) * self.epsilonA / (len(self.enemy_action_space) - 1)
        eps_greedy[other_agent_optimal_action] = 1 - self.epsilonA
        
        return eps_greedy
    def act(self, obs):
        
        eps_greedy_policy_level_0_agent = self.extract_epsilon_greedy()
        
        aux1 = np.tensordot(self.V, eps_greedy_policy_level_0_agent, axes=([1], [0]))
        
        # TODO: subset of self.system model is a bxs matrix adjust after definition of system model in the environment
        aux2 = np.tensordot(aux1, self.system_model[obs,:,:,:], axes=([0], [3]))
        
        # TODO: If aux2 is redefined modify this also
        aux3 = np.tensordot(self.reward_table[obs, :, :] + self.gammaB*aux2, eps_greedy_policy_level_0_agent, axes=([1], [0]))
        
        selected_action = np.argmax(aux3)
        
        return selected_action
        
    def update(self, obs, actions, rewards, new_obs):
        """Level-1 value iteration update"""
        # Extract actions of the DM and the Adversary respectively
        a, b = actions
        
        # Update the level-0 agent (DM) with the observed actions of the adversary
        # This only updates the belief or level-0 agent (DM) about the adversarys actions
        self.enemy.update(None, [None, b], None, None)
        
        eps_greedy_policy_level_0_agent = self.extract_epsilon_greedy()
        
        aux1 = np.tensordot(self.V, eps_greedy_policy_level_0_agent, axes=([1], [0]))
        
        # TODO: subset of self.system model is a bxs matrix adjust after definition of system model in the environment
        aux2 = np.tensordot(aux1, self.system_model[obs,a,:,:], axes=([0], [3]))
        
        # Update the value function of the level-1 agent (ADV)
        self.V[obs, a] = np.max(self.reward_table[obs, a, :] + self.gammaB*aux2)
        

class Level2DPAgent(Agent):
    """
    A value iteration agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular form, i.e., using a matrix V.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, reward_table, system_model):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilonA = epsilon
        self.epsilonB = self.epsilonA
        self.gammaA = gamma
        self.gammaB = self.gammaA
        # Value iteration does not impement learning rate, leaving for coherence with definition of other agents
        self.learning_rate = None

        self.action_space = action_space
        self.enemy_action_space = enemy_action_space
        
        # Environment atributes know to the agent
        self.reward_table = reward_table
        self.system_model = system_model

        ## Other agent
        self.enemy = Level1DPAgent(self.enemy_action_space, self.action_space, self.n_states, learning_rate=None,
                                   epsilon=self.epsilonB, gamma=self.gammaB, reward_table = self.reward_table,
                                   system_model = self.system_model)

        # This is the value function V(s, b) (i.e, the supported DM value function)
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        
    def extract_epsilon_greedy(self): 
        """Extracts the epsilon-greedy policy from the level-1 agent for all possible states."""
        other_agent_chosen_action_for_each_state = np.zeros(self.n_states)
        for obs in range(self.n_states):
            other_agent_chosen_action_for_each_state[obs] = self.enemy.act(obs)
        
        eps_greedy = np.ones((self.n_states, len(self.enemy_action_space))) * self.epsilonB / (len(self.enemy_action_space) - 1)
        
        for obs in range(self.n_states):
            eps_greedy[obs, other_agent_chosen_action_for_each_state[obs]] = 1 - self.epsilonB
        
        return eps_greedy

    def act(self, obs):

            eps_greedy_policy_level_1_agent = self.extract_epsilon_greedy()
            
            aux1 = np.sum(self.V * eps_greedy_policy_level_1_agent, axis=1)
            
            # TODO: subset of self.system model is a bxs matrix adjust after definition of system model in the environment
            aux2 = np.tensordot(aux1, self.system_model[obs,:,:,:], axes=([0], [3]))
            
            # TODO: If aux2 is redefined modify this also
            aux3 = np.tensordot(self.reward_table[obs, :, :] + self.gammaA*aux2, eps_greedy_policy_level_1_agent[obs], axes=([2], [0]))
            
            selected_action = np.argmax(aux3)
            
            return selected_action

    def update(self, obs, actions, rewards, new_obs):
        """Update of the value function of level-2 agent"""
        # Extract actions of the DM and the Adversary respectively
        a, b = actions
        
        self.enemy.update(obs, [a,b], None, None)

        eps_greedy_policy_level_1_agent = self.extract_epsilon_greedy()
        
        aux1 = np.sum(self.V * eps_greedy_policy_level_1_agent, axis=1)
        
        # TODO: subset of self.system model is a axs matrix adjust after definition of system model in the environment
        aux2 = np.tensordot(aux1, self.system_model[obs,:,b,:], axes=([0], [3]))
        
        # Update the value function of the level-2 agent (DM)
        self.V[obs, b] = np.max(self.reward_table[obs, :, b] + self.gammaA*aux2)

        
class Level2QAgentSoftmax(Level2QAgent):
    """
    A Q-learning agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
    She represents Q-values in a tabular form, i.e., using a matrix Q.
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
