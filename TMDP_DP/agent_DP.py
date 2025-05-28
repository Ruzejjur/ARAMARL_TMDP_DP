"""
This module implements several agents. An agent is characterized by two methods:
 * act : implements the policy, i.e., it returns agent's decisions to interact in a MDP or Markov Game.
 * update : the learning mechanism of the agent.
"""

import numpy as np
from numpy.random import choice
from itertools import product
import copy

def softmax(x, beta=1.0):
    x = x - np.max(x)  # stability
    e_x = np.exp(beta * x)
    return e_x / np.sum(e_x)

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
    """ A vanilla Q-learning agent that applies softmax policy."""
    
    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None, beta=1):
        IndQLearningAgent.__init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space)
        
        self.beta = beta
        
    def act(self, obs=None):
        
        return choice(self.action_space, p=softmax(self.Q[obs,:],self.beta))
    
class Level1QAgent(Agent):
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
        self.Dir = np.ones((self.n_states, len(self.enemy_action_space)))

    def act(self, obs=None):
        """An epsilon-greedy policy with explicit opponent modelling"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.action_space[np.argmax(np.dot(self.Q[obs], self.Dir[obs]/np.sum(self.Dir[obs])))]

    def update(self, obs, actions, rewards, new_obs):
        
        a0, a1 = actions
        r0, _ = rewards

        self.Dir[obs, a1] += 1 # Update beliefs about adversary
        
        self.Q[obs, a0, a1] = (1 - self.alpha)*self.Q[obs, a0, a1] + self.alpha*(r0 + self.gamma*np.max(np.dot(self.Q[new_obs], self.Dir[new_obs]/np.sum(self.Dir[new_obs]))))
    
    def get_Q_function(self):
        """Returns the Q-function of the agent"""
        return self.Q
    
    def get_Belief(self):
        """Returns the Dirichlet distribution of the agent"""
        return self.Dir

class Level1QAgentSoftmax(Level1QAgent):
    """
    A Q-learning agent that treats the other player as a level 0 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular form, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, beta=1):
        Level1QAgent.__init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma)
        
        self.beta = beta
    
    def act(self, obs=None):
        """Softmax policy"""
        
        # Calculate the mean value of Q-function with respect to Dir
        mean_values_QA_p_A_b = np.dot(self.Q[obs], self.Dir[obs]/np.sum(self.Dir[obs]))
        
        # Calculate the softmax selection probabilities
        p_A_softmax_selection = softmax(mean_values_QA_p_A_b, self.beta)
    
        # Return calculated DM's best action using softmax policy
        return choice(self.action_space, p=p_A_softmax_selection)
        
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
        self.epsilonB = epsilon
        self.gammaA = gamma
        self.gammaB = gamma
        
        self.action_space = action_space
        self.enemy_action_space = enemy_action_space

        # This is the Q-function Q_A(s, a, b) (i.e, the supported DM Q-function)
        self.QA = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])
        
        # Initializing adversary agent
        self.enemy = Level1QAgent(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB)


    def act(self, obs=None):
        """An epsilon-greedy policy"""

        if np.random.rand() < self.epsilonA:
            return choice(self.action_space)
        else:
            # Adversary's Q-function
            QB = self.enemy.get_Q_function()
            
            # Adversary's belief about DM's action (in form of weights)
            Dir_B = self.enemy.get_Belief()
            
            # Calculate the mean value of adversarys Q-function with respect to Dir_B
            mean_values_QB_p_A_b = np.dot(QB[obs], Dir_B[obs]/np.sum(Dir_B[obs]))
            
            # Calculate argmx a of the mean values of adversarys Q-function
            Adversary_best_action = np.argmax(mean_values_QB_p_A_b)
            
            # Calculate DM's belief about adversary's action using epsilon-greedy policy
            p_A_b = np.ones(len(self.enemy_action_space))
            p_A_b.fill(self.epsilonB/(len(self.enemy_action_space)-1))
            p_A_b[Adversary_best_action] = 1 - self.epsilonB
            
            # Calculated DM's best action using epsion-greedy policy
            Agent_best_action = np.argmax(np.dot(self.QA[obs], p_A_b))
            
            return Agent_best_action

    def update(self, obs, actions, rewards, new_obs):
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs)

        # We obtain opponent's next action (b') using Q_B
        
        # Adversary's Q-function
        QB = self.enemy.get_Q_function()
        
        # Adversary's belief about DM's action (in form of weights)
        Dir_B = self.enemy.get_Belief()
        
        # Calculate the mean value of adversarys Q-function with respect to Dir_B
        mean_values_QB_p_B_a = np.dot(QB[new_obs], Dir_B[new_obs]/np.sum(Dir_B[new_obs]))
        
        # Calculate argmx b of the mean values of adversarys Q-function
        Adversary_best_action = np.argmax(mean_values_QB_p_B_a)
        
        # Calculate DM's belief about adversary's action using epsilon-greedy policy
        p_A_b = np.ones(len(self.enemy_action_space))
        p_A_b.fill(self.epsilonB/(len(self.enemy_action_space)-1))
        p_A_b[Adversary_best_action] = 1 - self.epsilonB
        
        # Calculate the mean value of DM's Q-function with respect to DM's belief about adversary's actions p_A_b
        mean_values_QA_p_A_b = np.dot(self.QA[new_obs], p_A_b)
        
        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(mean_values_QA_p_A_b))


class Level2QAgentSoftmax(Level2QAgent):
    """
    A Q-learning agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their Q function.
    She represents Q-values in a tabular form, i.e., using a matrix Q.
    """

    def __init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, beta=1):
        Level2QAgent.__init__(self, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma)
        
        self.beta = beta
        
        # Initializing adversary agent
        self.enemy = Level1QAgentSoftmax(self.enemy_action_space, self.action_space, self.n_states,
            learning_rate=self.alphaB, epsilon=self.epsilonB, gamma=self.gammaB, beta=beta)
    
    def act(self, obs=None):
        """Softmax policy"""

        # Adversary's Q-function
        QB = self.enemy.get_Q_function()
        
        # Adversary's belief about DM's action (in form of weights)
        Dir_B = self.enemy.get_Belief()
        
        # Calculate the mean value of adversarys Q-function with respect to Dir_B
        mean_values_QB_p_B_a = np.dot(QB[obs], Dir_B[obs]/np.sum(Dir_B[obs]))
        
        p_A_b = softmax(mean_values_QB_p_B_a, self.beta)
        
        mean_values_QA_p_A_b = np.dot(self.QA[obs], p_A_b)
        
        p_A_softmax_selection = softmax(mean_values_QA_p_A_b, self.beta)
        
        # Return calculated DM's best action using softmax policy
        return choice(self.action_space, p=p_A_softmax_selection)
    
    def update(self, obs, actions, rewards, new_obs):
        a, b = actions
        rA, rB = rewards

        self.enemy.update(obs, [b,a], [rB, rA], new_obs)

        # We obtain opponent's next action (b') using Q_B
        
        # Adversary's Q-function
        QB = self.enemy.get_Q_function()
        
        # Adversary's belief about DM's action (in form of weights)
        Dir_B = self.enemy.get_Belief()
        
        # Calculate the mean value of adversarys Q-function with respect to Dir_B
        mean_values_QB_p_B_a = np.dot(QB[new_obs], Dir_B[new_obs]/np.sum(Dir_B[new_obs]))
        
        p_A_b = softmax(mean_values_QB_p_B_a, self.beta)
        
        # Calculate the mean value of DM's Q-function with respect to DM's belief about adversary's actions p_A_b
        mean_values_QA_p_A_b = np.dot(self.QA[new_obs], p_A_b)
        
        # Finally we update the supported agent's Q-function
        self.QA[obs, a, b] = (1 - self.alphaA)*self.QA[obs, a, b] + self.alphaA*(rA + self.gammaA*np.max(mean_values_QA_p_A_b))
        
        
### ============ Dynamic programming agents - value iteration ============


class Level1DPAgent(Agent):
    """
    A value iteration agent that treats the other player as a level 0 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular form, i.e., using a matrix.
    """

    def __init__(self, action_space, enemy_action_space, n_states, epsilon, gamma, env):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.enemy_action_space = enemy_action_space
        
        # This is the value function V(s,a)
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        
        # Parameters of the Dirichlet distribution used to model the other agent
        # Initialized using a uniform prior
        self.Dir = np.ones((self.n_states, len(self.enemy_action_space)))
        
        # Initialize an empty environment for simulation of steps 
        #* Warning: Do not copy env to many times -> extreme memory and computational overhead (garbage collection)
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.blue_player_execution_prob = 1
        self.env_snapshot.red_player_execution_prob = 1
        
        
        # All combination of radix encoded actions of DM and Adv
        grid_A, grid_B = np.meshgrid(np.array(range(len(self.env_snapshot.combined_actions_blue))), np.array(range(len(self.env_snapshot.combined_actions_red))))
        self.DM_Adv_act_combination = np.column_stack([grid_A.ravel(), grid_B.ravel()])
    
    def reset_sim_env(self,env):
        """
        Resets the simulated environment to its initial state:
        - Step counter is set to 0
        - Player and coin positions are restored to their starting values
        """
        
        # Reset the environment
        self.env_snapshot.reset()
        
        # Player positions
        self.env_snapshot.blue_player = env.blue_player.copy() 
        self.env_snapshot.red_player = env.red_player.copy()
        
        # Coin availability
        self.env_snapshot.coin1_available = env.coin1_available.copy()
        self.env_snapshot.coin2_available = env.coin2_available.copy()
        
        # Player coin collection
        self.env_snapshot.blue_collected_coin1 = env.blue_collected_coin1.copy()
        self.env_snapshot.blue_collected_coin2 = env.blue_collected_coin2.copy()
        self.env_snapshot.red_collected_coin1 = env.red_collected_coin1.copy()
        self.env_snapshot.red_collected_coin2 = env.red_collected_coin2.copy()
        
        
    def act(self, obs, env):
        
        ## Setup the simulated environment to identical state as the actual one 
        self.reset_sim_env(env)
        
        ## Simulate movement and collect rewards for each action
        
        # Initialize array for simulated rewards for all possible DM and Adv actions
        DM_rewards_for_act_comb = np.zeros((len(self.env_snapshot.combined_actions_blue), len(self.env_snapshot.combined_actions_red)))
        
        # Perform all possible actions of DM and Adv
        for act_comb in self.DM_Adv_act_combination: 
            # Perform steps for DM and Adv respectively and collect rewards
            _, rewards, _ = self.env_snapshot.step(act_comb)
            
            # Save only DM's reward
            DM_rewards_for_act_comb[act_comb[0], act_comb[1]] = rewards[0]
            
            # Reset simulated environment to the current real state
            self.reset_sim_env(env)
        
        ## Construct movement model for all possible actions action
        
        # Collect indicies of states to which the movement is possible
        
        # TODO: There is a better way to do this by preinitializing what we know
        probab = np.zeros((self.n_states, len(self.env_snapshot.combined_actions_blue), len(self.env_snapshot.combined_actions_red)))
        
        # TODO: Combine this with for cycle above
        # Perform all possible actions of DM and Adv
        
        for act_comb in self.DM_Adv_act_combination:
            s_new_intended, _ = self.env_snapshot.step(act_comb)
            
            
            
            probab[s_new_intended, act_comb[0], act_comb[1]] = env.blue_player_execution_prob*env.red_player_execution_prob
            
            # Reset simulated environment to current real state
            self.reset_sim_env(env)
            
            for act_comb_other in self.DM_Adv_act_combination: 
                if not np.array_equal(act_comb,act_comb_other): 
                    s_new_non_intended, _ = self.env_snapshot.step(act_comb_other)
                    probab[s_new_non_intended, act_comb[0], act_comb[1]] = (1 - env.blue_player_execution_prob*env.red_player_execution_prob)/len(self.DM_Adv_act_combination)

                    # Reset simulated environment to current real state
                    self.reset_sim_env(env)
                    
        ## Perform action selection
        
        return np.argmax(np.dot(DM_rewards_for_act_comb, self.Dir[obs]/np.sum(self.Dir[obs]))+self.gamma*(np.dot(np.dot(self.V,self.Dir/np.sum(self.Dir)), probab)))
        
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

    