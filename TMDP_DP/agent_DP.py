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

    def act(self, obs, env):
        """
        This implements the policy, \pi : S -> A.
        obs is the observed state s
        """
        raise NotImplementedError()

    def update(self, obs, actions, rewards, new_obs, env):
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

    def act(self, obs, env):

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

    def act(self, obs, env):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[np.argmax(self.Q[obs, :])]

    def update(self, obs, actions, rewards, new_obs, env=None):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))
        
class IndQLearningAgentSoftmax(IndQLearningAgent):
    """ A vanilla Q-learning agent that applies softmax policy."""
    
    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None, beta=1):
        IndQLearningAgent.__init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space)
        
        self.beta = beta
        
    def act(self, obs, env):
        
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

    def act(self, obs, env):
        """An epsilon-greedy policy with explicit opponent modelling"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.action_space[np.argmax(np.dot(self.Q[obs], self.Dir[obs]/np.sum(self.Dir[obs])))]

    def update(self, obs, actions, rewards, new_obs, env=None):
        
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
    
    def act(self, obs, env):
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


    def act(self, obs, env):
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

    def update(self, obs, actions, rewards, new_obs, env=None):
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
    
    def act(self, obs, env):
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
    
    def update(self, obs, actions, rewards, new_obs, env=None):
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


class Level1DPAgent_Stationary(Agent):
    """
    A value iteration agent that treats the other player as a level 0 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular form, i.e., using a matrix.
    Stationary in this case means that the agent has full information about transition distribution and precalculates it befefore running act() and update()
    """

    def __init__(self, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.player_id = player_id
        
        # Flag to ensure simulation and expected value calculation runs only once per step
        self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = False
        
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
        
        # Calculate P(edm,eadv | idm,iadv) - the 4D execution probability tensor
        
        # Break down possible combined actions of DM and Adversary into move and push
        if self.player_id == 0:
            DM_action_details = self.env_snapshot.combined_actions_blue
            Adv_action_details = self.env_snapshot.combined_actions_red
            
            DM_available_move_actions_num = len(env.available_move_actions_DM)
            Adv_available_move_actions_num = len(env.available_move_actions_Adv)
            
            # Saving number of DM and Adv actions
            self.num_DM_actions = len(self.env_snapshot.combined_actions_blue)
            self.num_Adv_actions = len(self.env_snapshot.combined_actions_red)
            
            
            
            DM_execution_prob = env.blue_player_execution_prob
            Adv_execution_prob = env.red_player_execution_prob
            
        else: 
            DM_action_details = self.env_snapshot.combined_actions_red
            Adv_action_details = self.env_snapshot.combined_actions_blue
            
            DM_available_move_actions_num = len(env.available_move_actions_Adv)
            Adv_available_move_actions_num = len(env.available_move_actions_DM)
            
            # Saving number of DM and Adv actions
            self.num_DM_actions = len(self.env_snapshot.combined_actions_red)
            self.num_Adv_actions = len(self.env_snapshot.combined_actions_blue)
            
            DM_execution_prob = env.red_player_execution_prob
            Adv_execution_prob = env.blue_player_execution_prob
            
        # Initialize array for executed action reward (executed action == moving to state s')
        self.DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        # Initialize array for calculation of E[V(s',b')|s']
        self.future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        
        # Pre-calculate the state transition probabilities based on execution probabilities
        self.prob_exec_tensor = self._calculate_execution_probabilities(DM_action_details, Adv_action_details,
                                         DM_execution_prob, Adv_execution_prob,
                                         DM_available_move_actions_num, Adv_available_move_actions_num)
        
 
        
    def reset_sim_env(self, obs):
        """
        Resets the simulated environment to the state represented by 'obs'.
        - Player and coin positions and collection status are restored from 'obs'.
        - Step counter is reset (as per self.env_snapshot.reset()).
        """
        # is implicitly called by setting player/coin positions
        self.env_snapshot.reset()

        # Radix decoding of the state ID 'obs'
        base_pos = self.env_snapshot.N * self.env_snapshot.N
        base_coll = 2  # 0 or 1 for each coin

        # Decode collected coin flags (in reverse order of encoding)
        self.env_snapshot.red_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.red_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin1 = bool(obs % base_coll) 
        obs //= base_coll

        # Decode player positions
        p2_flat = obs % base_pos  # Red player's flattened position
        obs //= base_pos
        
        p1_flat = obs  # Blue player's flattened position

        # Convert flattened positions back to 2D coordinates
        # p_flat = row + N * col
        blue_player_col = p1_flat // self.env_snapshot.N
        blue_player_row = p1_flat % self.env_snapshot.N
        
        # Setting blue player to coordinates extracted from state obs
        self.env_snapshot.blue_player = np.array([blue_player_row, blue_player_col])

        red_player_col = p2_flat // self.env_snapshot.N
        red_player_row = p2_flat % self.env_snapshot.N
        
        # Setting red player to coordinates extracted from state obs
        self.env_snapshot.red_player = np.array([red_player_row, red_player_col])
        
        # Set coin availability based on decoded collection status
        self.env_snapshot.coin1_available = not (self.env_snapshot.blue_collected_coin1 or self.env_snapshot.red_collected_coin1)
        self.env_snapshot.coin2_available = not (self.env_snapshot.blue_collected_coin2 or self.env_snapshot.red_collected_coin2)
    
    def _calculate_execution_probabilities(self,DM_action_details, Adv_action_details,
                                         DM_execution_prob, Adv_execution_prob,
                                         DM_available_move_actions_num, Adv_available_move_actions_num):
        """
        Pre-calculates the 4D tensor of execution probabilities:
        P[idm, iadv, edm, eadv] = P(edm, eadv | idm, iadv)
        where i = intended, e = executed, dm = decision maker, adv = adversary.
        """
        
        DM_moves = DM_action_details[:, 0]
        DM_pushes = DM_action_details[:, 1]
        Adv_moves = Adv_action_details[:, 0]
        Adv_pushes = Adv_action_details[:, 1]
        
        # Initialize matrix for storing probabilities of action executing conditioned by intended action for DM (intended x executed)
        prob_DM_part = np.zeros((self.num_DM_actions, self.num_DM_actions)) # P(edm|idm)
        
        # Creating a boolean matrix of intended x executed moves for probability assignment
        DM_moves_match = (DM_moves[:, np.newaxis] == DM_moves[np.newaxis, :])
        
        # Creating a boolean matrix of intended x executed pushes for probability assignment
        DM_pushes_match = (DM_pushes[:, np.newaxis] == DM_pushes[np.newaxis, :])
        
        # Setting probability for intended move AND push match executed move AND push
        prob_DM_part[DM_moves_match & DM_pushes_match] = DM_execution_prob

        # Setting probability for Intended push matches executed push, BUT intended move DOES NOT match executed move
        num_alt_DM = DM_available_move_actions_num - 1
        
        # Setting probability of unintended moves
        prob_DM_part[~DM_moves_match & DM_pushes_match] = (1.0 - DM_execution_prob) / num_alt_DM
        
        # Other cases stay zero

        # Initialize matrix for storing probabilities of action executing conditioned by intended action for Adv (intended x executed)
        prob_Adv_part = np.zeros((self.num_Adv_actions, self.num_Adv_actions))
        
        # Creating a boolean matrix of intended x executed moves for probability assignment
        Adv_moves_match = (Adv_moves[:, np.newaxis] == Adv_moves[np.newaxis, :])
        
        # Creating a boolean matrix of intended x executed pushes for probability assignment
        Adv_pushes_match = (Adv_pushes[:, np.newaxis] == Adv_pushes[np.newaxis, :])
        
        # Setting probability for intended move AND push match executed move AND push
        prob_Adv_part[Adv_moves_match & Adv_pushes_match] = Adv_execution_prob

        # Setting probability for Intended push matches executed push, BUT intended move DOES NOT match executed move
        num_alt_Adv = Adv_available_move_actions_num - 1
        
        # Setting probability of unintended moves
        prob_Adv_part[~Adv_moves_match & Adv_pushes_match] = (1.0 - Adv_execution_prob) / num_alt_Adv
        
        # Other cases stay zero
        
        # Combine to get 4D prob_exec_tensor: P[idm, iadv, edm, eadv]
        # containing joint probabilities of Adv and DM executing an action conditioned by intended action
        prob_exec_tensor = prob_DM_part[:, np.newaxis, :, np.newaxis] * prob_Adv_part[np.newaxis, :, np.newaxis, :]
        
        return prob_exec_tensor
    
    def _simulate_executed_outcomes(self, obs):
        """
        Simulates all possible executed action pairs from the given state 'obs'
        and returns the resulting next states and DM rewards.
        """
    
        # Run only if it was not already ran in this episode
        if not self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag:
            ## Determine reachable next states and their immediate rewards
            self.reset_sim_env(obs)
            executed_action_outcomes = {}  # (exec_dm_idx, exec_adv_idx) -> (s_prime, r_DM)
            actual_next_states_set = set() # To store unique s_prime values

            # Execute all actions with probab 1 to get reachable states and their rewards
            for DM_exec_idx in range(self.num_DM_actions):
                for Adv_exec_idx in range(self.num_Adv_actions):
                    act_comb_executed = (DM_exec_idx, Adv_exec_idx)
                    s_prime, rewards_vec, _ = self.env_snapshot.step(act_comb_executed)
                    
                    if self.player_id == 0:
                        DM_reward_for_exec = rewards_vec[0]
                    else: 
                        DM_reward_for_exec = rewards_vec[1]
                        
                    executed_action_outcomes[act_comb_executed] = (s_prime, DM_reward_for_exec)
                    actual_next_states_set.add(s_prime)
                    
                    self.reset_sim_env(obs) # Reset for the next simulation iteration
            
            self._calculate_expected_future_value(actual_next_states_set, executed_action_outcomes)
            
    def _calculate_expected_future_value(self, actual_next_states_set, executed_action_outcomes):
        """
        Simulates all possible executed action pairs from the given state 'obs',
        calculates the resulting next states and DM rewards, and computes
        E[V(s',b')|s'] for each reachable s'.
        """
        
        unique_s_primes = np.array(list(actual_next_states_set), dtype=int)

        ## Calculate E[V(s',b')|s'] only for unique_s_primes
        s_prime_to_expected_V = {}

        # Index V and Dir only for the relevant next states
        V_relevant = self.V[unique_s_primes, :]    # Shape: (len(unique_s_primes), num_Adv_actions)
        Dir_relevant = self.Dir[unique_s_primes, :] # Shape: (len(unique_s_primes), num_Adv_actions)

        # Calculate normalization constant only for s which are relevant
        dir_sum_relevant = np.sum(Dir_relevant, axis=1, keepdims=True)
        
        # Normalizing Dirichlet distribution only for relevant s
        beliefs_Adv_action_relevant_s_prime = Dir_relevant / dir_sum_relevant
        
        # Calculating mean value of b's for each s. This part is not dependent on transition model and can be precalculated.
        expected_V_values_for_unique_s_primes = np.sum(V_relevant * beliefs_Adv_action_relevant_s_prime, axis=1)
        
        # Save into dictionary for later lookup
        for i, s_prime_idx in enumerate(unique_s_primes):
            s_prime_to_expected_V[s_prime_idx] = expected_V_values_for_unique_s_primes[i]

        # --- Prepare arrays of R(edm,eadv) and E[V(s'_from_edm,eadv)] ---
        # These are rewards/values for *executed* actions.
        DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions)) 

        for DM_exec_idx in range(self.num_DM_actions):
            for Adv_exec_idx in range(self.num_Adv_actions):
                # Get next state and reward of DM for executed actions
                s_prime, r_DM = executed_action_outcomes[(DM_exec_idx, Adv_exec_idx)]
                DM_rewards_executed_array[DM_exec_idx, Adv_exec_idx] = r_DM
                # Get E[V(s')|s'] from our map, default to 0 if s_prime somehow not found 
                future_V_values_executed_array[DM_exec_idx, Adv_exec_idx] = s_prime_to_expected_V.get(s_prime, 0.0)
        
        self.DM_rewards_executed_array = DM_rewards_executed_array
        self.future_V_values_executed_array = future_V_values_executed_array
    
    def optim_act(self, obs): 
            
            # Perform sumproducts using np.einsum
            
            # Calculate expected reward r(s,a,b,s') with respect to p(s'|s,a,b)
            expected_DM_rewards = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, self.DM_rewards_executed_array)

            # Calculate the expectations of E[V(s',b')|s'] with respect to p(s'|s,a,b)
            weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, self.future_V_values_executed_array)

            # Final Action Selection
            p_b_given_s_obs = self.Dir[obs] / np.sum(self.Dir[obs])
            total_action_values = np.dot(expected_DM_rewards, p_b_given_s_obs) + self.gamma * np.dot(weighted_sum_future_V, p_b_given_s_obs)
            
            # Find indicies (actions) of all max values
            max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))

            # Choose randomly among them
            chosen_action = np.random.choice(max_indices)
            
            return chosen_action
    
    def act(self, obs, env):
        "Epsilon greedy action selection strategy."
        
        self._simulate_executed_outcomes(obs)
        # Set flag to True
        self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = True
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.optim_act(obs)
        
    def update(self, obs, actions, rewards, new_obs, env):
            # obs: current state where actions were taken
            # actions: tuple [a_dm_actually_taken, b_opp_actually_taken]
            # rewards, new_obs: not directly used for this type of V(s,b_opp) update,
            #                   as it's a model-based VI-like update based on current V and model.

            if self.player_id == 0:
                b_opp_taken_in_obs = actions[1] # The opponent's action we observed in 'obs'
            else: 
                b_opp_taken_in_obs = actions[0] # The opponent's action we observed in 'obs'

            # --- Calculate E[V(s',b_opp')|s'] ---
            # This is the expected value of next states, considering opponent's policy in those next states.
            # This code is largely the same as in act().
            
            # We already calculated this part in the act method which is ran 
            
            self._simulate_executed_outcomes(obs)

            # Calculate Q(obs, a_dm_intended, b_opp_taken_in_obs) for ALL a_dm_intended ---
            # We need to find the best DM response 'a_dm_intended' if opponent plays 'b_opp_taken_in_obs'.
            
            # prob_exec_tensor_for_fixed_b[idm, edm, eadv] = P(edm,eadv | idm, iadv=b_opp_taken_in_obs)
            prob_exec_tensor_for_fixed_b = self.prob_exec_tensor[:, b_opp_taken_in_obs, :, :]
            
            # E[Reward | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * R(edm,eadv)
            # Output shape: (num_DM_actions_intended,)
            expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, self.DM_rewards_executed_array)
            
            # E[Future_V | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * E[V(s'_from_edm,eadv)]
            # Output shape: (num_DM_actions_intended,)
            expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, self.future_V_values_executed_array)
            
            # Q_values[idm] = Q(obs, idm, b_opp_taken_in_obs)
            Q_values_for_dm_intentions = expected_rewards_all_idm + self.gamma * expected_future_V_all_idm
            
            # Update V(obs, b_opp_taken_in_obs) with the max Q-value (DM plays optimally against b_opp_taken_in_obs)
            self.V[obs, b_opp_taken_in_obs] = np.max(Q_values_for_dm_intentions)

            # Update the Dirichlet model for the opponent's observed action
            self.Dir[obs, b_opp_taken_in_obs] += 1         
            
            # Set flag for simulation of outcomes to false for next episode 
            self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = False      



class Level1DPAgent_NonStationary(Agent):
    """
    A value iteration agent that treats the other player as a level 0 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular form, i.e., using a matrix.
    NonStationary in this case means that the agent has full information about transition distribution which changes durring game and  she calculates it at each episode
    """

    def __init__(self, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.player_id = player_id
        
        # Flag to ensure simulation and expected value calculation runs only once per step
        self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = False
        
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
        
        # Calculate P(edm,eadv | idm,iadv) - the 4D execution probability tensor
        
        # Break down possible combined actions of DM and Adversary into move and push
        if self.player_id == 0:
            self.DM_action_details = self.env_snapshot.combined_actions_blue
            self.Adv_action_details = self.env_snapshot.combined_actions_red
            
            self.DM_available_move_actions_num = len(env.available_move_actions_DM)
            self.Adv_available_move_actions_num = len(env.available_move_actions_Adv)
            
            # Saving number of DM and Adv actions
            self.num_DM_actions = len(self.env_snapshot.combined_actions_blue)
            self.num_Adv_actions = len(self.env_snapshot.combined_actions_red)
            
        else: 
            self.DM_action_details = self.env_snapshot.combined_actions_red
            self.Adv_action_details = self.env_snapshot.combined_actions_blue
            
            self.DM_available_move_actions_num  = len(env.available_move_actions_Adv)
            self.Adv_available_move_actions_num = len(env.available_move_actions_DM)
            
            # Saving number of DM and Adv actions
            self.num_DM_actions = len(self.env_snapshot.combined_actions_red)
            self.num_Adv_actions = len(self.env_snapshot.combined_actions_blue)
            
            
        # Initialize array for executed action reward (executed action == moving to state s')
        self.DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        # Initialize array for calculation of E[V(s',b')|s']
        self.future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
 
        
    def reset_sim_env(self, obs):
        """
        Resets the simulated environment to the state represented by 'obs'.
        - Player and coin positions and collection status are restored from 'obs'.
        - Step counter is reset (as per self.env_snapshot.reset()).
        """
        # is implicitly called by setting player/coin positions
        self.env_snapshot.reset()

        # Radix decoding of the state ID 'obs'
        base_pos = self.env_snapshot.N * self.env_snapshot.N
        base_coll = 2  # 0 or 1 for each coin

        # Decode collected coin flags (in reverse order of encoding)
        self.env_snapshot.red_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.red_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin1 = bool(obs % base_coll) 
        obs //= base_coll

        # Decode player positions
        p2_flat = obs % base_pos  # Red player's flattened position
        obs //= base_pos
        
        p1_flat = obs  # Blue player's flattened position

        # Convert flattened positions back to 2D coordinates
        # p_flat = row + N * col
        blue_player_col = p1_flat // self.env_snapshot.N
        blue_player_row = p1_flat % self.env_snapshot.N
        
        # Setting blue player to coordinates extracted from state obs
        self.env_snapshot.blue_player = np.array([blue_player_row, blue_player_col])

        red_player_col = p2_flat // self.env_snapshot.N
        red_player_row = p2_flat % self.env_snapshot.N
        
        # Setting red player to coordinates extracted from state obs
        self.env_snapshot.red_player = np.array([red_player_row, red_player_col])
        
        # Set coin availability based on decoded collection status
        self.env_snapshot.coin1_available = not (self.env_snapshot.blue_collected_coin1 or self.env_snapshot.red_collected_coin1)
        self.env_snapshot.coin2_available = not (self.env_snapshot.blue_collected_coin2 or self.env_snapshot.red_collected_coin2)
    
    def _calculate_execution_probabilities(self,env):
        """
        Pre-calculates the 4D tensor of execution probabilities:
        P[idm, iadv, edm, eadv] = P(edm, eadv | idm, iadv)
        where i = intended, e = executed, dm = decision maker, adv = adversary.
        """
        
        # Break down possible combined actions of DM and Adversary into move and push
        if self.player_id == 0:
            
            DM_execution_prob = env.blue_player_execution_prob
            Adv_execution_prob = env.red_player_execution_prob
            
        else: 
            
            DM_execution_prob = env.red_player_execution_prob
            Adv_execution_prob = env.blue_player_execution_prob
        
        DM_moves = self.DM_action_details[:, 0]
        DM_pushes = self.DM_action_details[:, 1]
        Adv_moves = self.Adv_action_details[:, 0]
        Adv_pushes = self.Adv_action_details[:, 1]
        
        # Initialize matrix for storing probabilities of action executing conditioned by intended action for DM (intended x executed)
        prob_DM_part = np.zeros((self.num_DM_actions, self.num_DM_actions)) # P(edm|idm)
        
        # Creating a boolean matrix of intended x executed moves for probability assignment
        DM_moves_match = (DM_moves[:, np.newaxis] == DM_moves[np.newaxis, :])
        
        # Creating a boolean matrix of intended x executed pushes for probability assignment
        DM_pushes_match = (DM_pushes[:, np.newaxis] == DM_pushes[np.newaxis, :])
        
        # Setting probability for intended move AND push match executed move AND push
        prob_DM_part[DM_moves_match & DM_pushes_match] = DM_execution_prob

        # Setting probability for Intended push matches executed push, BUT intended move DOES NOT match executed move
        num_alt_DM = self.DM_available_move_actions_num - 1
        
        # Setting probability of unintended moves
        prob_DM_part[~DM_moves_match & DM_pushes_match] = (1.0 - DM_execution_prob) / num_alt_DM
        
        # Other cases stay zero

        # Initialize matrix for storing probabilities of action executing conditioned by intended action for Adv (intended x executed)
        prob_Adv_part = np.zeros((self.num_Adv_actions, self.num_Adv_actions))
        
        # Creating a boolean matrix of intended x executed moves for probability assignment
        Adv_moves_match = (Adv_moves[:, np.newaxis] == Adv_moves[np.newaxis, :])
        
        # Creating a boolean matrix of intended x executed pushes for probability assignment
        Adv_pushes_match = (Adv_pushes[:, np.newaxis] == Adv_pushes[np.newaxis, :])
        
        # Setting probability for intended move AND push match executed move AND push
        prob_Adv_part[Adv_moves_match & Adv_pushes_match] = Adv_execution_prob

        # Setting probability for Intended push matches executed push, BUT intended move DOES NOT match executed move
        num_alt_Adv = self.Adv_available_move_actions_num - 1
        
        # Setting probability of unintended moves
        prob_Adv_part[~Adv_moves_match & Adv_pushes_match] = (1.0 - Adv_execution_prob) / num_alt_Adv
        
        # Other cases stay zero
        
        # Combine to get 4D prob_exec_tensor: P[idm, iadv, edm, eadv]
        # containing joint probabilities of Adv and DM executing an action conditioned by intended action
        prob_exec_tensor = prob_DM_part[:, np.newaxis, :, np.newaxis] * prob_Adv_part[np.newaxis, :, np.newaxis, :]
        
        return prob_exec_tensor
    
    def _simulate_executed_outcomes(self, obs):
        """
        Simulates all possible executed action pairs from the given state 'obs'
        and returns the resulting next states and DM rewards.
        """
    
        # Run only if it was not already ran in this episode
        if not self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag:
            ## Determine reachable next states and their immediate rewards
            self.reset_sim_env(obs)
            executed_action_outcomes = {}  # (exec_dm_idx, exec_adv_idx) -> (s_prime, r_DM)
            actual_next_states_set = set() # To store unique s_prime values

            # Execute all actions with probab 1 to get reachable states and their rewards
            for DM_exec_idx in range(self.num_DM_actions):
                for Adv_exec_idx in range(self.num_Adv_actions):
                    act_comb_executed = (DM_exec_idx, Adv_exec_idx)
                    s_prime, rewards_vec, _ = self.env_snapshot.step(act_comb_executed)
                    
                    if self.player_id == 0:
                        DM_reward_for_exec = rewards_vec[0]
                    else: 
                        DM_reward_for_exec = rewards_vec[1]
                        
                    executed_action_outcomes[act_comb_executed] = (s_prime, DM_reward_for_exec)
                    actual_next_states_set.add(s_prime)
                    
                    self.reset_sim_env(obs) # Reset for the next simulation iteration
            
            self._calculate_expected_future_value(actual_next_states_set, executed_action_outcomes)
            
    def _calculate_expected_future_value(self, actual_next_states_set, executed_action_outcomes):
        """
        Simulates all possible executed action pairs from the given state 'obs',
        calculates the resulting next states and DM rewards, and computes
        E[V(s',b')|s'] for each reachable s'.
        """
        
        unique_s_primes = np.array(list(actual_next_states_set), dtype=int)

        ## Calculate E[V(s',b')|s'] only for unique_s_primes
        s_prime_to_expected_V = {}

        # Index V and Dir only for the relevant next states
        V_relevant = self.V[unique_s_primes, :]    # Shape: (len(unique_s_primes), num_Adv_actions)
        Dir_relevant = self.Dir[unique_s_primes, :] # Shape: (len(unique_s_primes), num_Adv_actions)

        # Calculate normalization constant only for s which are relevant
        dir_sum_relevant = np.sum(Dir_relevant, axis=1, keepdims=True)
        
        # Normalizing Dirichlet distribution only for relevant s
        beliefs_Adv_action_relevant_s_prime = Dir_relevant / dir_sum_relevant
        
        # Calculating mean value of b's for each s. This part is not dependent on transition model and can be precalculated.
        expected_V_values_for_unique_s_primes = np.sum(V_relevant * beliefs_Adv_action_relevant_s_prime, axis=1)
        
        # Save into dictionary for later lookup
        for i, s_prime_idx in enumerate(unique_s_primes):
            s_prime_to_expected_V[s_prime_idx] = expected_V_values_for_unique_s_primes[i]

        # --- Prepare arrays of R(edm,eadv) and E[V(s'_from_edm,eadv)] ---
        # These are rewards/values for *executed* actions.
        DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions)) 

        for DM_exec_idx in range(self.num_DM_actions):
            for Adv_exec_idx in range(self.num_Adv_actions):
                # Get next state and reward of DM for executed actions
                s_prime, r_DM = executed_action_outcomes[(DM_exec_idx, Adv_exec_idx)]
                DM_rewards_executed_array[DM_exec_idx, Adv_exec_idx] = r_DM
                # Get E[V(s')|s'] from our map, default to 0 if s_prime somehow not found 
                future_V_values_executed_array[DM_exec_idx, Adv_exec_idx] = s_prime_to_expected_V.get(s_prime, 0.0)
        
        self.DM_rewards_executed_array = DM_rewards_executed_array
        self.future_V_values_executed_array = future_V_values_executed_array
    
    def optim_act(self, obs): 
            
            # Perform sumproducts using np.einsum
            
            # Calculate expected reward r(s,a,b,s') with respect to p(s'|s,a,b)
            expected_DM_rewards = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, self.DM_rewards_executed_array)

            # Calculate the expectations of E[V(s',b')|s'] with respect to p(s'|s,a,b)
            weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, self.future_V_values_executed_array)

            # Final Action Selection
            p_b_given_s_obs = self.Dir[obs] / np.sum(self.Dir[obs])
            total_action_values = np.dot(expected_DM_rewards, p_b_given_s_obs) + self.gamma * np.dot(weighted_sum_future_V, p_b_given_s_obs)
            
            # Find indicies (actions) of all max values
            max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))

            # Choose randomly among them
            chosen_action = np.random.choice(max_indices)
            
            return chosen_action
    
    def act(self, obs, env):
        "Epsilon greedy action selection strategy."
            
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)
                                         
        
        self._simulate_executed_outcomes(obs)
        
        # Set flag to True
        self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = True
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.optim_act(obs)
        
    def update(self, obs, actions, rewards, new_obs, env):
            # obs: current state where actions were taken
            # actions: tuple [a_dm_actually_taken, b_opp_actually_taken]
            # rewards, new_obs: not directly used for this type of V(s,b_opp) update,
            #                   as it's a model-based VI-like update based on current V and model.

            if self.player_id == 0:
                b_opp_taken_in_obs = actions[1] # The opponent's action we observed in 'obs'
            else: 
                b_opp_taken_in_obs = actions[0] # The opponent's action we observed in 'obs'

            # --- Calculate E[V(s',b_opp')|s'] ---
            # This is the expected value of next states, considering opponent's policy in those next states.
            # This code is largely the same as in act().
            
            # We already calculated this part in the act method which is ran 
            
            self._simulate_executed_outcomes(obs)

            # Calculate Q(obs, a_dm_intended, b_opp_taken_in_obs) for ALL a_dm_intended ---
            # We need to find the best DM response 'a_dm_intended' if opponent plays 'b_opp_taken_in_obs'.
            
            # prob_exec_tensor_for_fixed_b[idm, edm, eadv] = P(edm,eadv | idm, iadv=b_opp_taken_in_obs)
            prob_exec_tensor_for_fixed_b = self.prob_exec_tensor[:, b_opp_taken_in_obs, :, :]
            
            # E[Reward | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * R(edm,eadv)
            # Output shape: (num_DM_actions_intended,)
            expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, self.DM_rewards_executed_array)
            
            # E[Future_V | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * E[V(s'_from_edm,eadv)]
            # Output shape: (num_DM_actions_intended,)
            expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, self.future_V_values_executed_array)
            
            # Q_values[idm] = Q(obs, idm, b_opp_taken_in_obs)
            Q_values_for_dm_intentions = expected_rewards_all_idm + self.gamma * expected_future_V_all_idm
            
            # Update V(obs, b_opp_taken_in_obs) with the max Q-value (DM plays optimally against b_opp_taken_in_obs)
            self.V[obs, b_opp_taken_in_obs] = np.max(Q_values_for_dm_intentions)

            # Update the Dirichlet model for the opponent's observed action
            self.Dir[obs, b_opp_taken_in_obs] += 1         
            
            # Set flag for simulation of outcomes to false for next episode 
            self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = False     

class Level1DPAgent_Dynamic(Agent):
    """
    A value iteration agent that treats the other player as a level 0 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular form, i.e., using a matrix.
    Dynamic in this case means that the agent learns transition distribution durring the game.
    """

    def __init__(self, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.player_id = player_id
        
        # Flag to ensure simulation and expected value calculation runs only once per step
        self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = False
        
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
        
        # Calculate P(edm,eadv | idm,iadv) - the 4D execution probability tensor
        
        # Break down possible combined actions of DM and Adversary into move and push
        if self.player_id == 0:
            self.DM_action_details = self.env_snapshot.combined_actions_blue
            self.Adv_action_details = self.env_snapshot.combined_actions_red
            
            self.DM_available_move_actions_num = len(env.available_move_actions_DM)
            self.Adv_available_move_actions_num = len(env.available_move_actions_Adv)
            
            # Saving number of DM and Adv actions
            self.num_DM_actions = len(self.env_snapshot.combined_actions_blue)
            self.num_Adv_actions = len(self.env_snapshot.combined_actions_red)
            
        else: 
            self.DM_action_details = self.env_snapshot.combined_actions_red
            self.Adv_action_details = self.env_snapshot.combined_actions_blue
            
            self.DM_available_move_actions_num  = len(env.available_move_actions_Adv)
            self.Adv_available_move_actions_num = len(env.available_move_actions_DM)
            
            # Saving number of DM and Adv actions
            self.num_DM_actions = len(self.env_snapshot.combined_actions_red)
            self.num_Adv_actions = len(self.env_snapshot.combined_actions_blue)
            
        # Initialize array for executed action reward (executed action == moving to state s')
        self.DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        # Initialize array for calculation of E[V(s',b')|s']
        self.future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        
        # Initialize array of weights for Dirichlet distribution for transition model estimate
        self.transition_model_weights = np.ones((self.num_DM_actions, self.num_Adv_actions, self.n_states, self.num_DM_actions, self.num_Adv_actions))
        
        self.prob_exec_tensor_for_obs = np.ones((self.num_DM_actions, self.num_Adv_actions, self.num_DM_actions, self.num_Adv_actions))
        
        # Create the lookup table
        self.s_prime_to_exec_actions = self._create_s_prime_lookup_table()
    
    def _create_s_prime_lookup_table(self):
        """
        Pre-computes a lookup table that maps (s, s') -> list of (edm, eadv).
        This tells us which executed actions could lead from state s to s'.
        This is computationally expensive and should only be run once.
        """
        print("Pre-computing the s' to executed actions lookup table. This may take a while...")
        
        # The table will be a dictionary: {s: {s_prime: [(edm, eadv), ...]}}
        lookup_table = {}

        for s in range(self.n_states):
            # Dictionary for this specific starting state 's'
            lookup_table[s] = {}
            
            # Reset the simulation environment to state 's'
            try:
                self.reset_sim_env(s)
            except (IndexError, ValueError):
                # Some state indices might be unreachable/invalid. Skip them.
                continue

            # Simulate every possible executed action pair
            for edm in range(self.num_DM_actions):
                for eadv in range(self.num_Adv_actions):

                    current_env_state = self.env_snapshot.get_state()
                    
                    s_prime, _, _ = self.env_snapshot.step((edm, eadv))

                    # If this s_prime hasn't been seen from this s, create a new list for it
                    if s_prime not in lookup_table[s]:
                        lookup_table[s][s_prime] = []
                    
                    # Add the executed action pair to the list for this outcome
                    lookup_table[s][s_prime].append((edm, eadv))
                    
                    # Restore the environment back to its state 's' for the next iteration.
                    self.reset_sim_env(current_env_state)
        
        print("Lookup table computation finished.")
        return lookup_table

    
    def reset_sim_env(self, obs):
        """
        Resets the simulated environment to the state represented by 'obs'.
        - Player and coin positions and collection status are restored from 'obs'.
        - Step counter is reset (as per self.env_snapshot.reset()).
        """
        # Reset the environment to a default initial state first
        self.env_snapshot.reset()

        # Radix decoding of the state ID 'obs'
        base_pos = self.env_snapshot.N * self.env_snapshot.N
        base_coll = 2  # 0 or 1 for each coin

        # Decode collected coin flags (in reverse order of encoding)
        self.env_snapshot.red_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.red_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll

        # Decode player positions
        p2_flat = obs % base_pos  # Red player's flattened position
        obs //= base_pos
        
        p1_flat = obs  # Blue player's flattened position (or simply obs)

        # Convert flattened positions back to 2D coordinates
        # p_flat = row + N * col
        blue_player_col = p1_flat // self.env_snapshot.N
        blue_player_row = p1_flat % self.env_snapshot.N
        
        # Setting blue player to coordinates extracted from state obs
        self.env_snapshot.blue_player = np.array([blue_player_row, blue_player_col])

        red_player_col = p2_flat // self.env_snapshot.N
        red_player_row = p2_flat % self.env_snapshot.N
        
        # Setting red player to coordinates extracted from state obs
        self.env_snapshot.red_player = np.array([red_player_row, red_player_col])
        
        # Set coin availability based on decoded collection status
        self.env_snapshot.coin1_available = not (self.env_snapshot.blue_collected_coin1 or self.env_snapshot.red_collected_coin1)
        self.env_snapshot.coin2_available = not (self.env_snapshot.blue_collected_coin2 or self.env_snapshot.red_collected_coin2)
        
    def _get_probabilities_for_state(self, obs):
        """
        Calculates P(edm, eadv | s=obs, idm, iadv) for a single given state.
        This is the "just-in-time" version of _calculate_execution_probabilities.
        """
        # Get the weights for the specific state 'obs'
        # This slices the giant 5D array into a much smaller 4D array (idm, iadv, edm, eadv)
        weights_for_obs = self.transition_model_weights[:, :, obs, :, :]
        
        # Sum over the outcomes (edm, eadv) to get the normalization counts for this state
        # The shape of weights_for_obs is (idm, iadv, edm, eadv)
        total_counts = np.sum(weights_for_obs, axis=(2, 3), keepdims=True)
        
        # Normalize to get the 4D probability tensor for this state
        # Use np.divide to prevent division-by-zero errors if a state was never visited
        prob_tensor_for_obs = np.divide(weights_for_obs, total_counts,
                                        out=np.zeros_like(weights_for_obs),
                                        where=total_counts!=0)
                                        
        return prob_tensor_for_obs
    
    def _simulate_executed_outcomes(self, obs):
        """
        Simulates all possible executed action pairs from the given state 'obs'
        and returns the resulting next states and DM rewards.
        """
    
        # Run only if it was not already ran in this episode
        if not self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag:
            ## Determine reachable next states and their immediate rewards
            self.reset_sim_env(obs)
            self.executed_action_outcomes = {}  # (exec_dm_idx, exec_adv_idx) -> (s_prime, r_DM)
            actual_next_states_set = set() # To store unique s_prime values

            # Execute all actions with probab 1 to get reachable states and their rewards
            for DM_exec_idx in range(self.num_DM_actions):
                for Adv_exec_idx in range(self.num_Adv_actions):
                    act_comb_executed = (DM_exec_idx, Adv_exec_idx)
                    s_prime, rewards_vec, _ = self.env_snapshot.step(act_comb_executed)
                    
                    if self.player_id == 0:
                        DM_reward_for_exec = rewards_vec[0]
                    else: 
                        DM_reward_for_exec = rewards_vec[1]
                    
                    self.executed_action_outcomes[act_comb_executed] = (s_prime, DM_reward_for_exec)
                    actual_next_states_set.add(s_prime)
                    
                    self.reset_sim_env(obs) # Reset for the next simulation iteration
            
            self._calculate_expected_future_value(actual_next_states_set)
            
    def _calculate_expected_future_value(self, actual_next_states_set):
        """
        Simulates all possible executed action pairs from the given state 'obs',
        calculates the resulting next states and DM rewards, and computes
        E[V(s',b')|s'] for each reachable s'.
        """
        
        unique_s_primes = np.array(list(actual_next_states_set), dtype=int)

        ## Calculate E[V(s',b')|s'] only for unique_s_primes
        s_prime_to_expected_V = {}

        # Index V and Dir only for the relevant next states
        V_relevant = self.V[unique_s_primes, :]    # Shape: (len(unique_s_primes), num_Adv_actions)
        Dir_relevant = self.Dir[unique_s_primes, :] # Shape: (len(unique_s_primes), num_Adv_actions)

        # Calculate normalization constant only for s which are relevant
        dir_sum_relevant = np.sum(Dir_relevant, axis=1, keepdims=True)
        
        # Normalizing Dirichlet distribution only for relevant s
        beliefs_Adv_action_relevant_s_prime = Dir_relevant / dir_sum_relevant
        
        # Calculating mean value of b's for each s. This part is not dependent on transition model and can be precalculated.
        expected_V_values_for_unique_s_primes = np.sum(V_relevant * beliefs_Adv_action_relevant_s_prime, axis=1)
        
        # Save into dictionary for later lookup
        for i, s_prime_idx in enumerate(unique_s_primes):
            s_prime_to_expected_V[s_prime_idx] = expected_V_values_for_unique_s_primes[i]

        # --- Prepare arrays of R(edm,eadv) and E[V(s'_from_edm,eadv)] ---
        # These are rewards/values for *executed* actions.
        DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions)) 

        for DM_exec_idx in range(self.num_DM_actions):
            for Adv_exec_idx in range(self.num_Adv_actions):
                # Get next state and reward of DM for executed actions
                s_prime, r_DM = self.executed_action_outcomes[(DM_exec_idx, Adv_exec_idx)]
                DM_rewards_executed_array[DM_exec_idx, Adv_exec_idx] = r_DM
                # Get E[V(s')|s'] from our map, default to 0 if s_prime somehow not found 
                future_V_values_executed_array[DM_exec_idx, Adv_exec_idx] = s_prime_to_expected_V.get(s_prime, 0.0)
        
        self.DM_rewards_executed_array = DM_rewards_executed_array
        self.future_V_values_executed_array = future_V_values_executed_array
    
    def optim_act(self, obs): 
            
        # Perform sumproducts using np.einsum with the 4D tensor for the current state
        
        # Calculate expected reward r(s,a,b,s') with respect to p(s'|s,a,b)
        expected_DM_rewards = np.einsum('ijkl,kl->ij', self.prob_exec_tensor_for_obs, self.DM_rewards_executed_array)

        # Calculate the expectations of E[V(s',b')|s'] with respect to p(s'|s,a,b)
        weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor_for_obs, self.future_V_values_executed_array)

        # Final Action Selection
        p_b_given_s_obs = self.Dir[obs] / np.sum(self.Dir[obs])
        total_action_values = np.dot(expected_DM_rewards, p_b_given_s_obs) + self.gamma * np.dot(weighted_sum_future_V, p_b_given_s_obs)
            
        # Find indicies (actions) of all max values
        max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))

        # Choose randomly among them
        chosen_action = np.random.choice(max_indices)
        
        return chosen_action
    
    def act(self, obs, env):
        "Epsilon greedy action selection strategy."
            
        self.prob_exec_tensor_for_obs = self._get_probabilities_for_state(obs)
        
        self._simulate_executed_outcomes(obs)
        
        # Set flag to True
        self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = True
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.optim_act(obs)
        
    def update(self, obs, actions, rewards, new_obs, env):
            # obs: current state where actions were taken
            # actions: tuple [a_dm_actually_taken, b_opp_actually_taken]
            # rewards, new_obs: not directly used for this type of V(s,b_opp) update,
            #                   as it's a model-based VI-like update based on current V and model.

            if self.player_id == 0:
                b_opp_taken_in_obs = actions[1] # The opponent's action we observed in 'obs'
            else: 
                b_opp_taken_in_obs = actions[0] # The opponent's action we observed in 'obs'

            # --- Calculate E[V(s',b_opp')|s'] ---
            # This is the expected value of next states, considering opponent's policy in those next states.
            # This code is largely the same as in act().
            
            # We already calculated this part in the act method which is ran 
            
            self._simulate_executed_outcomes(obs)

            # Calculate Q(obs, a_dm_intended, b_opp_taken_in_obs) for ALL a_dm_intended ---
            # We need to find the best DM response 'a_dm_intended' if opponent plays 'b_opp_taken_in_obs'.
            
            # prob_exec_tensor_for_fixed_b[idm, edm, eadv] = P(edm,eadv | idm, iadv=b_opp_taken_in_obs)
            prob_exec_tensor_for_fixed_b = self.prob_exec_tensor_for_obs[:, b_opp_taken_in_obs, :, :]
            
            # E[Reward | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * R(edm,eadv)
            # Output shape: (num_DM_actions_intended,)
            expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, self.DM_rewards_executed_array)
            
            # E[Future_V | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * E[V(s'_from_edm,eadv)]
            # Output shape: (num_DM_actions_intended,)
            expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, self.future_V_values_executed_array)
            
            # Q_values[idm] = Q(obs, idm, b_opp_taken_in_obs)
            Q_values_for_dm_intentions = expected_rewards_all_idm + self.gamma * expected_future_V_all_idm
            
            # Update V(obs, b_opp_taken_in_obs) with the max Q-value (DM plays optimally against b_opp_taken_in_obs)
            self.V[obs, b_opp_taken_in_obs] = np.max(Q_values_for_dm_intentions)

            # Update the Dirichlet model for the opponent's observed action
            self.Dir[obs, b_opp_taken_in_obs] += 1         
            
            # Set flag for simulation of outcomes to false for next episode 
            self._simulate_executed_outcomes_and_calculate_expected_value_in_s_prime_flag = False    
    
    def update_transition_model(self, intended_actions, obs, new_obs):
        """
        Updates the learned transition model based on an observed outcome.
        """
        idm, iadv = intended_actions
        
        # Look up all possible executed actions that could explain this transition
        # Use .get() to avoid errors if a state is not in the table (e.g., terminal)
        possible_executed_actions = self.s_prime_to_exec_actions.get(obs, {}).get(new_obs)

        if not possible_executed_actions:
            # This can happen if new_obs is a terminal state not reachable
            # in a single step in the simulation, or if obs is invalid.
            # We can't learn anything from this transition.
            return

        # Distribute the credit: increment the count for each possible executed action which lead to s'.
        # This assumes each is equally likely, which is a simplifying assumption.
        for edm, eadv in possible_executed_actions:
            self.transition_model_weights[idm, iadv, obs, edm, eadv] += 1


class Level2DPAgent_Stationary(Agent):
    """
    A value iteration agent that treats the other player as a level 1 agent.
    She learns from other's actions, estimating their value function.
    She represents value function in a tabular form, i.e., using a matrix V.
    Stationary in this case means that the agent has full information about transition distribution and precalculates it befefore running act() and update()
    """

    def __init__(self, action_space, enemy_action_space, n_states, epsilon, gamma, env):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.epsilonA = epsilon
        self.epsilonB = epsilon
        self.gammaA = gamma
        self.gammaB = gamma
        
        self.enemy_action_space = enemy_action_space

        # This is the value function V(s,a)
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        
        # Initializing adversary agent
        self.enemy = Level1DPAgent_Stationary(action_space=self.enemy_action_space,
                   enemy_action_space=self.action_space,
                   n_states=self.n_states,
                   epsilon=self.epsilonB,
                   gamma=self.gammaB,
                   player_id=1,
                   env=env)    
    
        # Initialize an empty environment for simulation of steps 
        #* Warning: Do not copy env to many times -> extreme memory and computational overhead (garbage collection)
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.blue_player_execution_prob = 1
        self.env_snapshot.red_player_execution_prob = 1
        
        # Calculate P(edm,eadv | idm,iadv) - the 4D execution probability tensor
        
        # Break down possible combined actions of DM and Adversary into move and push
        DM_action_details = self.env_snapshot.combined_actions_blue
        Adv_action_details = self.env_snapshot.combined_actions_red
        
        # Saving number of DM and Adv actions
        self.num_DM_actions = len(self.env_snapshot.combined_actions_blue)
        self.num_Adv_actions = len(self.env_snapshot.combined_actions_red)

        DM_moves = DM_action_details[:, 0]
        DM_pushes = DM_action_details[:, 1]
        Adv_moves = Adv_action_details[:, 0]
        Adv_pushes = Adv_action_details[:, 1]
        
        # Initialize matrix for storing probabilities of action executing conditioned by intended action for DM (intended x executed)
        prob_DM_part = np.zeros((self.num_DM_actions, self.num_DM_actions)) # P(edm|idm)
        
        # Creating a boolean matrix of intended x executed moves for probability assignment
        DM_moves_match = (DM_moves[:, np.newaxis] == DM_moves[np.newaxis, :])
        
        # Creating a boolean matrix of intended x executed pushes for probability assignment
        DM_pushes_match = (DM_pushes[:, np.newaxis] == DM_pushes[np.newaxis, :])
        
        # Setting probability for intended move AND push match executed move AND push
        prob_DM_part[DM_moves_match & DM_pushes_match] = env.blue_player_execution_prob

        # Setting probability for Intended push matches executed push, BUT intended move DOES NOT match executed move
        num_alt_DM = len(env.available_move_actions_DM) - 1
        
        # Setting probability of unintended moves
        prob_DM_part[~DM_moves_match & DM_pushes_match] = (1.0 - env.blue_player_execution_prob) / num_alt_DM
        
        # Other cases stay zero

        # Initialize matrix for storing probabilities of action executing conditioned by intended action for Adv (intended x executed)
        prob_Adv_part = np.zeros((self.num_Adv_actions, self.num_Adv_actions))
        
        # Creating a boolean matrix of intended x executed moves for probability assignment
        Adv_moves_match = (Adv_moves[:, np.newaxis] == Adv_moves[np.newaxis, :])
        
        # Creating a boolean matrix of intended x executed pushes for probability assignment
        Adv_pushes_match = (Adv_pushes[:, np.newaxis] == Adv_pushes[np.newaxis, :])
        
        # Setting probability for intended move AND push match executed move AND push
        prob_Adv_part[Adv_moves_match & Adv_pushes_match] = env.red_player_execution_prob

        # Setting probability for Intended push matches executed push, BUT intended move DOES NOT match executed move
        num_alt_Adv = len(env.available_move_actions_Adv) - 1
        
        # Setting probability of unintended moves
        prob_Adv_part[~Adv_moves_match & Adv_pushes_match] = (1.0 - env.red_player_execution_prob) / num_alt_Adv
        
        # Other cases stay zero
        
        # Combine to get 4D prob_exec_tensor: P[idm, iadv, edm, eadv]
        # containing joint probabilities of Adv and DM executing an action conditioned by intended action
        self.prob_exec_tensor = prob_DM_part[:, np.newaxis, :, np.newaxis] * prob_Adv_part[np.newaxis, :, np.newaxis, :]
        
    def reset_sim_env(self, obs):
        """
        Resets the simulated environment to the state represented by 'obs'.
        - Player and coin positions and collection status are restored from 'obs'.
        - Step counter is reset (as per self.env_snapshot.reset()).
        """
        # Reset the environment to a default initial state first
        self.env_snapshot.reset()

        # Radix decoding of the state ID 'obs'
        base_pos = self.env_snapshot.N * self.env_snapshot.N
        base_coll = 2  # 0 or 1 for each coin

        # Decode collected coin flags (in reverse order of encoding)
        self.env_snapshot.red_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.red_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll

        self.env_snapshot.blue_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll

        # Decode player positions
        p2_flat = obs % base_pos  # Red player's flattened position
        obs //= base_pos
        
        p1_flat = obs  # Blue player's flattened position (or simply obs)

        # Convert flattened positions back to 2D coordinates
        # p_flat = row + N * col
        blue_player_col = p1_flat // self.env_snapshot.N
        blue_player_row = p1_flat % self.env_snapshot.N
        
        # Setting blue player to coordinates extracted from state obs
        self.env_snapshot.blue_player = np.array([blue_player_row, blue_player_col])

        red_player_col = p2_flat // self.env_snapshot.N
        red_player_row = p2_flat % self.env_snapshot.N
        
        # Setting red player to coordinates extracted from state obs
        self.env_snapshot.red_player = np.array([red_player_row, red_player_col])
        
        # Set coin availability based on decoded collection status
        self.env_snapshot.coin1_available = not (self.env_snapshot.blue_collected_coin1 or self.env_snapshot.red_collected_coin1)
        self.env_snapshot.coin2_available = not (self.env_snapshot.blue_collected_coin2 or self.env_snapshot.red_collected_coin2)
    
    def optim_act(self, obs):
            ## Determine reachable next states and their immediate rewards
            
            self.reset_sim_env(obs)
            executed_action_outcomes = {}  # (exec_dm_idx, exec_adv_idx) -> (s_prime, r_DM)
            actual_next_states_set = set() # To store unique s_prime values

            # Execute all actions with probab 1 to get reachable states and their rewards
            for DM_exec_idx in range(self.num_DM_actions):
                for Adv_exec_idx in range(self.num_Adv_actions):
                    act_comb_executed = (DM_exec_idx, Adv_exec_idx)
                    s_prime, rewards_vec, _ = self.env_snapshot.step(act_comb_executed)
                    DM_reward_for_exec = rewards_vec[0]
                    
                    executed_action_outcomes[act_comb_executed] = (s_prime, DM_reward_for_exec)
                    actual_next_states_set.add(s_prime)
                    
                    self.reset_sim_env(obs) # Reset for the next simulation iteration
            
            unique_s_primes = np.array(list(actual_next_states_set), dtype=int)

            ## Calculate E[V(s',b')|s'] only for unique_s_primes
            s_prime_to_expected_V = {}

            # Index V and Dir only for the relevant next states
            V_relevant = self.V[unique_s_primes, :]    # Shape: (len(unique_s_primes), num_Adv_actions)
            
            # Initialize an array to store epsilon-greedy policies for each unique_s_prime
            eps_greed_enemy_policies_for_s_primes = np.zeros((len(unique_s_primes), self.num_Adv_actions))
            
            # Simulating epsilon greedy strategies for each reacheble s'
            for i, s_prime_idx in enumerate(unique_s_primes):
                enemy_opt_act = self.enemy.optim_act(s_prime_idx)
                
                prob_non_optimal = self.epsilonB / (self.num_Adv_actions - 1)
                eps_greed_enemy_policies_for_s_primes[i, :] = prob_non_optimal
                eps_greed_enemy_policies_for_s_primes[i, enemy_opt_act] = 1.0 - self.epsilonB
            
            # Calculating mean value of b's for each s. This part is not dependent on transition model and can be precalculated.
            expected_V_values_for_unique_s_primes = np.sum(V_relevant * eps_greed_enemy_policies_for_s_primes, axis=1)
            
            # Save into dictionary for later lookup
            for i, s_prime_idx in enumerate(unique_s_primes):
                s_prime_to_expected_V[s_prime_idx] = expected_V_values_for_unique_s_primes[i]

            # --- Prepare arrays of R(edm,eadv) and E[V(s'_from_edm,eadv)] ---
            # These are rewards/values for *executed* actions.
            DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
            future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions)) 

            for DM_exec_idx in range(self.num_DM_actions):
                for Adv_exec_idx in range(self.num_Adv_actions):
                    # Get next state and reward of DM for executed actions
                    s_prime, r_DM = executed_action_outcomes[(DM_exec_idx, Adv_exec_idx)]
                    DM_rewards_executed_array[DM_exec_idx, Adv_exec_idx] = r_DM
                    # Get E[V(s')|s'] from our map, default to 0 if s_prime somehow not found 
                    future_V_values_executed_array[DM_exec_idx, Adv_exec_idx] = s_prime_to_expected_V.get(s_prime, 0.0)

            # Perform sumproducts using np.einsum
            
            # Calculate expected reward r(s,a,b,s') with respect to p(s'|s,a,b)
            expected_DM_rewards = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, DM_rewards_executed_array)

            # Calculate the expectations of E[V(s',b')|s'] with respect to p(s'|s,a,b)
            weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, future_V_values_executed_array)

            # Final Action Selection
            
            eps_greed_enemy_policy_for_obs = np.zeros(self.num_Adv_actions)
            
            # Calculate estimated adversary epsilon greedy strategy in state obs
            enemy_opt_act = self.enemy.optim_act(obs)
                
            prob_non_optimal = self.epsilonB / (self.num_Adv_actions - 1)
            eps_greed_enemy_policy_for_obs[:] = prob_non_optimal
            eps_greed_enemy_policy_for_obs[enemy_opt_act] = 1.0 - self.epsilonB
                 
            total_action_values = np.dot(expected_DM_rewards, eps_greed_enemy_policy_for_obs) + self.gammaA * np.dot(weighted_sum_future_V, eps_greed_enemy_policy_for_obs)
            
            # Find indicies (actions) of all max values
            max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))

            # Choose randomly among them
            chosen_action = np.random.choice(max_indices)
            
            return chosen_action
    
    def act(self, obs, env):
        
        if np.random.rand() < self.epsilonA:
            return np.random.choice(self.action_space)
        else:
            return self.optim_act(obs)
 

    def update(self, obs, actions, rewards, new_obs, env):
            # obs: current state where actions were taken
            # actions: tuple [a_dm_actually_taken, b_opp_actually_taken]
            # rewards, new_obs: not directly used for this type of V(s,b_opp) update,
            #                   as it's a model-based VI-like update based on current V and model.

            self.enemy.update(obs, actions, rewards, new_obs, env)

            # if self.player_id == 0:
            b_opp_taken_in_obs = actions[1] # The opponent's action we observed in 'obs'
            # else: 
            #     b_opp_taken_in_obs = actions[0] # The opponent's action we observed in 'obs'

            # --- Calculate E[V(s',b_opp')|s'] ---
            # This is the expected value of next states, considering opponent's policy in those next states.
            # This code is largely the same as in act().

            self.reset_sim_env(obs)
            
            executed_action_outcomes = {}  # (exec_dm_idx, exec_adv_idx) -> (s_prime, r_DM)
            actual_next_states_set = set() # To store unique s_prime values

            # Execute all actions with probab 1 to get reachable states and their rewards
            for DM_exec_idx in range(self.num_DM_actions):
                for Adv_exec_idx in range(self.num_Adv_actions):
                    act_comb_executed = (DM_exec_idx, Adv_exec_idx)
                    s_prime, rewards_vec, _ = self.env_snapshot.step(act_comb_executed)
                    DM_reward_for_exec = rewards_vec[0]
                    
                    executed_action_outcomes[act_comb_executed] = (s_prime, DM_reward_for_exec)
                    actual_next_states_set.add(s_prime)
                    
                    self.reset_sim_env(obs) # Reset for the next simulation iteration
            
            unique_s_primes = np.array(list(actual_next_states_set), dtype=int)

            ## Calculate E[V(s',b')|s'] only for unique_s_primes
            s_prime_to_expected_V = {}

            # Index V and Dir only for the relevant next states
            V_relevant = self.V[unique_s_primes, :]    # Shape: (len(unique_s_primes), num_Adv_actions)
            
            # Initialize an array to store epsilon-greedy policies for each unique_s_prime
            eps_greed_enemy_policies_for_s_primes = np.zeros((len(unique_s_primes), self.num_Adv_actions))
            
            # Simulating epsilon greedy strategies for each reacheble s'
            for i, s_prime_idx in enumerate(unique_s_primes):
                enemy_opt_act = self.enemy.optim_act(s_prime_idx)
                
                prob_non_optimal = self.epsilonB / (self.num_Adv_actions - 1)
                eps_greed_enemy_policies_for_s_primes[i, :] = prob_non_optimal
                eps_greed_enemy_policies_for_s_primes[i, enemy_opt_act] = 1.0 - self.epsilonB
            
            # Calculating mean value of b's for each s. This part is not dependent on transition model and can be precalculated.
            expected_V_values_for_unique_s_primes = np.sum(V_relevant * eps_greed_enemy_policies_for_s_primes, axis=1)
            
            # Save into dictionary for later lookup
            for i, s_prime_idx in enumerate(unique_s_primes):
                s_prime_to_expected_V[s_prime_idx] = expected_V_values_for_unique_s_primes[i]

            # --- Prepare arrays of R(edm,eadv) and E[V(s'_from_edm,eadv)] ---
            # These are rewards/values for *executed* actions.
            DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
            future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions)) 

            for DM_exec_idx in range(self.num_DM_actions):
                for Adv_exec_idx in range(self.num_Adv_actions):
                    # Get next state and reward of DM for executed actions
                    s_prime, r_DM = executed_action_outcomes[(DM_exec_idx, Adv_exec_idx)]
                    DM_rewards_executed_array[DM_exec_idx, Adv_exec_idx] = r_DM
                    # Get E[V(s')|s'] from our map, default to 0 if s_prime somehow not found 
                    future_V_values_executed_array[DM_exec_idx, Adv_exec_idx] = s_prime_to_expected_V.get(s_prime, 0.0)

            # Calculate Q(obs, a_dm_intended, b_opp_taken_in_obs) for ALL a_dm_intended ---
            # We need to find the best DM response 'a_dm_intended' if opponent plays 'b_opp_taken_in_obs'.
            
            # prob_exec_tensor_for_fixed_b[idm, edm, eadv] = P(edm,eadv | idm, iadv=b_opp_taken_in_obs)
            prob_exec_tensor_for_fixed_b = self.prob_exec_tensor[:, b_opp_taken_in_obs, :, :]
            
            # E[Reward | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * R(edm,eadv)
            # Output shape: (num_DM_actions_intended,)
            expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, DM_rewards_executed_array)
            
            # E[Future_V | obs, idm, b_opp_taken_in_obs] = sum_{edm,eadv} P(edm,eadv | idm, b_opp_taken_in_obs) * E[V(s'_from_edm,eadv)]
            # Output shape: (num_DM_actions_intended,)
            expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_for_fixed_b, future_V_values_executed_array)
                 
            # Q_values[idm] = Q(obs, idm, b_opp_taken_in_obs)
            Q_values_for_dm_intentions = expected_rewards_all_idm + self.gammaA * expected_future_V_all_idm
            
            # Update V(obs, b_opp_taken_in_obs) with the max Q-value (DM plays optimally against b_opp_taken_in_obs)
            self.V[obs, b_opp_taken_in_obs] = np.max(Q_values_for_dm_intentions)
        

    