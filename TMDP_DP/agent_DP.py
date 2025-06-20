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
        
    def act(self, obs, env):
        
        return choice(self.action_space, p=softmax(self.Q[obs,:],self.beta))
    
### ============ Q-learning agents ============ 

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
    def __init__(self, k, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma):
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

        # Q-function Q(s, a, b), where a is self action, b is opponent action
        self.Q = np.zeros([self.n_states, len(self.action_space), len(self.enemy_action_space)])

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
                gamma=self.gamma
            )

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

        # Update the internal opponent model (recursively if k > 1)
        if self.k > 1:
            # Update the enemy model with its own perspective (actions and rewards swapped)
            self.enemy.update(obs, [a_adv, a_dm], [r_adv, r_dm], new_obs)
        else: # k == 1
            # Update the Dirichlet count for the opponent's observed action
            self.Dir[obs, a_adv] += 1
        
        # Calculate the value of the next state (max Q') for the Bellman update
        opponent_policy_new = self.get_opponent_policy(new_obs)
        expected_q_new = np.dot(self.Q[new_obs], opponent_policy_new)
        max_q_new = np.max(expected_q_new)
        
        # Update our own Q-function
        current_q = self.Q[obs, a_dm, a_adv]
        self.Q[obs, a_dm, a_adv] = (1 - self.alpha) * current_q + self.alpha * (r_dm + self.gamma * max_q_new)


class LevelKQAgentSoftmax(LevelKQAgent):
    """
    A version of the Level-k Q-learning agent that uses a softmax policy
    for action selection instead of epsilon-greedy.
    """
    def __init__(self, k, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma, beta=1.0):
        # Call the parent constructor
        super().__init__(k, action_space, enemy_action_space, n_states, learning_rate, epsilon, gamma)
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
        
        
### ============ Dynamic programming agents - value iteration ============ 

class LevelKDPAgent_Stationary(Agent):
    """
    A template for a level-k value iteration agent.
    - k=1: A level-1 agent that models its opponent as a level-0 (zero-intelligence) agent 
           using a Dirichlet distribution to learn their action probabilities. This is the base case.
    - k>1: A level-k agent that models its opponent as a level-(k-1) agent. It recursively
           builds a model of the opponent to predict their actions.

    This agent assumes a stationary environment with a known transition model.
    """

    def __init__(self, k, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        if k < 1:
            raise ValueError("Level k must be a positive integer.")
            
        Agent.__init__(self, action_space)

        # Core agent parameters
        self.k = k
        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma
        self.player_id = player_id
        self.enemy_action_space = enemy_action_space

        # Value function V(s, b_opp), where b_opp is the opponent's action
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        
        # --- Common Initialization for all levels ---
        
        # Flag to prevent re-simulation within a single step
        self._simulate_flag = False
        
        # Simulation environment for lookaheads
        self.env_snapshot = copy.deepcopy(env)
        
        # Setting probabilities of execution to 1, i.e. no stochasticity.
        # This is for mapping of reachable states to executed actions.
        self.env_snapshot.blue_player_execution_prob = 1
        self.env_snapshot.red_player_execution_prob = 1
        
        # Determine action spaces and details based on player_id
        if self.player_id == 0:
            # Extract radix encoded actions
            self.DM_action_details = self.env_snapshot.combined_actions_blue
            self.Adv_action_details = self.env_snapshot.combined_actions_red
            
            # Seave number of radix encoded actions
            self.num_DM_actions = len(env.combined_actions_blue)
            self.num_Adv_actions = len(env.combined_actions_red)
            
            # Save number of only move actions
            self.DM_available_move_actions_num = len(env.available_move_actions_DM)
            self.Adv_available_move_actions_num = len(env.available_move_actions_Adv)
        else: 
            self.DM_action_details = self.env_snapshot.combined_actions_red
            self.Adv_action_details = self.env_snapshot.combined_actions_blue
            
            self.num_DM_actions = len(env.combined_actions_red)
            self.num_Adv_actions = len(env.combined_actions_blue)
            
            self.DM_available_move_actions_num = len(env.available_move_actions_Adv)
            self.Adv_available_move_actions_num = len(env.available_move_actions_DM)

        ## Initialize arrays for storing simulation results
        
        # Array for reward of DM mapped to executed actions representing reachable state s'
        self.DM_rewards_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))
        # Array for V(s') = E_{p(b_opp|s')}[V(s',b_opp)] mapped to executed actions representing reachable state s'
        self.future_V_values_executed_array = np.zeros((self.num_DM_actions, self.num_Adv_actions))

        # Pre-calculate the state transition probabilities (for agent with full knowledge of transition model)
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)
        
        # --- Level-Specific Initialization ---
        
        # Initialize enemy object
        self.enemy = None
        
        # Initialize Dirichlet distribution representing model of opponents actions p(b|s)
        self.Dir = None
        
        # Initizlize arrays base on agents level. 
        if self.k == 1:
            # BASE CASE: Level-1 agent models opponent with a Dirichlet distribution 
            self.Dir = np.ones((self.n_states, len(self.enemy_action_space)))
        
        elif self.k > 1:
            # RECURSIVE STEP: Level-k agent models opponent as a Level-(k-1) agent
            self.enemy = LevelKDPAgent_Stationary(
                k=self.k - 1, # The recursive call
                action_space=self.enemy_action_space, # Swapping actions space and enemy action space
                enemy_action_space=self.action_space,
                n_states=self.n_states,
                epsilon=self.epsilon, # Using same parameters for the modeled enemy
                gamma=self.gamma,
                player_id=1 - self.player_id, # Opponent has the opposite player_id
                env=env
            )

    def get_opponent_policy(self, obs):
        """
        Abstracts the way the opponent's policy is estimated based on the agent's level k.
        Returns a probability distribution over the opponent's actions in state 'obs'.
        """
        if self.k == 1:
            # Level-1 models opponent policy from Dirichlet counts
            dir_sum = np.sum(self.Dir[obs])
            
            # Robustness check to prevent division by zero
            if dir_sum == 0:
                # Handle case where state has not been visited to avoid division by zero
                return np.ones(self.num_Adv_actions) / self.num_Adv_actions
            return self.Dir[obs] / dir_sum
        else: # k > 1
            # Level-k models opponent as a rational (epsilon-greedy) agent
            enemy_policy = np.zeros(self.num_Adv_actions)
            enemy_opt_act = self.enemy.optim_act(obs)
            
            # Handle the case where there is only one action
            if self.num_Adv_actions > 1:
                # Construct epsilon-greedy strategy
                prob_non_optimal = self.epsilon / (self.num_Adv_actions)
                enemy_policy[:] = prob_non_optimal
                enemy_policy[enemy_opt_act] += 1.0 - self.epsilon
            else:
                enemy_policy[enemy_opt_act] = 1.0

            return enemy_policy
            
    def optim_act(self, obs):
        """Selects the optimal action based on the model."""
        # Expected immediate rewards for each intended action (idm, iadv) pair
        expected_DM_rewards = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, self.DM_rewards_executed_array)
        # Expected future values for each intended action(idm, iadv) pair
        weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, self.future_V_values_executed_array)
        
        # Get opponent's predicted policy for the current state
        opponent_policy_in_obs = self.get_opponent_policy(obs)
        
        # Calculate total value for each of DM actions, marginalized over opponent's policy
        total_action_values = np.dot(expected_DM_rewards, opponent_policy_in_obs) + \
                              self.gamma * np.dot(weighted_sum_future_V, opponent_policy_in_obs)
        
        # Choose best action (breaking ties randomly)
        max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))
        return np.random.choice(max_indices)

    def get_policy(self, obs):
        """
        Calculates this agent's own epsilon-greedy policy distribution.
        This is the same logic used to model the opponent.
        """
        # Get the optimal action deterministically.
        optimal_action_idx = self.optim_act(obs)

        # Construct the epsilon-greedy policy distribution.
        num_actions = len(self.action_space)
        policy = np.zeros(num_actions)

        if num_actions > 1:
            prob_non_optimal = self.epsilon / num_actions
            policy[:] = prob_non_optimal
            policy[optimal_action_idx] += 1.0 - self.epsilon
        else:
            policy[optimal_action_idx] = 1.0

        # Ensure probabilities sum to 1, correcting for potential float precision errors
        return policy / np.sum(policy)

    def act(self, obs, env):
        """
        Epsilon-greedy action selection based on a full policy distribution.
        """
        # Ensure all necessary simulations for the current state are run.
        # This is required by optim_act, which is called by get_policy.
        self._simulate_executed_outcomes(obs)
        self._simulate_flag = True

        # Get the policy distribution for the current state
        policy = self.get_policy(obs)

        # Sample an action from the policy
        return choice(self.action_space, p=policy)

    def update(self, obs, actions, rewards, new_obs):
        """Updates the agent's value function and internal models."""
        # Determine opponent's observed action
        b_opp_taken_in_obs = actions[1] if self.player_id == 0 else actions[0]

        # Update opponent model (recursively for k>1, or Dirichlet for k=1)
        if self.k > 1:
            self.enemy.update(obs, actions, rewards, new_obs)
        else: # k == 1
            self.Dir[obs, b_opp_taken_in_obs] += 1
            
        # Ensure simulations are run for the current state before updating V
        self._simulate_executed_outcomes(obs)

        # --- Update V(obs, b_opp) ---
        
        # Extract transition model entry for specific taken action
        prob_exec_tensor_fixed_b = self.prob_exec_tensor[:, b_opp_taken_in_obs, :, :]
        
        # Calculate expected rewards
        expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, self.DM_rewards_executed_array)
        
        # Calculated expected future values
        expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, self.future_V_values_executed_array)
        
        # Calculate value of optimal action for each action
        Q_values_for_dm_intentions = expected_rewards_all_idm + self.gamma * expected_future_V_all_idm
        self.V[obs, b_opp_taken_in_obs] = np.max(Q_values_for_dm_intentions)
        
        # Reset simulation flag for the next step
        self._simulate_flag = False

    def _simulate_executed_outcomes(self, obs):
        """Simulates all outcomes from 'obs' to get rewards and next states and map them to executed actions."""
        
        # Check if simulation already ran in this step
        if self._simulate_flag:
            return

        # Reset simulation environment to initial state (obs)
        self.reset_sim_env(obs)
        
        # Prepare objects for saving of rewards and state information from reachable states
        executed_action_outcomes = {}
        actual_next_states_set = set()

        # Run through all possible radix encoded actions
        for DM_exec_idx in range(self.num_DM_actions):
            for Adv_exec_idx in range(self.num_Adv_actions):
                act_comb_executed = (DM_exec_idx, Adv_exec_idx)
                
                # Simulate a step with probability 1
                s_prime, rewards_vec, _ = self.env_snapshot.step(act_comb_executed)
                
                # Extract recieved reward based on player_id
                DM_reward_for_exec = rewards_vec[self.player_id]
                
                # Map reached state s' and recieved reward to executed action combination
                executed_action_outcomes[act_comb_executed] = (s_prime, DM_reward_for_exec)
                
                # Save s' to set of reachable states
                actual_next_states_set.add(s_prime)
                
                # Reset simulation before inspecting next set of actions
                self.reset_sim_env(obs)

        # Calculate expected future value only for reachable states (states with non-zero probability of transition)
        self._calculate_expected_future_value(actual_next_states_set, executed_action_outcomes)

    def _calculate_expected_future_value(self, actual_next_states_set, executed_action_outcomes):
        """Calculates E_{p(b_opp|s'))}[V(s', b')] for each possible next state s'."""
        
        # Extract only unique states from reached states
        unique_s_primes = np.array(list(actual_next_states_set), dtype=int)
        # Initialize array for V(s') = E_{p(b_opp|s'))}[V(s', b')]
        s_prime_to_expected_V = {}

        # Loop through all unique reachable states s'
        for s_prime_idx in unique_s_primes:
            # Extract values only for reachable states s'
            V_s_prime = self.V[s_prime_idx, :]
            
            # Fetch estimate of opponents policy p(b|s)
            opponent_policy_in_s_prime = self.get_opponent_policy(s_prime_idx)
            
            # Calculate E_{p(b_opp|s')}[V(s', b')] for specific s'
            s_prime_to_expected_V[s_prime_idx] = np.dot(V_s_prime, opponent_policy_in_s_prime)
        
        # Map the results to executed actions representing reachable state s'
        for DM_exec_idx in range(self.num_DM_actions):
            for Adv_exec_idx in range(self.num_Adv_actions):
                # Extract reached state and reward for the agent based on executed actions
                s_prime, r_DM = executed_action_outcomes[(DM_exec_idx, Adv_exec_idx)]
                # Map the recieved reward to executed actions pair
                self.DM_rewards_executed_array[DM_exec_idx, Adv_exec_idx] = r_DM
                # Map E_{p(b_opp|s')}[V(s', b')] to executed action pair (replacing s' for pair of executed actions)
                self.future_V_values_executed_array[DM_exec_idx, Adv_exec_idx] = s_prime_to_expected_V.get(s_prime, 0.0)

    def reset_sim_env(self, obs):
        
        # Reset the saved environment
        self.env_snapshot.reset()
        
        # Perform radix decoding of state
        base_pos = self.env_snapshot.N * self.env_snapshot.N
        base_coll = 2
        
        self.env_snapshot.red_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll
        self.env_snapshot.red_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll
        self.env_snapshot.blue_collected_coin2 = bool(obs % base_coll)
        obs //= base_coll
        self.env_snapshot.blue_collected_coin1 = bool(obs % base_coll)
        obs //= base_coll
        p2_flat = obs % base_pos
        obs //= base_pos
        p1_flat = obs
        
        # Decode state infromation into [x,y] coordinates
        blue_player_col = p1_flat // self.env_snapshot.N
        blue_player_row = p1_flat % self.env_snapshot.N
        
        # Set the player location based on decoded information
        self.env_snapshot.blue_player = np.array([blue_player_row, blue_player_col])
        
        # Same for the adversary
        red_player_col = p2_flat // self.env_snapshot.N
        red_player_row = p2_flat % self.env_snapshot.N
        
        self.env_snapshot.red_player = np.array([red_player_row, red_player_col])
        
        # Set coin availability based on information about individual player coin collection
        self.env_snapshot.coin1_available = not (self.env_snapshot.blue_collected_coin1 or self.env_snapshot.red_collected_coin1)
        self.env_snapshot.coin2_available = not (self.env_snapshot.blue_collected_coin2 or self.env_snapshot.red_collected_coin2)

    def _calculate_execution_probabilities(self, env):
        # Determine current execution probabilities from the passed environment
        if self.player_id == 0:
            DM_execution_prob = env.blue_player_execution_prob
            Adv_execution_prob = env.red_player_execution_prob
        else:
            DM_execution_prob = env.red_player_execution_prob
            Adv_execution_prob = env.blue_player_execution_prob

        # Extract move and push actions for DM and Adv
        DM_moves = self.DM_action_details[:, 0]
        DM_pushes = self.DM_action_details[:, 1]
        Adv_moves = self.Adv_action_details[:, 0]
        Adv_pushes = self.Adv_action_details[:, 1]
        
        # Initialize array for saving probabilities of reaching reachable state s'
        # represented as combination of executed actions of DM and Adv 
        prob_DM_part = np.zeros((self.num_DM_actions, self.num_DM_actions))
        
        # Calculate logic mask representing match of intended move and executed move action
        # Note: Push is deterministic
        DM_moves_match = (DM_moves[:, np.newaxis] == DM_moves[np.newaxis, :])
        DM_pushes_match = (DM_pushes[:, np.newaxis] == DM_pushes[np.newaxis, :])
        
        # Set the probabilities of intended action executing (intended action == executed action)
        # to execution probability provided from the environment
        prob_DM_part[DM_moves_match & DM_pushes_match] = DM_execution_prob
        
        # Set the probabilities of unintended action executing to uniform 
        num_alt_DM = self.DM_available_move_actions_num - 1
        
        # Robustness check if num of actions is > 0
        if num_alt_DM > 0:
            prob_DM_part[~DM_moves_match & DM_pushes_match] = (1.0 - DM_execution_prob) / num_alt_DM
        
        # Set the same for Adversary
        prob_Adv_part = np.zeros((self.num_Adv_actions, self.num_Adv_actions))
        
        Adv_moves_match = (Adv_moves[:, np.newaxis] == Adv_moves[np.newaxis, :])
        Adv_pushes_match = (Adv_pushes[:, np.newaxis] == Adv_pushes[np.newaxis, :])
        
        prob_Adv_part[Adv_moves_match & Adv_pushes_match] = Adv_execution_prob
        
        num_alt_Adv = self.Adv_available_move_actions_num - 1
        if num_alt_Adv > 0:
            prob_Adv_part[~Adv_moves_match & Adv_pushes_match] = (1.0 - Adv_execution_prob) / num_alt_Adv
        
        # Calculate probability of reaching state s'
        # represented using probability of executing intended actions as product of probabilities of execution of DM and Adv actions
        prob_exec_tensor = prob_DM_part[:, np.newaxis, :, np.newaxis] * prob_Adv_part[np.newaxis, :, np.newaxis, :]
        
        return prob_exec_tensor
    

class LevelKDPAgent_NonStationary(LevelKDPAgent_Stationary):
    """
    A non-stationary version of the level-k DP agent.
    This agent assumes the transition probabilities (player execution probabilities)
    can change over time and recalculates them at each step.

    Inherits from LevelKDPAgent_Stationary and overrides methods to handle
    non-stationarity.
    """

    def __init__(self, k, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        # Call the parent constructor, which will set up most of the attributes.
        # It will also calculate an initial prob_exec_tensor, which we will override in act().
        super().__init__(k, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env)

        # --- Override the Opponent Model for k > 1 ---
        # The parent creates a Stationary opponent. We need to replace it with a NonStationary one.
        if self.k > 1:
            self.enemy = LevelKDPAgent_NonStationary( # Use the NonStationary class recursively
                k=self.k - 1,
                action_space=self.enemy_action_space,
                enemy_action_space=self.action_space,
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
        # Recalculate for the current agent
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)

        # Recursively call for the opponent model if it exists
        if self.k > 1 and self.enemy:
            self.enemy.recalculate_transition_model(env)

    def act(self, obs, env):
        """
        Epsilon-greedy action selection for the non-stationary case.
        First, it updates the transition model based on the current environment state.
        """
        # Recalculate the transition probabilities for this agent and its opponent model
        self.recalculate_transition_model(env)
        
        # Now, call the parent's act method, which will use the updated prob_exec_tensor
        return super().act(obs, env)

class LevelKDPAgent_Dynamic(LevelKDPAgent_Stationary):
    """
    A dynamic version of the level-k DP agent.
    This agent learns the transition model P(s' | s, a, b) online using a
    Dirichlet distribution over possible outcomes.

    Inherits from LevelKDPAgent_Stationary and overrides methods related to
    the transition model.
    """

    def __init__(self, k, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        # Call the parent constructor to set up V, env_snapshot, action spaces, etc.
        # It will also create a base Level-K Stationary opponent model.
        super().__init__(k, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env)

        # --- Dynamic-Specific Overrides and Additions ---

        # The dynamic agent does not have a single, pre-calculated tensor.
        # This will be calculated on-the-fly for each state.
        self.prob_exec_tensor = None 

        # 5D array for Dirichlet counts: P(idm, iadv, s_start, edm, eadv)
        # Where (edm, eadv) represent executed actions, which are placeholder for s'. This is done for numerical tracability of the whole method.
        # We learn a separate transition model for each starting state 's'.
        self.transition_model_weights = np.ones(
            (self.num_DM_actions, self.num_Adv_actions, self.n_states, self.num_DM_actions, self.num_Adv_actions)
        )

        # Pre-compute the lookup table for which executed actions lead from s to s'.
        # This is a one-time, expensive operation.
        self.s_prime_to_exec_actions = self._create_s_prime_lookup_table()

        # Override the opponent model to be dynamic.
        if self.k > 1:
            self.enemy = LevelKDPAgent_Dynamic( 
                k=self.k - 1, # Recursive call to the Dynamic class
                action_space=self.enemy_action_space, # Swapping actions space and enemy action space
                enemy_action_space=self.action_space, 
                n_states=self.n_states,
                epsilon=self.epsilon, # Using same parameters for the modeled enemy
                gamma=self.gamma,
                player_id=1 - self.player_id, # Opponent has the opposite player_id
                env=env
            )

    def get_probabilities_for_state(self, obs):
        """
        Calculates P(edm, eadv | s=obs, idm, iadv) for a single given state
        based on the learned transition_model_weights.
        """
        weights_for_obs = self.transition_model_weights[:, :, obs, :, :]
        total_counts = np.sum(weights_for_obs, axis=(2, 3), keepdims=True) # Keep dimensions the same for broadcasting
        
        # Use np.divide to handle states that have never been visited (total_counts=0)
        prob_tensor_for_obs = np.divide(
            weights_for_obs, total_counts,
            out=np.zeros_like(weights_for_obs),
            where=total_counts != 0
        )
        return prob_tensor_for_obs
    
    def optim_act(self, obs):
        """
        Selects the optimal action for a dynamic agent.
        """
        # Extrat the transition probabilities for the current state (obs)
        self.prob_exec_tensor = self.get_probabilities_for_state(obs)

        return super().optim_act(obs)

    def update(self, obs, actions, rewards, new_obs):
        """
        Updates the value function and the learned transition model.
        """
        # First, update the transition model with the new observation.
        # Note: This must be done BEFORE the value function is updated to get most up to date version of transition model.
        idm, iadv = actions
        
        # Look up all possible executed actions that could explain the transition from obs to new_obs
        possible_executed_actions = self.s_prime_to_exec_actions.get(obs, {}).get(new_obs, [])

        # Update the Dirichlet counts for each possible executed action pair
        for edm, eadv in possible_executed_actions:
            self.transition_model_weights[idm, iadv, obs, edm, eadv] += 1
        
        # After updating the model, get the fresh probability tensor for this state.
        self.prob_exec_tensor = self.get_probabilities_for_state(obs)
        
        # Call the parent's update method to handle the Bellman update for V(s,b)
        # and the recursive update of the opponent model.
        super().update(obs, actions, rewards, new_obs)
    
    def _create_s_prime_lookup_table(self):
        """
        Pre-computes a lookup table that maps (s, s') -> list of (edm, eadv).
        This tells us which executed actions could lead from state s to s'.
        This is computationally expensive and should only be run once.
        """
        print("Pre-computing the s' to executed actions lookup table for agent at level-" + str(self.k) + ". This may take a while...")
        
        lookup_table = {}
        for s in range(self.n_states):
            lookup_table[s] = {}
            # Robustness check for inherited reset_sim_env method.
            # If it happends that the state cannot be reset into for some reason.
            try:
                self.reset_sim_env(s)
            except (IndexError, ValueError):
                continue

            # Loop through every possible EXECUTED action for the Decision Maker (DM)
            for edm in range(self.num_DM_actions):
                
                # Loop through every possible EXECUTED action for the Adversary (Adv)
                for eadv in range(self.num_Adv_actions):
                    
                    # Save the starting state before we modify the environment.
                    current_env_state = self.env_snapshot.get_state()
                    
                    # Simulate one step with this specific action pair.
                    #    The simulation environment has execution probabilities set to 1,
                    #    so the intended action IS the executed action.
                    s_prime, _, _ = self.env_snapshot.step((edm, eadv))

                    # Initialize the list if this is the first time we've reached s_prime from s.
                    if s_prime not in lookup_table[s]:
                        lookup_table[s][s_prime] = []
                    
                    # Add the current action pair to the list of possible causes.
                    #    This is the core of the logic: we are mapping the outcome (s -> s_prime)
                    #    back to a cause (the action pair (edm, eadv)).
                    lookup_table[s][s_prime].append((edm, eadv))
                    
                    # Restore the environment to the original starting state.
                    # This ensures that the next iteration of the inner loop (for the next `eadv`)
                    # starts from the exact same state `s`, not from the `s_prime` we just reached.
                    self.reset_sim_env(current_env_state)
        
        print("Lookup table computation finished.")
        return lookup_table
    