"""
This module implements several agents. An agent is characterized by two methods:
 * act : implements the policy, i.e., it returns agent's decisions to interact in a MDP or Markov Game.
 * update : the learning mechanism of the agent.
"""

import numpy as np
from numpy.random import choice
from itertools import product
import copy
from tqdm.notebook import tqdm

def softmax(x, beta=1.0):
    x = x - np.max(x)  # stability
    e_x = np.exp(beta * x)
    return e_x / np.sum(e_x)

def manhattan_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Points must have the same dimension")
    return sum(abs(a - b) for a, b in zip(p, q))


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
    
    def update_epsilon(self, new_epsilon):
        """Updating the epsilon for the whole hierarchy."""
        
        raise NotImplementedError()

### ============ Simple agents ============

class ManhattanAgent(Agent):
    """
    A simple agent which minimizes Manhattan distance to the closest available coin and utilizes the push action when the players are adjacent.
    In case of ties between distances, the agent chooses its target coin randomly.
    """

    def __init__(self, action_space, coin_location, grid_size, player_id):
        Agent.__init__(self, action_space)
        
        # Save grid size for state decoding
        self.grid_size = grid_size
        
        # Set player id (0 for blue, 1 for red)
        self.player_id = player_id
        
        # Set coin locations
        self.coin_location = coin_location
        
        # Initializing array for temporary savinf of possible radix encoded actions
        self.possible_actions = None
        
    def decode_state(self, obs):
        """Decodes the state ID to get positions and coin availability."""
        
        # Setting bases for radix decoding
        base_pos, base_coll = self.grid_size**2, 2
        
        # Radix decoding of state
        state_copy = obs
        c_r2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll); state_copy //= base_coll
        p2_flat = state_copy % base_pos; state_copy //= base_pos
        p1_flat = state_copy
        
        # Decoding player locations on the grid
        blue_player = np.array([p1_flat % self.grid_size, p1_flat // self.grid_size])
        red_player_pos = np.array([p2_flat % self.grid_size, p2_flat // self.grid_size])
        
        # Setting coin availability based on coin collection indicators from the state
        coin1_available = not (c_b1 or c_r1)
        coin2_available = not (c_b2 or c_r2)

        return blue_player, red_player_pos, coin1_available, coin2_available
    
    def _are_players_adjacent(self, player1_pos, player2_pos):
        # Checks if players are in immediately neighboring cells (8 directions)
        
        # Calculate the absolute difference in row and column coordinates
        row_diff_abs = np.abs(player1_pos[0] - player2_pos[0])
        col_diff_abs = np.abs(player1_pos[1] - player2_pos[1])
        
        # For 8-directional adjacency (including diagonals):
        # - The maximum coordinate difference must be 1.
        # - This also implies they are not on the same cell (where max diff would be 0).
        return np.max([row_diff_abs, col_diff_abs]) == 1
    
    def compute_possible_actions(self, direction_vec, players_are_adjacent):
        """
        Computes a numpy array of possible optimal move actions and a push action based on player adjacency,
        which can be a single action or two actions in case of a diagonal move.
        """
        possible_moves = []
        row_dir, col_dir = direction_vec[0], direction_vec[1]

        # Determine vertical move component
        if row_dir > 0:
            possible_moves.append(2) # Down
        elif row_dir < 0:
            possible_moves.append(0) # Up

        # Determine horizontal move component
        if col_dir > 0:
            possible_moves.append(1) # Right
        elif col_dir < 0:
            possible_moves.append(3) # Left

        # Convert to a numpy array for vectorized operations
        possible_actions = np.array(possible_moves, dtype=int)

        # Apply radix shift for the "push" action if adjacent
        if players_are_adjacent:
            possible_actions += 4
            
        return possible_actions

    def act(self, obs, env=None):
        """
        Decides which action to take.
        1. Finds the closest available coin.
        2. Computes the direction towards it.
        3. Selects a move action to reduce the distance and push the opponent if it is near.
        """
        # Decoding radix encoded state
        blue_player_pos, red_player_pos, coin1_available, coin2_available = self.decode_state(obs)

        # Determine the agent's current position based on player_id
        player_pos = blue_player_pos if self.player_id == 0 else red_player_pos

        # Check if player are adjacent
        players_are_adjacent = self._are_players_adjacent(blue_player_pos, red_player_pos)
        
        # Set coin positions
        coin1_pos = self.coin_location[0]
        coin2_pos = self.coin_location[1]
        
        # Calculate Manhattan distances to each coin, if it's available
        # Set distance do ininity if not
        dist1 = manhattan_distance(player_pos, coin1_pos) if coin1_available else float('inf')
        dist2 = manhattan_distance(player_pos, coin2_pos) if coin2_available else float('inf')

        # Initialize target directional vector
        target_direction = None
        
        # If both coins are gone, just move randomly (e.g., up/down)
        if not coin1_available and not coin2_available:
            return choice(np.array([0, 2]), p=[0.5, 0.5])

        # Decide which coin to target based on distance
        if dist1 < dist2:
            target_direction = coin1_pos - player_pos
        elif dist2 < dist1:
            target_direction = coin2_pos - player_pos
        else: # Distances are equal (or only one coin is left)
            # If both are available and equidistant, choose one randomly
            if coin1_available and coin2_available:
                chosen_coin_pos = coin1_pos if np.random.rand() < 0.5 else coin2_pos
                target_direction = chosen_coin_pos - player_pos
            # Otherwise, target the only available coin
            elif coin1_available:
                target_direction = coin1_pos - player_pos
            else: # only coin2 is available
                target_direction = coin2_pos - player_pos

        # Compute possible actions: the move action (0-3) = ("Up", "Right", "Down", "Left"). This index works for the combined action
        # because the first 4 actions in the environment are the 'no push' moves.
        self.possible_actions = self.compute_possible_actions(target_direction, players_are_adjacent)
        
        # Choose one of the optimal actions uniformly at random.
        # In case there is only one action, this just chooses the single action
        return np.random.choice(self.possible_actions)
        
    
class ManhattanAgent_Passive(ManhattanAgent):
    """
    A Manhattan agent that never uses the push action.
    """
    def compute_possible_actions(self, direction_vec, players_are_adjacent):
        # Call the parent's method but always with players_are_adjacent=False
        # This ensures the radix_shift is never applied.
        return super().compute_possible_actions(direction_vec, False)
    
class ManhattanAgent_Aggressive(ManhattanAgent):
    """
    A Manhattan agent that always uses the push action and targets opponent if it is closer to player than a coin.
    """
    
    def act(self, obs, env=None):
        """
        Decides which action to take.
        1. Finds the closest available coin or opponent
        2. Computes the direction towards it.
        3. Selects a move action to reduce the distance and push the opponent if it is near.
        """
        # Decoding radix encoded state
        blue_player_pos, red_player_pos, coin1_available, coin2_available = self.decode_state(obs)

        # Determine the agent's and opponent's current position based on player_id
        player_pos = blue_player_pos if self.player_id == 0 else red_player_pos
        opponent_pos = red_player_pos if self.player_id == 0 else blue_player_pos
        
        # Check if player are adjacent
        players_are_adjacent = self._are_players_adjacent(blue_player_pos, red_player_pos)
        
        # Set coin positions
        coin1_pos = self.coin_location[0]
        coin2_pos = self.coin_location[1]
        
        # Location of target objects array 
        target_loc = np.array([coin1_pos, coin2_pos, opponent_pos])
        
        # Calculate Manhattan distances to each coin, if it's available
        # Set distance do ininity if not.
        dist1 = manhattan_distance(player_pos, coin1_pos) if coin1_available else float('inf')
        dist2 = manhattan_distance(player_pos, coin2_pos) if coin2_available else float('inf')
        dist3 = manhattan_distance(player_pos, opponent_pos)

        # Array of distances 
        dist_array = np.array([dist1, dist2, dist3])

        # Detect indicies of equal distances to targets
        min_dist_idx = np.flatnonzero(dist_array == np.min(dist_array))

        # In case of tie with targeting the opponen choose to target the closest coin
        if 2 in min_dist_idx and any(i in min_dist_idx for i in (0, 1, 2, 3)):
            min_dist_idx = np.argmin(dist_array[0:2]) # Re-evaluate distance between coins
        else:
            # Break ties randomly
            min_dist_idx = choice(min_dist_idx)
        
        # Initialize target directional vector
        target_direction = target_loc[min_dist_idx] - player_pos
    
        # Compute possible actions: the move action (0-3) = ("Up", "Right", "Down", "Left"). This index works for the combined action
        # because the first 4 actions in the environment are the 'no push' moves.
        self.possible_actions = self.compute_possible_actions(target_direction, players_are_adjacent)
        
        # Choose one of the optimal actions uniformly at random.
        # In case there is only one action, this just chooses the single action
        return np.random.choice(self.possible_actions)
        
    
class ManhattanAgent_Ultra_Aggressive(ManhattanAgent):
    """
    An aggressive Manhattan agent that identifies the most 'urgent' target on the board.

    It calculates five key distances: player-to-coin1, player-to-coin2, 
    opponent-to-coin1, opponent-to-coin2, and player-to-opponent. It then
    targets the subject of whichever distance is the absolute minimum, allowing it
    to dynamically switch between scoring, attacking, and intercepting. It will
    always use the push action if adjacent to the opponent.
    """
    
    def act(self, obs, env=None):
        """
        Decides which action to take.
        1. Finds the closest available coin or opponent.
        2. Computes the direction towards it.
        3. Selects a move action to reduce the distance and push the opponent if it is near.
        """
        # Decoding radix encoded state
        blue_player_pos, red_player_pos, coin1_available, coin2_available = self.decode_state(obs)

        # Determine the agent's and opponent's current position based on player_id
        player_pos = blue_player_pos if self.player_id == 0 else red_player_pos
        opponent_pos = red_player_pos if self.player_id == 0 else blue_player_pos
        
        # Check if player are adjacent
        players_are_adjacent = self._are_players_adjacent(blue_player_pos, red_player_pos)
        
        # Set coin positions
        coin1_pos = self.coin_location[0]
        coin2_pos = self.coin_location[1]
        
        # Location of target objects array 
        target_loc = np.array([coin1_pos, coin2_pos, coin1_pos, coin2_pos, opponent_pos])
        
        # Calculate Manhattan distances to each coin, if it's available
        # Set distance do ininity if not.
        dist1 = manhattan_distance(player_pos, coin1_pos) if coin1_available else float('inf')
        dist2 = manhattan_distance(player_pos, coin2_pos) if coin2_available else float('inf')
        dist3 = manhattan_distance(opponent_pos, coin1_pos) if coin1_available else float('inf')
        dist4 = manhattan_distance(opponent_pos, coin2_pos) if coin2_available else float('inf')
        dist5 = manhattan_distance(player_pos, opponent_pos)
        
        # Array of distances 
        dist_array = np.array([dist1, dist2, dist3, dist4, dist5])

        # Detect indicies of equal distances to targets
        min_dist_idx = np.flatnonzero(dist_array == np.min(dist_array))

        # In case of tie with targeting the opponen choose to target the closest coin
        if 4 in min_dist_idx and any(i in min_dist_idx for i in (0, 1, 2, 3)):
            min_dist_idx = np.argmin(dist_array[0:4]) # Re-evaluate distance between coins
        else:
            # Break ties randomly
            min_dist_idx = choice(min_dist_idx)
        
        # Initialize target directional vector
        target_direction = target_loc[min_dist_idx] - player_pos
    
        # Compute possible actions: the move action (0-3) = ("Up", "Right", "Down", "Left"). This index works for the combined action
        # because the first 4 actions in the environment are the 'no push' moves.
        self.possible_actions = self.compute_possible_actions(target_direction, players_are_adjacent)
        
        # Choose one of the optimal actions uniformly at random.
        # In case there is only one action, this just chooses the single action
        return np.random.choice(self.possible_actions)
        
    

### ============ Q-learning agents ============ 


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
        
    
    def update_epsilon(self, new_epsilon):
        """Updating the epsilon for the whole hierarchy."""
        
        self.epsilon = new_epsilon
        
class IndQLearningAgentSoftmax(IndQLearningAgent):
    """ A vanilla Q-learning agent that applies softmax policy."""
    
    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space=None, beta=1):
        IndQLearningAgent.__init__(self, action_space, n_states, learning_rate, epsilon, gamma, enemy_action_space)
        
        self.beta = beta
        
    def act(self, obs, env):
        
        return choice(self.action_space, p=softmax(self.Q[obs,:],self.beta))

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
        
        
### ============ Dynamic programming agents - online value iteration ============

class LevelKDPAgent_Stationary(Agent):
    """
    - This version pre-computes the transition and reward models at initialization
      to avoid costly simulations during the learning loop.
    - k=1: Models its opponent as a level-0 (zero-intelligence) agent
           using a Dirichlet distribution to learn their action probabilities.
    - k>1: Models its opponent as a level-(k-1) agent recursively.
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
        
        # Simulation environment for one-time model computation
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.blue_player_execution_prob = 1.0
        self.env_snapshot.red_player_execution_prob = 1.0
        
        # Determine action spaces and details based on player_id
        if self.player_id == 0:
            self.DM_action_details = self.env_snapshot.combined_actions_blue
            self.Adv_action_details = self.env_snapshot.combined_actions_red
            self.num_DM_actions = len(env.combined_actions_blue)
            self.num_Adv_actions = len(env.combined_actions_red)
            self.DM_available_move_actions_num = len(env.available_move_actions_DM)
            self.Adv_available_move_actions_num = len(env.available_move_actions_Adv)
        else: 
            self.DM_action_details = self.env_snapshot.combined_actions_red
            self.Adv_action_details = self.env_snapshot.combined_actions_blue
            self.num_DM_actions = len(env.combined_actions_red)
            self.num_Adv_actions = len(env.combined_actions_blue)
            self.DM_available_move_actions_num = len(env.available_move_actions_Adv)
            self.Adv_available_move_actions_num = len(env.available_move_actions_DM)

        # Pre-calculate the state-independent execution probability tensor P(edm, eadv | idm, iadv)
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)

        # Pre-compute state and reward lookup tables
        # These tensors store the outcomes for every state and EXECUTED action pair.
        self.S_prime_lookup, self.R_lookup = self._precompute_lookups()
        
        # --- Level-Specific Initialization ---
        self.enemy = None
        self.Dir = None
        
        if self.k == 1:
            # BASE CASE: Level-1 models opponent with a Dirichlet distribution
            self.Dir = np.ones((self.n_states, len(self.enemy_action_space)))
        elif self.k > 1:
            # RECURSIVE STEP: Level-k models opponent as a Level-(k-1) agent
            self.enemy = LevelKDPAgent_Stationary(
                k=self.k - 1,
                action_space=self.enemy_action_space,
                enemy_action_space=self.action_space,
                n_states=self.n_states,
                epsilon=self.epsilon,
                gamma=self.gamma,
                player_id=1 - self.player_id,
                env=env
            )

    def _precompute_lookups(self):
        """
        One-time, expensive computation to build lookup table that maps
        (s, edm, eadv) -> (s', r) for all states and executed actions,
        eliminating the need for simulation in the main training loop.
        """
        if self.player_id == 0:
            decs_str = f"Pre-computing state and reward lookup tables for Level-{self.k} DP Agent (DM)... (this may take a moment)"
        else:
            decs_str = f"Pre-computing state and reward lookup tables for Level-{self.k} DP Agent (Adv)... (this may take a moment)"
        
        s_prime_lookup = np.zeros((self.n_states, self.num_DM_actions, self.num_Adv_actions), dtype=int)
        r_lookup = np.zeros((self.n_states, self.num_DM_actions, self.num_Adv_actions))

        for s in tqdm(range(self.n_states), desc=decs_str):
            try:
                self.reset_sim_env(s)
            except (IndexError, ValueError):
                continue # Skip unreachable/invalid states

            for edm in range(self.num_DM_actions):
                for eadv in range(self.num_Adv_actions):
                    # Save state before stepping
                    current_env_state = self.env_snapshot.get_state()
                    
                    # Create the action pair in the correct [blue, red] order for the engine
                    if self.player_id == 0: # I am the Blue Player (DM)
                        action_pair = (edm, eadv)
                    else: # I am the Red Player (Adv)
                        action_pair = (eadv, edm)

                    # Simulate one step with the correctly ordered action pair
                    s_prime, rewards_vec, _ = self.env_snapshot.step(action_pair)
                    
                    # Store the resulting state and reward
                    s_prime_lookup[s, edm, eadv] = s_prime
                    r_lookup[s, edm, eadv] = rewards_vec[self.player_id]
                    
                    # Restore environment to the original state for the next action pair
                    self.reset_sim_env(current_env_state)
        
        print("Lookup table pre-computation finished.")
        return s_prime_lookup, r_lookup

    def get_opponent_policy(self, obs):
        """
        Estimates the opponent's policy (probability distribution over actions) for a given state.
        """
        if self.k == 1:
            # Level-1 models opponent policy from Dirichlet counts
            dir_sum = np.sum(self.Dir[obs])
            if dir_sum == 0:
                # Return a uniform policy if the state has not been seen
                return np.ones(self.num_Adv_actions) / self.num_Adv_actions
            return self.Dir[obs] / dir_sum
        else: # k > 1
            # Level-k recursively asks its internal Level-(k-1) model for its policy
            enemy_policy = np.zeros(self.num_Adv_actions)
            enemy_opt_act = self.enemy.optim_act(obs)
            
            if self.num_Adv_actions > 1:
                # Construct an epsilon-greedy policy for the opponent
                prob_non_optimal = self.epsilon / self.num_Adv_actions
                enemy_policy[:] = prob_non_optimal
                enemy_policy[enemy_opt_act] += 1.0 - self.epsilon
            else:
                enemy_policy[enemy_opt_act] = 1.0
            return enemy_policy

    def _calculate_expected_future_values(self, s_prime_array):
        """
        Efficiently calculates E_{p(b'|s')}[V(s', b')] for an array of next states.
        It avoids re-computation by operating only on unique states.
        """
        # Find unique next states to avoid redundant calculations
        unique_s_primes, inverse_indices = np.unique(s_prime_array, return_inverse=True)
        
        # Calculate the expected value for each unique next state
        expected_V_map = {s_prime: np.dot(self.V[s_prime, :], self.get_opponent_policy(s_prime)) for s_prime in unique_s_primes}
        
        # Map the computed values back to the original array shape using the inverse indices
        expected_V_flat = np.array([expected_V_map.get(s, 0) for s in unique_s_primes])
        return expected_V_flat[inverse_indices].reshape(s_prime_array.shape)

    def optim_act(self, obs):
        """Selects the optimal action based on the pre-computed model using vectorized operations."""
        # Get outcomes for the current state from the pre-computed lookup tables
        R_exec_obs = self.R_lookup[obs, :, :]
        S_prime_exec_obs = self.S_prime_lookup[obs, :, :]
        
        # Calculate expected future values for all possible executed outcomes
        future_V_values_executed = self._calculate_expected_future_values(S_prime_exec_obs)

        # Expected immediate rewards for each INTENDED action (idm, iadv) pair, averaged over execution stochasticity
        expected_DM_rewards = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, R_exec_obs)
        
        # Expected future values for each INTENDED action (idm, iadv) pair
        weighted_sum_future_V = np.einsum('ijkl,kl->ij', self.prob_exec_tensor, future_V_values_executed)
        
        # Get opponent's predicted policy for the current state
        opponent_policy_in_obs = self.get_opponent_policy(obs)
        
        # Calculate total value for each of our actions, marginalized over the opponent's policy
        total_action_values = np.dot(expected_DM_rewards, opponent_policy_in_obs) + \
                              self.gamma * np.dot(weighted_sum_future_V, opponent_policy_in_obs)
        
        # Choose the best action, breaking ties randomly
        max_indices = np.flatnonzero(total_action_values == np.max(total_action_values))
        return np.random.choice(max_indices)

    def get_policy(self, obs):
        """Calculates this agent's own epsilon-greedy policy distribution."""
        optimal_action_idx = self.optim_act(obs)
        num_actions = len(self.action_space)
        policy = np.full(num_actions, self.epsilon / num_actions)
        policy[optimal_action_idx] += 1.0 - self.epsilon
        return policy / np.sum(policy)

    def act(self, obs, env):
        """Selects an action based on the epsilon-greedy policy."""
        policy = self.get_policy(obs)
        return choice(self.action_space, p=policy)

    def update(self, obs, actions, rewards, new_obs):
        """Updates the agent's value function V(s,b) and its internal opponent model."""
        # Determine opponent's observed action
        b_opp_taken_in_obs = actions[1] if self.player_id == 0 else actions[0]

        # Update opponent model (recursively for k>1, or Dirichlet for k=1)
        if self.k > 1:
            self.enemy.update(obs, actions, rewards, new_obs)
        else:
            self.Dir[obs, b_opp_taken_in_obs] += 1
            
        # --- Vectorized V(obs, b_opp) Update ---
        
        # Get outcomes for the current state from the lookup tables
        R_exec_obs = self.R_lookup[obs, :, :]
        S_prime_exec_obs = self.S_prime_lookup[obs, :, :]
        
        # Calculate expected future values for all outcomes
        future_V_values_executed = self._calculate_expected_future_values(S_prime_exec_obs)
        
        # Extract transition probabilities conditioned on the opponent's TAKEN action
        prob_exec_tensor_fixed_b = self.prob_exec_tensor[:, b_opp_taken_in_obs, :, :]
        
        # Calculate expected rewards and future values for each of our INTENDED actions
        expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, R_exec_obs)
        expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, future_V_values_executed)
        
        # Standard Bellman update for the value function
        Q_values_for_dm_intentions = expected_rewards_all_idm + self.gamma * expected_future_V_all_idm
        self.V[obs, b_opp_taken_in_obs] = np.max(Q_values_for_dm_intentions)

    def reset_sim_env(self, obs):
        """Resets the simulation environment to a specific state observation for model pre-computation."""
        
        self.env_snapshot.reset()
        base_pos, base_coll = self.env_snapshot.N**2, 2
        state_copy = obs
        c_r2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll); state_copy //= base_coll
        p2_flat = state_copy % base_pos; state_copy //= base_pos
        p1_flat = state_copy
        self.env_snapshot.blue_player = np.array([p1_flat % self.env_snapshot.N, p1_flat // self.env_snapshot.N])
        self.env_snapshot.red_player = np.array([p2_flat % self.env_snapshot.N, p2_flat // self.env_snapshot.N])
        self.env_snapshot.blue_collected_coin1, self.env_snapshot.blue_collected_coin2 = c_b1, c_b2
        self.env_snapshot.red_collected_coin1, self.env_snapshot.red_collected_coin2 = c_r1, c_r2
        self.env_snapshot.coin1_available = not (c_b1 or c_r1)
        self.env_snapshot.coin2_available = not (c_b2 or c_r2)
    
    def _calculate_execution_probabilities(self, env):
        """Calculates the state-independent transition tensor P(executed | intended)."""
        
        if self.player_id == 0:
            DM_exec_prob, Adv_exec_prob = env.blue_player_execution_prob, env.red_player_execution_prob
        else:
            DM_exec_prob, Adv_exec_prob = env.red_player_execution_prob, env.blue_player_execution_prob
        
        DM_moves, DM_pushes = self.DM_action_details[:, 0], self.DM_action_details[:, 1]
        Adv_moves = self.Adv_action_details[:, 0]
        Adv_pushes = self.Adv_action_details[:, 1]

        prob_DM = np.zeros((self.num_DM_actions, self.num_DM_actions))
        m_match = (DM_moves[:, None] == DM_moves[None, :]); p_match = (DM_pushes[:, None] == DM_pushes[None, :])
        prob_DM[m_match & p_match] = DM_exec_prob
        if self.DM_available_move_actions_num > 1: prob_DM[~m_match & p_match] = (1.0 - DM_exec_prob) / (self.DM_available_move_actions_num - 1)
        
        prob_Adv = np.zeros((self.num_Adv_actions, self.num_Adv_actions))
        m_match = (Adv_moves[:, None] == Adv_moves[None, :]); p_match = (Adv_pushes[:, None] == Adv_pushes[None, :])
        prob_Adv[m_match & p_match] = Adv_exec_prob
        if self.Adv_available_move_actions_num > 1: prob_Adv[~m_match & p_match] = (1.0 - Adv_exec_prob) / (self.Adv_available_move_actions_num - 1)

        return prob_DM[:, np.newaxis, :, np.newaxis] * prob_Adv[np.newaxis, :, np.newaxis, :]

    def update_epsilon(self, new_epsilon):
        """Updating the epsilon for the whole hierarchy."""
        
        self.epsilon = new_epsilon
        if self.k > 1 and self.enemy:
            self.enemy.update_epsilon(new_epsilon)

class LevelKDPAgent_NonStationary(LevelKDPAgent_Stationary):
    """
    Inherits the pre-computation but recalculates execution probabilities each step.
    """
    def __init__(self, k, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        # Avoid calling the parent's __init__ to prevent creating a redundant stationary opponent.
        Agent.__init__(self, action_space)
        
        # Manually copy necessary parameter setup from the parent
        self.k, self.n_states, self.epsilon, self.gamma, self.player_id = k, n_states, epsilon, gamma, player_id
        self.enemy_action_space = enemy_action_space
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.blue_player_execution_prob = 1.0
        self.env_snapshot.red_player_execution_prob = 1.0
        if self.player_id == 0:
            self.DM_action_details = self.env_snapshot.combined_actions_blue; self.Adv_action_details = self.env_snapshot.combined_actions_red
            self.num_DM_actions = len(env.combined_actions_blue); self.num_Adv_actions = len(env.combined_actions_red)
            self.DM_available_move_actions_num = len(env.available_move_actions_DM); self.Adv_available_move_actions_num = len(env.available_move_actions_Adv)
        else: 
            self.DM_action_details = self.env_snapshot.combined_actions_red; self.Adv_action_details = self.env_snapshot.combined_actions_blue
            self.num_DM_actions = len(env.combined_actions_red); self.num_Adv_actions = len(env.combined_actions_blue)
            self.DM_available_move_actions_num = len(env.available_move_actions_Adv); self.Adv_available_move_actions_num = len(env.available_move_actions_DM)

        # Still perform the one-time model pre-computation
        self.S_prime_lookup, self.R_lookup = self._precompute_lookups()
        self.prob_exec_tensor = self._calculate_execution_probabilities(env) # Initial calculation

        # Now, create the correct (NonStationary) opponent model from the start
        self.enemy = None
        self.Dir = None
        if self.k == 1:
            self.Dir = np.ones((self.n_states, len(self.enemy_action_space)))
        elif self.k > 1:
            self.enemy = LevelKDPAgent_NonStationary(
                k=self.k - 1,
                action_space=self.enemy_action_space, enemy_action_space=self.action_space,
                n_states=self.n_states, epsilon=self.epsilon, gamma=self.gamma,
                player_id=1 - self.player_id, env=env
            )

    def recalculate_transition_model(self, env):
        """
        Recursively recalculates the transition probability tensor for this agent
        and all agents in its opponent model hierarchy.
        """
        self.prob_exec_tensor = self._calculate_execution_probabilities(env)
        if self.k > 1 and self.enemy:
            self.enemy.recalculate_transition_model(env)

    def act(self, obs, env):
        """
        Action selection for the non-stationary case.
        First, it updates the transition model based on the current environment state.
        """
        self.recalculate_transition_model(env)
        return super().act(obs, env)


class LevelKDPAgent_Dynamic(LevelKDPAgent_Stationary):
    """
    Learns the transition probabilities P(executed | intended) online.
    """
    def __init__(self, k, action_space, enemy_action_space, n_states, epsilon, gamma, player_id, env):
        # Avoid calling the parent's __init__ to prevent redundant Stationary enemy initialization.
        Agent.__init__(self, action_space)
        
        # Manually copy necessary parameter setup
        self.k, self.n_states, self.epsilon, self.gamma, self.player_id = k, n_states, epsilon, gamma, player_id
        self.enemy_action_space = enemy_action_space
        self.V = np.zeros([self.n_states, len(self.enemy_action_space)])
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.blue_player_execution_prob = 1.0; self.env_snapshot.red_player_execution_prob = 1.0
        if self.player_id == 0:
            self.DM_action_details = self.env_snapshot.combined_actions_blue; self.Adv_action_details = self.env_snapshot.combined_actions_red
            self.num_DM_actions = len(env.combined_actions_blue); self.num_Adv_actions = len(env.combined_actions_red)
        else: 
            self.DM_action_details = self.env_snapshot.combined_actions_red; self.Adv_action_details = self.env_snapshot.combined_actions_blue
            self.num_DM_actions = len(env.combined_actions_red); self.num_Adv_actions = len(env.combined_actions_blue)

        # Still perform the one-time (s, edm, eadv) -> (s', r) model pre-computation
        self.S_prime_lookup, self.R_lookup = self._precompute_lookups()
        
        # --- Dynamic-Specific Initialization ---
        self.prob_exec_tensor = None # This agent learns the transition model, so no fixed tensor
        self.transition_model_weights = np.ones(
            (self.n_states, self.num_DM_actions, self.num_Adv_actions, self.num_DM_actions, self.num_Adv_actions)
        )

        # Create the correct (Dynamic) opponent model from the start
        self.enemy = None
        self.Dir = None
        if self.k == 1:
            self.Dir = np.ones((self.n_states, len(self.enemy_action_space)))
        elif self.k > 1:
            self.enemy = LevelKDPAgent_Dynamic(
                k=self.k - 1,
                action_space=self.enemy_action_space, enemy_action_space=self.action_space, 
                n_states=self.n_states, epsilon=self.epsilon, gamma=self.gamma,
                player_id=1 - self.player_id, env=env
            )

    def get_probabilities_for_state(self, obs):
        """
        Calculates P(edm, eadv | s=obs, idm, iadv) from learned Dirichlet counts for the current state.
        """
        weights_for_obs = self.transition_model_weights[obs, :, :, :, :]
        total_counts = np.sum(weights_for_obs, axis=(2, 3), keepdims=True)
        
        prob_tensor_for_obs = np.divide(
            weights_for_obs, total_counts,
            out=np.zeros_like(weights_for_obs),
            where=total_counts != 0
        )
        return prob_tensor_for_obs
    
    def optim_act(self, obs):
        """Selects optimal action using the dynamically learned transition model."""
        # Calculate transition probabilities for the current state on-the-fly
        self.prob_exec_tensor = self.get_probabilities_for_state(obs)
        
        # Call the parent's (Stationary) efficient optim_act method
        return super().optim_act(obs)

    def update(self, obs, actions, rewards, new_obs):
        """Updates the value function and the learned transition model."""
        idm, iadv = actions
        
        # Find which executed actions could have caused the transition s -> new_obs
        # This is fast because S_prime_lookup is pre-computed.
        possible_exec_actions_mask = (self.S_prime_lookup[obs] == new_obs)
        edm_indices, eadv_indices = np.where(possible_exec_actions_mask)
        
        # Update the Dirichlet counts for all plausible executed actions
        for edm, eadv in zip(edm_indices, eadv_indices):
            self.transition_model_weights[obs, idm, iadv, edm, eadv] += 1
        
        # Get the fresh probability tensor for this state before updating the value function
        self.prob_exec_tensor = self.get_probabilities_for_state(obs)
        
        # Call the parent's update method to handle the Bellman update and opponent model
        super().update(obs, actions, rewards, new_obs)
        

### ============ Dynamic programming agents - offline value iteration (perfect opponent model) ============

class DPAgent_PerfectModel(LevelKDPAgent_Stationary):
    """
    Computes the optimal policy (a best response) against a known, fixed opponent
    policy using offline value iteration.

    This agent inherits the efficient, vectorized model-handling and Bellman update
    logic from LevelKDPAgent_Stationary and overrides the opponent model to be
    a "perfect" (i.e., known and fixed) model of a ManhattanAgent.
    """
    def __init__(self, action_space, enemy_action_space, n_states, gamma, player_id, env, coin_location, grid_size):
        # Initialize as a Level-1 DP Agent to leverage its pre-computation methods.
        # We pass epsilon=0 because the final policy will be deterministic.
        super().__init__(k=1, action_space=action_space, enemy_action_space=enemy_action_space,
                         n_states=n_states, epsilon=0, gamma=gamma, player_id=player_id, env=env)

        # Define the fixed opponent we are solving against.
        self.enemy_agent = ManhattanAgent_Ultra_Aggressive(action_space=self.enemy_action_space,
                                          coin_location=coin_location,
                                          grid_size=grid_size,
                                          player_id=1 - self.player_id)

        # 3. Pre-compute the opponent's full, fixed policy table.
        print("Pre-computing the perfect opponent policy table...")
        self.opponent_policy_table = self._precompute_opponent_policy()
        print("Opponent policy table finished.")

        # 4. The final optimal policy π*(s) will be stored here.
        self.optim_policy_table = np.zeros(self.n_states, dtype=int)

        # 5. Run offline value iteration to find the optimal value function V*(s, b)
        # and then extract the final policy.
        self.run_value_iteration()
        self.extract_optimal_policy()

    def _precompute_opponent_policy(self):
        """Computes the opponent's action probability distribution for every state."""
        policy_table = np.zeros((self.n_states, self.num_Adv_actions))
        for s in tqdm(range(self.n_states), desc="Pre-computing opponent policy table"):
            try:
                # The act method of ManhattanAgent sets its `possible_actions` attribute.
                self.enemy_agent.act(s, None)
                possible_actions = self.enemy_agent.possible_actions

                if possible_actions is not None and len(possible_actions) > 0:
                    policy_table[s, possible_actions] = 1.0 / len(possible_actions)
                else: # Handle cases where no specific action is determined (e.g., no coins left)
                    policy_table[s, :] = 1.0 / self.num_Adv_actions # Assume uniform random
            except (ValueError, IndexError): # Catch errors from invalid/unreachable states
                policy_table[s, :] = 1.0 / self.num_Adv_actions
        return policy_table

    def get_opponent_policy(self, obs):
        """
        Overrides the parent method. Instead of learning, it returns the
        pre-computed policy for the fixed opponent.
        """
        return self.opponent_policy_table[obs]

    def run_value_iteration(self, theta=1e-2, max_iters=10000):
        """
        Performs offline, in-place (asynchronous) value iteration on the V(s, b) table
        until convergence. This is more memory-efficient as it avoids copying the
        large V matrix in each iteration.
        """
        print("Starting offline in-place value iteration on V(s, b)...")

        # Manually create a tqdm instance for fine-grained control in first
        # We set the total to be the convergence threshold 'theta'.
        # Updating the bar by how much we have *not* yet converged.
        with tqdm(total=1.0, desc="Convergence Progress", unit='', 
                bar_format='{l_bar}{bar:10}{r_bar}') as progress_bar:
        

            for i in range(max_iters):
                delta = 0  # Initialize max change for this sweep to zero

                # Loop through all states to update their values
                for s in range(self.n_states):
                    
                    # Get outcomes for the current state from the lookup tables
                    R_exec_obs = self.R_lookup[s, :, :]
                    S_prime_exec_obs = self.S_prime_lookup[s, :, :]
                    
                    # Calculate expected future values for all outcomes
                    future_V_values_executed = self._calculate_expected_future_values(S_prime_exec_obs)
                    
                    # Loop through all possible opponent actions `b`
                    for b_opp in range(self.num_Adv_actions):
                        
                        # Store the value of V(s, b_opp) before the update
                        v_s_b_old = self.V[s, b_opp]
                        
                        # Extract transition probabilities conditioned on the opponent's TAKEN action
                        prob_exec_tensor_fixed_b = self.prob_exec_tensor[:, b_opp, :, :]
                        
                        # Calculate expected rewards and future values for each of our INTENDED actions
                        expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, R_exec_obs)
                        expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, future_V_values_executed)
                        
                        Q_values_for_dm_intentions = expected_rewards_all_idm + self.gamma * expected_future_V_all_idm
                        # The new value is the max Q-value
                        v_s_b_new = np.max(Q_values_for_dm_intentions)

                        # --- Update V in-place and track the change ---
                        self.V[s, b_opp] = v_s_b_new
                        delta = max(delta, np.abs(v_s_b_new - v_s_b_old))
                        
                # --- TQDM Update Logic ---
                
                # Calculate progress as a fraction from 0.0 to 1.0
                # 0.0 means not converged at all (delta is large)
                # 1.0 means fully converged (delta < theta)
                # We clip the value between 0 and 1.
                progress = max(0.0, 1.0 - (delta / theta))
                
                # Calculate the needed *increment* to update the bar.
                # This is more robust than setting .n directly.
                update_amount = progress - progress_bar.n
                if update_amount > 0:
                    progress_bar.update(update_amount)
                
                # Update the postfix to show the iteration and delta values.
                # Postfix is generally better for stats than the description.
                progress_bar.set_postfix(iter=i + 1, delta=f"{delta:.6f}", refresh=True)
                
                # --- Convergence Check ---
                if delta < theta:
                    # Ensure the bar is full upon completion.
                    if progress_bar.n < 1.0:
                        progress_bar.update(1.0 - progress_bar.n)
                    
                    print(f"\nValue iteration converged after {i + 1} iterations.")
                    return

        print(f"Value iteration did not converge after {max_iters} iterations.")


    def extract_optimal_policy(self):
        """
        Extracts the optimal deterministic policy π*(s) after value iteration.
        This reuses the parent's `optim_act` method.
        """
        print("Extracting optimal policy...")
        for s in range(self.n_states):
            # The parent's optim_act function finds the best action 'a' that maximizes
            # the expected Q-value, marginalized over the opponent's policy at state 's'.
            # This is exactly what we need for policy extraction.
            self.optim_policy_table[s] = self.optim_act(s)
        print("Optimal policy extracted.")

    def act(self, obs, env=None):
        """
        Returns the pre-computed optimal action for the given state.
        """
        return self.optim_policy_table[obs]

    def update(self, obs, actions, rewards, new_obs):
        """
        This agent is an offline solver. It does not learn during episodes.
        """
        pass
        
        
        