"""
This module implements several environments, i.e., the simulators in which agents will interact and learn.
Any environment is characterized by the following two methods:
 * step : receives the actions taken by the agents, and returns the new state of the simulator and the rewards
          perceived by each agent, amongst other things.
 * reset : sets the simulator at the initial state.
"""

import numpy as np

class CoinGame():
    """
    Coin game environment for two agents on a grid. Who batlle for two coins.
    """

    def __init__(self, max_steps=5, size_square_grid=4, push_distance=1):
        
        self.max_steps = max_steps  # Maximum number of steps per episode
        self.step_count = 0  # Counter for steps taken in the current episode
        self.N = size_square_grid  # Number of rows and columns in a square grid
        
        self.push_distance = push_distance # Distance to push the players away from each other
        
        self.available_actions_DM = np.array([0, 1, 2, 3])  # Actions available to the decision-maker
        self.available_actions_Adv = np.array([0, 1, 2, 3])  # Actions available to the adversary
        
        self.actions_push = np.array([0,1]) # push or not push
        
        # Creating combined actions for both players
        grid_A, grid_B = np.meshgrid(self.available_actions_DM, self.actions_push)
        self.combined_actions_blue = np.column_stack([grid_A.ravel(), grid_B.ravel()])
        
        grid_A, grid_B = np.meshgrid(self.available_actions_Adv, self.actions_push)
        self.combined_actions_red = np.column_stack([grid_A.ravel(), grid_B.ravel()])

        
        self.blue_player_execution_prob = 0.8 # Probability of executing the intended action for the blue player
        self.red_player_execution_prob = 0.8 # Probability of executing the intended action for the red player
        
        # Player positions
        self.blue_player = np.array([self.N // 2, 0]) # Centered start at left edge
        self.red_player = np.array([self.N // 2, self.N -1])   # Centered start at right edge
        
        self.coin_1 = np.array([0, self.N // 2]) # Initial position [row, col] of the coin 1
        self.coin_2 = np.array([self.N - 1, self.N // 2]) # Initial position [row, col] of the coin 2
        
        # Keep track of whether coins are still available on the map
        self.coin1_available = True
        self.coin2_available = True
        
        # Seve initial position of players and coins for reseting the environment
        self.blue_player_initial = self.blue_player.copy()  # Initial position [row, col] of the blue player (DM)
        self.red_player_initial = self.red_player.copy()  # Initial position [row, col] of the red player (ADV)
        
        self.coin_1_initial = self.coin_1.copy() # Initial position [row, col] of the coin 1
        self.coin_2_initial = self.coin_2.copy() # Initial position [row, col] of the coin 2
        
        self.done = False # Flag to indicate if episode is done
        
        # Track who collected which coin
        self.blue_collected_coin1 = False
        self.blue_collected_coin2 = False
        self.red_collected_coin1 = False
        self.red_collected_coin2 = False
        

    def get_state(self):
        """
        Returns a unique integer representing the full state of the environment,
        based the positions of the blue player, red player, coin availability, and collected coin counts.

        Each entity’s 2D position on an N×N grid is flattened and combined using radix encoding,
        ensuring every state has a unique ID.
        """

        # Flatten 2D positions into 1D indices (column-major order)
        p1 = self.blue_player[0] + self.N * self.blue_player[1]  # Blue player's position
        p2 = self.red_player[0] + self.N * self.red_player[1]    # Red player's position
        
        # Coin availability on map
        c1_avail_val = 1 if self.coin1_available else 0
        c2_avail_val = 1 if self.coin2_available else 0
        
        # How many coins each player has collected *so far*
        # (0, 1, or 2 - though game ends at 2 for one player)
        # For state simplicity, let's use: 0 = 0 coins, 1 = 1 coin, 2 = 2 coins (winning state)
        blue_coins_collected_count = int(self.blue_collected_coin1) + int(self.blue_collected_coin2)
        red_coins_collected_count = int(self.red_collected_coin1) + int(self.red_collected_coin2)
        
        # Radix encoding:
        # Max values: p_flat=N*N-1, c_avail=1, collected_count=2
        # Order: P1_pos, P2_pos, C1_avail, C2_avail, P1_coll_count, P2_coll_count
        base_pos = self.N * self.N
        base_avail = 2 # (0 or 1)
        base_coll_count = 3 # (0, 1, or 2)

        state_id = p1
        state_id = state_id * base_pos + p2
        state_id = state_id * base_avail + c1_avail_val
        state_id = state_id * base_avail + c2_avail_val
        state_id = state_id * base_coll_count + blue_coins_collected_count
        state_id = state_id * base_coll_count + red_coins_collected_count
        
        return int(state_id)
    
    @property
    def n_states(self):
        return (self.N**4) * (2**2) * (3**2)

    def reset(self):
        """
        Resets the environment to its initial state:
        - Step counter is set to 0
        - Player and coin positions are restored to their starting values
        """
        
        self.step_count = 0  # Reset step counter to start a new episode
        
        self.done = False # Reset done flag to indicate the episode is not finished

        # Set initial positions for the agents
        self.blue_player = self.blue_player_initial.copy()  # Starting position of the blue player
        self.red_player = self.red_player_initial.copy()  # Starting position of the red player

        # Set initial positions for the coins
        self.coin_1 = self.coin_1_initial.copy()   # Starting position of the coin 1
        self.coin_2 = self.coin_2_initial.copy()   # Starting position of the coin 2
        
        # Reset coin availability
        self.coin1_available = True
        self.coin2_available = True
        
        # Reset coin collection flags for each player
        self.blue_collected_coin1 = False
        self.blue_collected_coin2 = False
        self.red_collected_coin1 = False
        self.red_collected_coin2 = False

        return 
    
    def _are_players_adjacent(self, player1_pos, player2_pos):
        # Only checks if the players are horizontaly or verticaly adjacent, not diagonally
        return np.sum(np.abs(player1_pos - player2_pos)) == 1

    def step(self, action):
        """
        Executes one environment step given the actions of both agents.

        - Updates agent positions based on chosen actions
        - Handles coin collection and assigns corresponding rewards
        - Returns the new state, rewards for each agent, and whether the episode is done
        """
        
        # Unpack actions for both agents
        ac0, ac1 = action  # ac0: blue player's action, ac1: red player's action

        # Increment step count for this episode
        self.step_count += 1

        # Initialize step rewards
        reward_blue, reward_red = -0.1, -0.1
        
        # Store original positions for push logic
        original_blue_pos = self.blue_player.copy()
        original_red_pos = self.red_player.copy()
        
        # Check if players are adjacent 
        players_are_adjacent = self._are_players_adjacent(original_blue_pos, original_red_pos)
        
        # --- Effective actions ---
        
        # Define movement deltas for up, right, down, left
        deltas = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        
        # TODO: Apply radix decoding instead of this.
        # Randomly select the actual action based on the execution probability of the blue player
        if np.random.rand() < self.blue_player_execution_prob:
            actual_action_blue = self.combined_actions_blue[ac0]
        else:
            num_total_blue_actions = len(self.combined_actions_blue)
            alternatives = [a for a in range(num_total_blue_actions) if a != ac0]
            actual_action_blue = self.combined_actions_blue[np.random.choice(alternatives)]   

        ## Red player movement 
        
        # Randomly select the actual action based on the execution probability of the red player
        if np.random.rand() < self.red_player_execution_prob:
            actual_action_red = self.combined_actions_red[ac1]
        else:
            num_total_red_actions = len(self.combined_actions_red)
            alternatives = [a for a in range(num_total_red_actions) if a != ac1]
            actual_action_red = self.combined_actions_red[np.random.choice(alternatives)]   
            
    
        # --- 1. Resolve Push Attempts ---
        
        if actual_action_blue[1] == 1 and actual_action_red[1] == 1:
            if players_are_adjacent:
               reward_blue = -0.05
               reward_red = -0.05
            else: 
               reward_blue = -0.05 # Small penalty for failed push (not adjacent)   
               reward_red = -0.05 # Small penalty for failed push (not adjacent) 
               
        elif actual_action_blue[1] == 1 and actual_action_red[1] != 1: 
            if players_are_adjacent: 
                push_direction = original_red_pos - original_blue_pos
                pushed_red_player_position = original_red_pos + self.push_distance * push_direction
                
                self.red_player = np.clip(pushed_red_player_position, 0, self.N-1)
                reward_blue = 0.2 # Reward blue for succesfull push
                reward_red = -0.2 # Penilize red for no being pushed
            else: 
                reward_blue = -0.05 # Penilize blue if push was chosen but agents were not adjacent
                
        elif actual_action_blue[1] != 1 and actual_action_red[1] == 1: 
            if players_are_adjacent: 
                push_direction = original_blue_pos - original_red_pos
                pushed_blue_player_position = original_blue_pos + self.push_distance * push_direction
                
                self.blue_player = np.clip(pushed_blue_player_position, 0, self.N-1)
                reward_blue = -0.2 # Reward blue for succesfull push
                reward_red = 0.2 # Penilize red for no being pushed
            else: 
                reward_red = -0.05 # Penilize red if push was chosen but agents were not adjacent
        
        # Apply movement
        if actual_action_blue[1] != 1:
            new_position = self.blue_player + deltas[actual_action_blue[0]]
            new_position = np.clip(new_position, 0, self.N - 1)
            self.blue_player = new_position
                
        # Apply movement
        if actual_action_red[1] != 1:
            new_position = self.red_player + deltas[actual_action_red[0]]
            new_position = np.clip(new_position, 0, self.N - 1)
            self.red_player = new_position

        # --- Coin collection logic ---
        
        # Coin 1
        if self.coin1_available:
            blue_on_coin_1 = np.array_equal(self.blue_player, self.coin_1)
            red_on_coin_1 = np.array_equal(self.red_player, self.coin_1)

            if blue_on_coin_1 and not red_on_coin_1:
                self.blue_collected_coin1 = True
                self.coin1_available = False
                reward_blue = 2.0 # Add to base penalty, or set directly
                reward_red = -0.5 # Optional: small penalty for opponent scoring
                
            elif red_on_coin_1 and not blue_on_coin_1:
                self.red_collected_coin1 = True
                self.coin1_available = False
                reward_red = 2.0
                reward_blue = -0.5

            #elif blue_on_coin_1 and red_on_coin_1: # Contested
                # reward_blue and reward_red keep their -0.1, or add a specific contest penalty

        # Coin 2
        if self.coin2_available:
            blue_on_coin_2 = np.array_equal(self.blue_player, self.coin_2)
            red_on_coin_2 = np.array_equal(self.red_player, self.coin_2)

            if blue_on_coin_2 and not red_on_coin_2:
                self.blue_collected_coin2 = True
                self.coin2_available = False
                reward_blue = 2.0
                reward_red = -0.5
                
            elif red_on_coin_2 and not blue_on_coin_2:
                self.red_collected_coin2 = True
                self.coin2_available = False
                reward_red = 2.0
                reward_blue = -0.5
                
            #elif blue_on_coin_2 and red_on_coin_2: # Contested
                

        # After all coin collection attempts for the step:
        if (self.blue_collected_coin1 and self.blue_collected_coin2) or \
            (self.red_collected_coin1 and self.red_collected_coin2):
            # Blue winds
            if (self.blue_collected_coin1 and self.blue_collected_coin2):
                reward_blue = 10 # Win bonus
                reward_red = -5  # Loss penalty
                
            # Red wins
            else: 
                reward_red = 10 # Win bonus
                reward_blue = -5 # Loss penalty
            self.done = True
        elif (self.blue_collected_coin1 and self.red_collected_coin2) or \
            (self.red_collected_coin1 and self.blue_collected_coin2 ):
                reward_blue = -1
                reward_red = -1
                self.done = True
        
        # Check if max step coun is reached
        if self.step_count == self.max_steps:
            self.done = True

        # Return new state, rewards, and done flag
        return self.get_state(), np.array([reward_blue, reward_red]), self.done