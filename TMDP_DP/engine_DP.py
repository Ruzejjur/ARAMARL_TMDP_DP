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
        
        self.available_move_actions_DM = np.array([0, 1, 2, 3])  # Movement actions available to the decision-maker
        self.available_move_actions_Adv = np.array([0, 1, 2, 3])  # Movement actions available to the adversary
        
        self.actions_push = np.array([0,1]) # No push and push actions available to both players
        
        # Creating combined actions for both players
        grid_A, grid_B = np.meshgrid(self.available_move_actions_DM, self.actions_push)
        self.combined_actions_blue = np.column_stack([grid_A.ravel(), grid_B.ravel()])
        
        grid_A, grid_B = np.meshgrid(self.available_move_actions_Adv, self.actions_push)
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
        based on the positions of the blue player, red player, coin availability, and collected coin counts.

        Each entity’s 2D position on an N×N grid is flattened and combined using radix encoding,
        ensuring every state has a unique ID.
        """

        # Flatten 2D positions into 1D indices (column-major order)
        p1 = self.blue_player[0] + self.N * self.blue_player[1]  # Blue player's position
        p2 = self.red_player[0] + self.N * self.red_player[1]    # Red player's position
        
        # Radix encoding:
        # Max values: p_flat=N*N-1, c_avail=1, collected_count=2
        # Order: P1_pos, P2_pos, C1_avail, C2_avail, P1_coll_count, P2_coll_count
        base_pos = self.N * self.N
        base_coll = 2 # 0 or 1 for each coin

        state_id = p1
        state_id = state_id * base_pos + p2
        state_id = state_id * base_coll + int(self.blue_collected_coin1)
        state_id = state_id * base_coll + int(self.blue_collected_coin2)
        state_id = state_id * base_coll + int(self.red_collected_coin1)
        state_id = state_id * base_coll + int(self.red_collected_coin2)
        
        return int(state_id)
    
    @property
    def n_states(self):
        return (self.N**4) * (2**4)

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
        # Checks if players are in immediately neighboring cells (8 directions)
        
        # Calculate the absolute difference in row and column coordinates
        row_diff_abs = np.abs(player1_pos[0] - player2_pos[0])
        col_diff_abs = np.abs(player1_pos[1] - player2_pos[1])
        
        # For 8-directional adjacency (including diagonals):
        # - The maximum coordinate difference must be 1.
        # - This also implies they are not on the same cell (where max diff would be 0).
        return np.max([row_diff_abs, col_diff_abs]) == 1

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
        deltas = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)]) # 0:Up, 1:Right, 2:Down, 3:Left
        
        ## --- Blue player action resolution ---

        # Get the intended combined action (move, push) for blue
        intended_combined_action_blue = self.combined_actions_blue[ac0]
        intended_move_value_blue = intended_combined_action_blue[0] 
        intended_push_action_blue = intended_combined_action_blue[1]
        
        # Stochastic selection for the movememnt part
        if np.random.rand() < self.blue_player_execution_prob:
            actual_move_action_blue = intended_move_value_blue
        else:
            # Choose a different move action randomly from the available move actions
            possible_move_values_blue = self.available_move_actions_DM

            alternatives_blue = [a for a in possible_move_values_blue if a != intended_move_value_blue]
            actual_move_action_blue = np.random.choice(alternatives_blue)   
        
        # Final actual action for blue player (stochastic move, deterministic push)
        actual_action_blue = [actual_move_action_blue, intended_push_action_blue]

        ## --- Red player action resolution ---

        # Get the intended combined action (move, push) for red
        intended_combined_action_red = self.combined_actions_red[ac1]
        intended_move_value_red = intended_combined_action_red[0]  
        intended_push_action_red = intended_combined_action_red[1]
        
        # Stochastic selection for the movement part
        if np.random.rand() < self.red_player_execution_prob:
            actual_move_action_red = intended_move_value_red
        else:
            # Choose a different move action randomly from the available move actions
            possible_move_values_red = self.available_move_actions_Adv # Corrected: use _Adv
            
            alternatives_red = [a for a in possible_move_values_red if a != intended_move_value_red]
            actual_move_action_red = np.random.choice(alternatives_red)

        # Final actual action for red player (stochastic move, deterministic push)
        actual_action_red = [actual_move_action_red, intended_push_action_red]
            
    
        # --- Resolve Push Attempts ---
        
        pushed_blue_this_step = False # Flag to track if blue player was pushed this step
        pushed_red_this_step = False # Flag to track if red player was pushed this step
        
        if actual_action_blue[1] == 1 and actual_action_red[1] == 1: # Both players attempts to push each other
            if players_are_adjacent:
               reward_blue += -0.05
               reward_red += -0.05
            else: 
               reward_blue += -0.05 # Small penalty for failed push (not adjacent)   
               reward_red += -0.05 # Small penalty for failed push (not adjacent) 
               
        elif actual_action_blue[1] == 1: # Only blue player attempts to push
            if players_are_adjacent: 
                push_direction = original_red_pos - original_blue_pos
                pushed_red_player_position = original_red_pos + self.push_distance * push_direction
                
                self.red_player = np.clip(pushed_red_player_position, 0, self.N-1)
                reward_blue += 0.4 # Reward blue for succesfull push (net +0.3 after push)
                reward_red += -0.2 # Penilize red for being pushed (net -0.3 after push)
                pushed_red_this_step = True
            else: 
                reward_blue += -0.05 # Penilize blue if push was chosen but agents were not adjacent
                
        elif actual_action_red[1] == 1: # Only red player attempts to push
            if players_are_adjacent: 
                push_direction = original_blue_pos - original_red_pos
                pushed_blue_player_position = original_blue_pos + self.push_distance * push_direction
                
                self.blue_player = np.clip(pushed_blue_player_position, 0, self.N-1)
                reward_blue += -0.2 # Penilize blue for being pushed (net -0.3 after push)
                reward_red += 0.4 # Reward red for succesfull push (net +0.3 after push)
                pushed_blue_this_step = True 
            else: 
                reward_red += -0.05 # Penilize red if push was chosen but agents were not adjacent
        
        # --- Apply Movement ---
        
        out_of_boundry_penalty = -0.5 # Adjust as needed
        
        # Copy the current positions of agents for further checking if
        # agents try to move to the same position as the other one
        new_blue_position = self.blue_player.copy()
        new_red_position = self.red_player.copy()
        
        # Blue Player Movement (only if not pushed by Red this turn)
        if not pushed_blue_this_step:
            new_blue_position = original_blue_pos + deltas[actual_action_blue[0]]
            
            # Check if the candidate position is out of bounds and penilize if so
            if new_blue_position[0] < 0 or new_blue_position[0] >= self.N or \
                new_blue_position[1] < 0 or new_blue_position[1] >= self.N:
                reward_blue += out_of_boundry_penalty
                
        # Red Player Movement (only if not pushed by Blue this turn)
        if not pushed_red_this_step:
            new_red_position = original_red_pos + deltas[actual_action_red[0]]
            
            # Check if the candidate position is out of bounds and penilize if so
            if new_red_position[0] < 0 or new_red_position[0] >= self.N or \
                new_red_position[1] < 0 or new_red_position[1] >= self.N:
                reward_red += out_of_boundry_penalty
            
        # Check if agents did not end up on the same spot
        # Penilize both agents for this
        if not np.array_equal(new_blue_position, new_red_position):
            self.blue_player = np.clip(new_blue_position, 0, self.N - 1)
            self.red_player = np.clip(new_red_position, 0, self.N - 1)
        else: 
            reward_blue += -0.05
            reward_red += -0.05
            
        
        # --- Coin collection logic ---
        
        # Coin 1
        if self.coin1_available:
            blue_on_coin_1 = np.array_equal(self.blue_player, self.coin_1)
            red_on_coin_1 = np.array_equal(self.red_player, self.coin_1)

            if blue_on_coin_1 and not red_on_coin_1:
                self.blue_collected_coin1 = True
                self.coin1_available = False
                reward_blue += 2.0 
                reward_red += -0.5
                
            elif red_on_coin_1 and not blue_on_coin_1:
                self.red_collected_coin1 = True
                self.coin1_available = False
                reward_red += 2.0
                reward_blue += -0.5
                
            elif blue_on_coin_1 and red_on_coin_1: # Contested Coin 1
                # No one gets it, small penalty for both.
                reward_blue += -0.2
                reward_red += -0.2
  

        # Coin 2
        if self.coin2_available:
            blue_on_coin_2 = np.array_equal(self.blue_player, self.coin_2)
            red_on_coin_2 = np.array_equal(self.red_player, self.coin_2)

            if blue_on_coin_2 and not red_on_coin_2:
                self.blue_collected_coin2 = True
                self.coin2_available = False
                reward_blue += 2.0
                reward_red += -0.5
                
            elif red_on_coin_2 and not blue_on_coin_2:
                self.red_collected_coin2 = True
                self.coin2_available = False
                reward_red += 2.0
                reward_blue += -0.5
            elif blue_on_coin_2 and red_on_coin_2: # Contested Coin 2
                # No one gets it, small penalty for both.
                reward_blue += -0.2 
                reward_red += -0.2

        # --- Game End Conditions & Final Rewards (These will OVERWRITE step rewards) --

        blue_wins = self.blue_collected_coin1 and self.blue_collected_coin2
        red_wins = self.red_collected_coin1 and self.red_collected_coin2
        
        # Draw condition: both coins are gone, and no one has two.
        # This includes scenarios where each player has one coin.
        is_draw = (not self.coin1_available and not self.coin2_available) and not blue_wins and not red_wins

        if blue_wins:
            reward_blue = 10 
            reward_red = -10  
            self.done = True
        elif red_wins:
            reward_red = 10 
            reward_blue = -10
            self.done = True
        elif is_draw:
            reward_blue = -5 
            reward_red = -5  
            self.done = True
        
        # Check if max step count is reached without win of either agents or draw and penilze if so
        if not self.done and self.step_count >= self.max_steps:
            self.done = True
            
            # Consider if players have any coins when timeout occurs
            blue_coins_at_timeout = int(self.blue_collected_coin1) + int(self.blue_collected_coin2)
            red_coins_at_timeout = int(self.red_collected_coin1) + int(self.red_collected_coin2)

            if blue_coins_at_timeout > red_coins_at_timeout:
                reward_blue += 1 # Small bonus for leading at timeout
                reward_red += -1
            elif red_coins_at_timeout > blue_coins_at_timeout:
                reward_red += 1
                reward_blue += -1
            else: # Equal coins or zero coins for both at timeout
                reward_blue += -2 # Timeout penalty if no clear leader or no progress
                reward_red += -2

        # Return new state, rewards, and done flag
        return self.get_state(), np.array([reward_blue, reward_red]), self.done