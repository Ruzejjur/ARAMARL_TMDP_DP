import numpy as np
from numpy.random import choice

from .base import Agent
from .utils import manhattan_distance



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
        
        if row_dir == 0 and col_dir == 0:
            possible_moves = [0,1,2,3]
        
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
        
    