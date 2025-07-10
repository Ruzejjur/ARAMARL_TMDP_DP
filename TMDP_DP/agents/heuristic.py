import numpy as np
from numpy.random import choice

from .utils import manhattan_distance

# --- Type Aliases for Readability ---
State = int
Action = int
Policy = np.ndarray

class ManhattanAgent():
    """
    A heuristic agent that moves towards the closest available coin.

    This agent uses the Manhattan distance to determine the closest coin and
    moves towards it. It will use the "push" action if it is adjacent to the
    opponent. If two coins are equidistant, it chooses one to target randomly.

    Attributes:
        grid_n (int): The size of the grid (N x N).
        player_id (int): The agent's identifier (0, 1).
        coin_location (np.ndarray): The coordinates of the two coins.
        optimal_actions_cache (np.ndarray): Caches the list of optimal actions for
            the last observed state. Used by DPAgent_PerfectModel.
        action_space (np.ndarray): The set of all possible actions.
    """
    def __init__(self, action_space: np.ndarray, coin_location: np.ndarray, grid_size: int, player_id: int):
        
        # The size of the N x N grid
        self.grid_size = grid_size
        
        # Player identifier (0 or 1)
        self.player_id = player_id
        
        # Fixed locations of the coins
        self.coin_location = coin_location
        
        # Caches the optimal actions for the most recent state. This is primarily
        # for inspection by offline solvers like DPAgent_PerfectModel.
        self.optimal_actions_cache = None
        
        # The agent's full action space
        self.action_space = action_space
        
    def decode_state(self, obs:State) -> tuple:
        """
        Decodes the integer state representation to get player positions and
        coin availability.

        Args:
            obs: The integer state ID.

        Returns:
            A tuple containing:
            - player_0_pos (np.ndarray): Coordinates of player 0.
            - player_1_pos (np.ndarray): Coordinates of player 1.
            - coin0_available (bool): True if coin 1 has not been collected.
            - coin1_available (bool): True if coin 2 has not been collected.
        """
        
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
        player_0_pos = np.array([p1_flat % self.grid_size, p1_flat // self.grid_size])
        player_1_pos = np.array([p2_flat % self.grid_size, p2_flat // self.grid_size])
        
        # Setting coin availability based on coin collection indicators from the state
        coin0_available = not (c_b1 or c_r1)
        coin1_available = not (c_b2 or c_r2)

        return player_0_pos, player_1_pos, coin0_available, coin1_available
    
    def _are_players_adjacent(self, player1_pos: np.ndarray, player2_pos: np.ndarray) -> bool:
        """
        Checks if players are in immediately neighboring cells (including diagonals).
        """
        
        # Calculate the absolute difference in row and column coordinates
        row_diff_abs = np.abs(player1_pos[0] - player2_pos[0])
        col_diff_abs = np.abs(player1_pos[1] - player2_pos[1])
        
        # For 8-directional adjacency (including diagonals):
        # - The maximum coordinate difference must be 1.
        # - This also implies they are not on the same cell (where max diff would be 0).
        return np.max([row_diff_abs, col_diff_abs]) == 1
    
    def compute_possible_actions(self, target_direction_vector: np.ndarray, players_are_adjacent: bool) -> Policy:
        """
        Determines optimal move(s) based on the direction to the target.

        A diagonal direction results in two optimal moves (e.g., Up and Left).
        Also determines if a push action should be combined with the move.

        Args:
            target_direction_vector: A vector [row_diff, col_diff] pointing to the target.
            players_are_adjacent: True if the opponent is in a neighboring cell.

        Returns:
            A numpy array of integer action IDs for the optimal moves.
        """
        possible_moves = []
        row_dir, col_dir = target_direction_vector[0], target_direction_vector[1]
        
        # If there's no clear direction (e.g., on top of the target), any move is possible.
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
        optimal_actions = np.array(possible_moves, dtype=int)

        # If adjacent, combine the move with a push. In the environment's action
        # encoding, push actions are offset by 4 from their non-push counterparts.
        if players_are_adjacent:
            optimal_actions += 4
            
        return optimal_actions

    def act(self, obs:State, env=None) -> Action:
        """
        Decides which action to take by finding the closest available coin and
        moving towards it, pushing if the opponent is adjacent.
        """
        # Decoding radix encoded state
        player_0_pos, player_1_pos, coin0_available, coin1_available = self.decode_state(obs)

        # Determine the agent's current position based on player_id
        player_pos = player_0_pos if self.player_id == 0 else player_1_pos

        # Check if player are adjacent
        players_are_adjacent = self._are_players_adjacent(player_0_pos, player_1_pos)
        
        # Set coin positions
        coin1_pos = self.coin_location[0]
        coin2_pos = self.coin_location[1]
        
        # Calculate Manhattan distances to each coin, if it's available
        # Set distance do ininity if not
        dist_to_coin1 = manhattan_distance(player_pos, coin1_pos) if coin0_available else float('inf')
        dist_to_coin2 = manhattan_distance(player_pos, coin2_pos) if coin1_available else float('inf')

        # Calculate target directional vector
        target_direction = None
        
        # --- Target Selection Logic ---
        # Handle case where no coins are available. Move randomly.
        if not coin0_available and not coin1_available:
            return choice(np.array([0, 2]), p=[0.5, 0.5])

        # Decide which coin to target based on distance
        if dist_to_coin1 < dist_to_coin2:
            target_direction = coin1_pos - player_pos
        elif dist_to_coin2 < dist_to_coin1:
            target_direction = coin2_pos - player_pos
        else: # Distances are equal (or only one coin is left)
            # Handle tie in distance: target one of the two coins randomly.
            if coin0_available and coin1_available:
                chosen_coin_pos = coin1_pos if np.random.rand() < 0.5 else coin2_pos
                target_direction = chosen_coin_pos - player_pos
            # Otherwise, target the only available coin
            elif coin0_available:
                target_direction = coin1_pos - player_pos
            else: # only coin2 is available
                target_direction = coin2_pos - player_pos

        # Compute optimal actions based on the target and cache them.
        self.optimal_actions_cache = self.compute_possible_actions(target_direction, players_are_adjacent)
        
        # Choose one of the optimal actions uniformly at random.
        return np.random.choice(self.optimal_actions_cache)
        
    
class ManhattanAgent_Passive(ManhattanAgent):
    """
    A Manhattan agent that never uses the push action.
    It follows the base ManhattanAgent's logic for targeting coins but will
    never choose to push, even when adjacent to an opponent.
    """
    def compute_possible_actions(self, target_direction_vector: np.ndarray, players_are_adjacent:bool) -> Policy:
        """
        Override the parent's method to always ignore player adjacency.
        This ensures the push action (radix shift) is never applied.
        """
        return super().compute_possible_actions(target_direction_vector, False)
    
class ManhattanAgent_Aggressive(ManhattanAgent):
    """
    A Manhattan agent that may target the opponent.
    
    This agent considers the opponent as a potential target. It will move
    towards whichever is closer: coin1, coin2, or the opponent. In case of a
    tie between a coin and the opponent, it prioritizes the coin. It always
    pushes when adjacent.
    """
    
    def act(self, obs:State, env=None) -> Action:
        """
        Decides an action by targeting the closest object, be it a coin or the opponent.

        The agent's logic is as follows:
        1.  Calculates the Manhattan distance to each available coin and to the opponent.
        2.  Identifies the target with the minimum distance.
        3.  Tie-breaking: If the opponent is tied with a coin for the minimum
            distance, the agent will prioritize targeting the coin. If multiple
            targets of the same type are tied (e.g., two coins), it chooses randomly.
        4.  Computes a direction vector towards the chosen target.
        5.  Determines the optimal move or moves to reduce the distance.
        6.  Always combines the move with a "push" if adjacent to the opponent.
        7.  Caches the resulting optimal action(s) and returns one chosen
            uniformly at random.

        Args:
            obs: The current state observation.
            env: The environment (not used in this heuristic agent).

        Returns:
            The integer ID of the chosen action.
        """
        # Decoding radix encoded state
        player_0_pos, player_1_pos, coin0_available, coin1_available = self.decode_state(obs)

        # Determine the agent's and opponent's current position based on player_id
        player_pos = player_0_pos if self.player_id == 0 else player_1_pos
        opponent_pos = player_1_pos if self.player_id == 0 else player_0_pos
        
        # Check if player are adjacent
        players_are_adjacent = self._are_players_adjacent(player_0_pos, player_1_pos)
        
        # Set coin positions
        coin1_pos = self.coin_location[0]
        coin2_pos = self.coin_location[1]
        
        # --- Target Selection Logic --
        # Location of target objects array 
        target_locations = np.array([coin1_pos, coin2_pos, opponent_pos])
        
        # Calculate Manhattan distances to each coin, if it's available
        # Set distance do ininity if not.
        dist_to_coin1 = manhattan_distance(player_pos, coin1_pos) if coin0_available else float('inf')
        dist_to_coin2 = manhattan_distance(player_pos, coin2_pos) if coin1_available else float('inf')
        # Calculate Manhattan distances to opponent
        dist_player_to_opponent = manhattan_distance(player_pos, opponent_pos)

        # Array of distances 
        dist_array = np.array([dist_to_coin1, dist_to_coin2, dist_player_to_opponent])
        
        # Detect indicies of equal distances to targets
        min_dist_idx = np.flatnonzero(dist_array == np.min(dist_array))

        # Tie-breaking: If the opponent (index 2) is tied with a coin for the
        # minimum distance, prioritize the coin.
        if 2 in min_dist_idx and (0 in min_dist_idx or 1 in min_dist_idx):
            min_dist_idx = np.argmin(dist_array[0:2]) # Re-evaluate distance between coins
        else:
            # Break ties randomly
            min_dist_idx = choice(min_dist_idx)
        
        # Calculate target directional vector
        target_direction = target_locations[min_dist_idx] - player_pos
    
        # Compute optimal actions based on the target and cache them.
        self.optimal_actions_cache = self.compute_possible_actions(target_direction, players_are_adjacent)
        
        # Choose one of the optimal actions uniformly at random.
        return np.random.choice(self.optimal_actions_cache)
        
    
class ManhattanAgent_Ultra_Aggressive(ManhattanAgent):
    """
    An aggressive agent that targets based on the 'most urgent' threat on the board.

    It calculates five key distances: player-to-coin1/2, opponent-to-coin1/2,
    and player-to-opponent. It targets the subject of the minimum distance,
    allowing it to dynamically switch between scoring (player-coin dist is min),
    intercepting (opponent-coin dist is min), and attacking (player-opponent dist
    is min). It always pushes when adjacent to opponent.
    """
    
    def act(self, obs:State, env=None) -> Action:
        """
        Selects an action based on the most 'urgent' tactical situation.

        This agent's logic is designed to be highly dynamic by evaluating five
        key distances to determine the most pressing action:
        1.  **Player to Coin 1/2:** The distance for the agent to score.
        2.  **Opponent to Coin 1/2:** The distance for the opponent to score.
            A short distance here implies an urgent need to intercept.
        3.  **Player to Opponent:** The distance for a direct attack.

        The agent's decision process is as follows:
        1.  It calculates all five distances.
        2.  It finds the absolute minimum distance among them. This identifies the
            most urgent situation (e.g., opponent is one step from a coin).
        3.  The agent targets the subject of that minimum distance. If it's an
            interception, it moves towards the threatened coin.
        4.  **Tie-breaking:** If attacking the opponent is tied with any other
            objective (scoring or intercepting), the other objective is always
            prioritized. Ties between non-attack objectives are broken randomly.
        5.  It always combines its move with a "push" if adjacent to the opponent.
        6.  The resulting optimal action(s) are cached, and one is returned.

        Args:
            obs: The current state observation.
            env: The environment (not used in this heuristic agent).

        Returns:
            The integer ID of the chosen action.
        """
        # Decoding radix encoded state
        player_0_pos, player_1_pos, coin0_available, coin1_available = self.decode_state(obs)

        # Determine the agent's and opponent's current position based on player_id
        player_pos = player_0_pos if self.player_id == 0 else player_1_pos
        opponent_pos = player_1_pos if self.player_id == 0 else player_0_pos
        
        # Check if player are adjacent
        players_are_adjacent = self._are_players_adjacent(player_0_pos, player_1_pos)
        
        # Set coin positions
        coin1_pos = self.coin_location[0]
        coin2_pos = self.coin_location[1]
        
        # --- Target Selection Logic ---
        # Location of target objects array 
        target_locations = np.array([coin1_pos, coin2_pos, coin1_pos, coin2_pos, opponent_pos])
        
        # Calculate Manhattan distances to each coin, if it's available
        # Set distance do ininity if not.
        # NOTE: Distances: p=player, o=opponent, c1/c2=coins
        dist_p_c1 = manhattan_distance(player_pos, coin1_pos) if coin0_available else float('inf')
        dist_p_c2 = manhattan_distance(player_pos, coin2_pos) if coin1_available else float('inf')
        dist_o_c1 = manhattan_distance(opponent_pos, coin1_pos) if coin0_available else float('inf')
        dist_o_c2 = manhattan_distance(opponent_pos, coin2_pos) if coin1_available else float('inf')
        # Calculate Manhattan distances to opponent
        dist_p_o = manhattan_distance(player_pos, opponent_pos)
        
        # Array of distances 
        dist_array = np.array([dist_p_c1, dist_p_c2, dist_o_c1, dist_o_c2, dist_p_o])

        # Detect indicies of equal distances to targets
        min_dist_idx = np.flatnonzero(dist_array == np.min(dist_array))

        # Tie-breaking: If attacking the opponent (index 4) is tied with any
        # other objective, prioritize the other objective (coin or intercept).
        if 4 in min_dist_idx and any(i in min_dist_idx for i in (0, 1, 2, 3)):
            min_dist_idx = np.argmin(dist_array[0:4]) # Re-evaluate distance between coins
        else:
            # Break ties randomly
            min_dist_idx = choice(min_dist_idx)
        
        # Calculate target directional vector
        target_direction = target_locations[min_dist_idx] - player_pos
    
        # Compute optimal actions based on the target and cache them.
        self.optimal_actions_cache = self.compute_possible_actions(target_direction, players_are_adjacent)
        
        # Choose one of the optimal actions uniformly at random.
        return np.random.choice(self.optimal_actions_cache)
        
    