"""
This module implements the simulation environment for the project.

Environments define the world in which agents interact. They are primarily
characterized by two methods:
- step: Receives actions from agents and returns the new state, rewards, and
        a done flag.
- reset: Resets the simulator to its initial state for a new episode.
"""
import numpy as np

class CoinGame():
    """
    A 2-player grid-based environment where agents compete to collect coins.

    In this game, two players (Player 0 and Player 1) navigate an N x N grid.
    Two coins are placed on the grid. The objective is to collect both coins before
    the opponent does. Agents can also "push" an adjacent opponent to move them.

    State Representation:
    The state is encoded into a single integer using radix encoding, capturing:
    - Player 0's position
    - Player 1's position
    - Collection status of both coins for each player

    Action Space:
    Actions are combined from a move (Up, Right, Down, Left) and a push (Push, No Push).

    Attributes:
        max_steps (int): The maximum number of steps allowed per episode.
        grid_size (int): The size of one dimension of the square grid.
        push_distance (int): The number of cells an opponent is moved when pushed.
        step_count (int): The current step count within the episode.
        done (bool): A flag indicating if the episode has terminated.
        
        player_0_pos (np.ndarray): The [row, col] coordinates of Player 0.
        player_1_pos (np.ndarray): The [row, col] coordinates of Player 1.
        coin_0_pos (np.ndarray): The [row, col] coordinates of Coin 1.
        coin_1_pos (np.ndarray): The [row, col] coordinates of Coin 2.

        player_0_execution_prob (float): Probability of Player 0's intended move succeeding.
        player_1_execution_prob (float): Probability of Player 1's intended move succeeding.

        available_move_actions (np.ndarray): Array of possible move actions [0, 1, 2, 3].
        available_push_actions (np.ndarray): Array of push actions [0, 1].
        combined_actions (np.ndarray): All possible combinations of (move, push) actions.
    """
    
    # --- Reward and Penalty Constants ---
        # NOTE: _DELTA suffix indicates the value is added to the running reward total.
    #       Values without the suffix will overwrite the running total.
    _STEP_PENALTY = -0.1
    
    _PUSH_REWARD_DELTA = 0.4
    _PUSH_PENALTY_DELTA = -0.2
    _PUSH_BUT_NOT_ADJACENT_PENALTY_DELTA = -0.05
    _BOTH_PUSH_PENALTY_DELTA = -0.05
    
    _COLLISION_PENALTY_DELTA = -0.05
    _OUT_OF_BOUNDS_PENALTY_DELTA = -0.5
    
    _COIN_REWARD_DELTA = 2.0
    _COIN_STEAL_PENALTY_DELTA = -0.5
    _CONTESTED_COIN_PENALTY_DELTA = -0.2
    
    _WIN_REWARD = 10
    _LOSS_PENALTY = -10
    _DRAW_PENALTY = -5
    
    _TIMEOUT_PENALTY_DELTA = -2
    _TIMEOUT_LEAD_BONUS_DELTA = 1
    _TIMEOUT_LEAD_LOSS_DELTA = -1
    

    def __init__(self, max_steps: int = 50, grid_size: int = 5, push_distance: int = 1):
        """
        Initializes the CoinGame environment.

        Args:
            max_steps (int): The maximum number of steps allowed per episode.
            grid_size (int): The size of one dimension of the square grid (N).
            push_distance (int): The number of cells an opponent is moved when pushed.
        """
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.push_distance = push_distance
        
        # --- Action Space Definition ---
        self.available_move_actions = np.array([0, 1, 2, 3])  # 0:Up, 1:Right, 2:Down, 3:Left
        self.available_push_actions = np.array([0, 1])        # 0:No push, 1:Push
        
        # Create all combined (move, push) actions, identical for both players
        grid_move, grid_push = np.meshgrid(self.available_move_actions, self.available_push_actions)
        self.combined_actions = np.column_stack([grid_move.ravel(), grid_push.ravel()])

        # --- Stochasticity Parameters ---
        self.player_0_execution_prob = 1.0
        self.player_1_execution_prob = 1.0
        
        # --- Initial Positions ---
        self.player_0_initial_pos = np.array([self.grid_size // 2, 0])
        self.player_1_initial_pos = np.array([self.grid_size // 2, self.grid_size - 1])
        
        self.coin_0_initial_pos = np.array([0, self.grid_size // 2])
        self.coin_1_initial_pos = np.array([self.grid_size - 1, self.grid_size // 2])
        
        # --- Dynamic State Variables ---
        self.step_count = 0
        self.done = False
        
        self.player_0_pos = self.player_0_initial_pos.copy()
        self.player_1_pos = self.player_1_initial_pos.copy()
        
        self.coin_0_pos = self.coin_0_initial_pos.copy()
        self.coin_1_pos = self.coin_1_initial_pos.copy()
        self.coin0_available = True
        self.coin1_available = True
        
        self.player_0_collected_coin0 = False
        self.player_0_collected_coin1 = False
        self.player_1_collected_coin0 = False
        self.player_1_collected_coin1 = False
        
        # --- Movement Deltas ---
        self._deltas = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]]) # Up, Right, Down, Left
        
        # Call reset at the beggining to be save
        self.reset()
        
    def _resolve_action(self, player_id: int, intended_action_id: int) -> tuple:
        """
        Determines the actual action executed by a player, accounting for stochasticity.

        Args:
            player_id (int): The ID of the player (0 or 1).
            intended_action_id (int): The action the player intended to take.

        Returns:
            Tuple[int, int]: A tuple of the (actual_move, actual_push) executed.
        """
        # Extract intended actions
        intended_move, intended_push = self.combined_actions[intended_action_id]
        # Set probability of execution based on player_id
        exec_prob = self.player_0_execution_prob if player_id == 0 else self.player_1_execution_prob
        
        # Evaluate if the intended action is executed or alternative is selected
        # Note: Push action is deterministic
        if np.random.rand() < exec_prob:
            actual_move = intended_move
        else:
            alternatives = [a for a in self.available_move_actions if a != intended_move]
            actual_move = np.random.choice(alternatives) if alternatives else intended_move
        
        return actual_move, intended_push

    def _resolve_coin_collection(self, coin_index: int) -> tuple:
            """
            Handles the logic for collecting a single coin.

            This helper method checks if a specific coin is available, determines if
            any player is on it, updates the game state (coin availability and
            collection flags) directly, and returns the rewards/penalties incurred.

            Args:
                coin_index (int): The index of the coin to resolve (0 or 1).

            Returns:
                tuple: A tuple containing the reward deltas
                    (reward_delta_0, reward_delta_1) for this event.
            """
            # Determine which coin attributes to use based on the index
            if coin_index == 0:
                if not self.coin0_available:
                    return 0.0, 0.0  # Coin already gone, no rewards
                coin_pos = self.coin_0_pos
            else: # coin_index == 2
                if not self.coin1_available:
                    return 0.0, 0.0
                coin_pos = self.coin_1_pos

            reward_delta_0, reward_delta_1 = 0.0, 0.0
            
            p0_on_c = np.array_equal(self.player_0_pos, coin_pos)
            p1_on_c = np.array_equal(self.player_1_pos, coin_pos)

            # Helper to update state, avoiding code repetition
            def update_collection_state(is_coin_0, p0_collects, p1_collects):
                if is_coin_0:
                    self.coin0_available = False
                    if p0_collects:
                        self.player_0_collected_coin0 = True
                    if p1_collects:
                        self.player_1_collected_coin0 = True
                else:
                    self.coin1_available = False
                    if p0_collects:
                        self.player_0_collected_coin1 = True
                    if p1_collects:
                        self.player_1_collected_coin1 = True

            # --- Resolve collection ---
            if p0_on_c and not p1_on_c:
                update_collection_state(coin_index == 0, p0_collects=True, p1_collects=False)
                reward_delta_0 += self._COIN_REWARD_DELTA
                reward_delta_1 += self._COIN_STEAL_PENALTY_DELTA
            elif p1_on_c and not p0_on_c:
                update_collection_state(coin_index == 0, p0_collects=False, p1_collects=True)
                reward_delta_1 += self._COIN_REWARD_DELTA
                reward_delta_0 += self._COIN_STEAL_PENALTY_DELTA
            elif p0_on_c and p1_on_c: # Contested coin
                reward_delta_0 += self._CONTESTED_COIN_PENALTY_DELTA
                reward_delta_1 += self._CONTESTED_COIN_PENALTY_DELTA
            
            return reward_delta_0, reward_delta_1

    def get_state(self) -> int:
        """
        Computes a unique integer ID for the current state using radix encoding.

        The encoding order is:
        Player 0 Pos -> Player 1 Pos -> P0 coin0 -> P0 coin1 -> P1 coin0 -> P1 coin1
        
        Returns:
            int: The unique integer representing the environment state.
        """
        # Flatten 2D positions to 1D indices for encoding.
        p0_flat = self.player_0_pos[0] + self.grid_size * self.player_0_pos[1]
        p1_flat = self.player_1_pos[0] + self.grid_size * self.player_1_pos[1]
        
        # Radix encoding: Each component is a "digit" in a mixed-radix number system.
        base_pos = self.grid_size * self.grid_size
        base_coll = 2  # Boolean (0 or 1)

        state_id = p0_flat
        state_id = state_id * base_pos + p1_flat
        state_id = state_id * base_coll + int(self.player_0_collected_coin0)
        state_id = state_id * base_coll + int(self.player_0_collected_coin1)
        state_id = state_id * base_coll + int(self.player_1_collected_coin0)
        state_id = state_id * base_coll + int(self.player_1_collected_coin1)
        
        return int(state_id)
    
    @property
    def n_states(self) -> int:
        """
        Calculates the total size of the state space.

        The size is (N^2)^2 * 2^4, from two player positions on an N*N grid and
        four independent coin collection flags.
        """
        return (self.grid_size ** 4) * (2 ** 4)

    def reset(self) -> None:
        """
        Resets the environment to its initial state for a new episode.

        This restores all player and coin positions, resets the step counter,
        and makes all coins available again.
        """
        self.step_count = 0
        self.done = False

        self.player_0_pos = self.player_0_initial_pos.copy()
        self.player_1_pos = self.player_1_initial_pos.copy()
        self.coin_0_pos = self.coin_0_initial_pos.copy()
        self.coin_1_pos = self.coin_1_initial_pos.copy()
        
        self.coin0_available = True
        self.coin1_available = True
        
        self.player_0_collected_coin0 = False
        self.player_0_collected_coin1 = False
        self.player_1_collected_coin0 = False
        self.player_1_collected_coin1 = False
    
    def _are_players_adjacent(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """
        Checks if two players are in immediately neighboring cells (8 directions).

        This is equivalent to checking if their Chebyshev distance is 1.

        Args:
            pos1 (np.ndarray): The [row, col] position of the first player.
            pos2 (np.ndarray): The [row, col] position of the second player.

        Returns:
            bool: True if players are adjacent, False otherwise.
        """
        return np.max(np.abs(pos1 - pos2)) == 1

    def step(self, actions: list) -> tuple:
        """
        Executes one time step in the environment.

        This involves resolving agent actions (including stochasticity and pushes),
        updating positions, handling coin collections, and determining rewards.

        Args:
            actions (list): A list [action_0, action_1] containing the
                integer action IDs for each player.

        Returns:
            tuple: A tuple containing:
                - The new state ID (int).
                - A numpy array of rewards [reward_0, reward_1].
                - A done flag (bool) indicating if the episode has ended.
        """
        
        # Unpack actions
        action_0, action_1 = actions
        # Increment step count
        self.step_count += 1
        
        # Set reward for step
        reward_0, reward_1 = self._STEP_PENALTY, self._STEP_PENALTY
        
        # Store original positions for push logic
        original_pos_0 = self.player_0_pos.copy()
        original_pos_1 = self.player_1_pos.copy()
        
        # Check if players are adjacent 
        players_are_adjacent = self._are_players_adjacent(original_pos_0, original_pos_1)
        
        # --- Action Resolution (Intended vs. Actual) ---

        actual_move_0, actual_push_0 = self._resolve_action(0, action_0)
        actual_move_1, actual_push_1 = self._resolve_action(1, action_1)
    
        # --- Resolve Push Attempts ---
        
        pushed_0_this_step, pushed_1_this_step = False, False
        if actual_push_0 and actual_push_1:  # Both push
            if players_are_adjacent:
                reward_0 += self._BOTH_PUSH_PENALTY_DELTA
                reward_1 += self._BOTH_PUSH_PENALTY_DELTA
            else: 
                reward_0 += self._PUSH_BUT_NOT_ADJACENT_PENALTY_DELTA
                reward_1 += self._PUSH_BUT_NOT_ADJACENT_PENALTY_DELTA
        elif actual_push_0:  # Player 0 pushes
            if players_are_adjacent:
                push_direction = original_pos_1 - original_pos_0
                self.player_1_pos = np.clip(original_pos_1 + self.push_distance * push_direction, 0, self.grid_size - 1)
                reward_0 += self._PUSH_REWARD_DELTA
                reward_1 += self._PUSH_PENALTY_DELTA
                pushed_1_this_step = True
            else:
                reward_0 += self._PUSH_BUT_NOT_ADJACENT_PENALTY_DELTA
        elif actual_push_1:  # Player 1 pushes
            if players_are_adjacent:
                push_direction = original_pos_0 - original_pos_1
                self.player_0_pos = np.clip(original_pos_0 + self.push_distance * push_direction, 0, self.grid_size - 1)
                reward_1 += self._PUSH_REWARD_DELTA
                reward_0 += self._PUSH_PENALTY_DELTA
                pushed_0_this_step = True
            else:
                reward_1 += self._PUSH_BUT_NOT_ADJACENT_PENALTY_DELTA
        
        # --- Apply Movement ---
        
        new_pos_0 = self.player_0_pos.copy()
        new_pos_1 = self.player_1_pos.copy()

        if not pushed_0_this_step:
            # Calculate candidate position
            candidate_pos = original_pos_0 + self._deltas[actual_move_0]
            # Resolve out of bounds movement
            if np.any(candidate_pos < 0) or np.any(candidate_pos >= self.grid_size):
                reward_0 += self._OUT_OF_BOUNDS_PENALTY_DELTA
            new_pos_0 = np.clip(candidate_pos, 0, self.grid_size - 1)
            
        if not pushed_1_this_step:
            # Calculate candidate position 
            candidate_pos = original_pos_1 + self._deltas[actual_move_1]
            # Resolve out of bounds movement
            if np.any(candidate_pos < 0) or np.any(candidate_pos >= self.grid_size):
                reward_1 += self._OUT_OF_BOUNDS_PENALTY_DELTA
            new_pos_1 = np.clip(candidate_pos, 0, self.grid_size - 1)
        
        # Resolve collisions: if agents move to the same spot, they bounce back.
        if not np.array_equal(new_pos_0, new_pos_1):
            self.player_0_pos = new_pos_0
            self.player_1_pos = new_pos_1
        else:
            reward_0 += self._COLLISION_PENALTY_DELTA
            reward_1 += self._COLLISION_PENALTY_DELTA
            
        
        # --- Coin collection logic ---
        
        # Coin 1
        delta_0, delta_1 = self._resolve_coin_collection(1)
        reward_0 += delta_0
        reward_1 += delta_1

        # Coin 2
        delta_0, delta_1 = self._resolve_coin_collection(2)
        reward_0 += delta_0
        reward_1 += delta_1

        # --- Game End Conditions (overwrite step rewards) ---

        p0_wins = self.player_0_collected_coin0 and self.player_0_collected_coin1
        p1_wins = self.player_1_collected_coin0 and self.player_1_collected_coin1
        
        # Draw condition: both coins are gone, and no one has two.
        # NOTE: This includes scenarios where each player has one coin.
        is_draw = (not self.coin0_available and not self.coin1_available) and not p0_wins and not p1_wins

        if p0_wins:
            reward_0, reward_1, self.done = self._WIN_REWARD, self._LOSS_PENALTY, True
        elif p1_wins:
            reward_0, reward_1, self.done = self._LOSS_PENALTY, self._WIN_REWARD, True
        elif is_draw:
            reward_0, reward_1, self.done = self._DRAW_PENALTY, self._DRAW_PENALTY, True
            
        # Check for timeout
        if not self.done and self.step_count >= self.max_steps:
            self.done = True
            p0_coins = int(self.player_0_collected_coin0) + int(self.player_0_collected_coin1)
            p1_coins = int(self.player_1_collected_coin0) + int(self.player_1_collected_coin1)
            if p0_coins > p1_coins:
                reward_0 += self._TIMEOUT_LEAD_BONUS_DELTA
                reward_1 += self._TIMEOUT_LEAD_LOSS_DELTA
            elif p1_coins > p0_coins:
                reward_1 += self._TIMEOUT_LEAD_BONUS_DELTA
                reward_0 += self._TIMEOUT_LEAD_LOSS_DELTA
            else:
                reward_0 += self._TIMEOUT_PENALTY_DELTA
                reward_1 += self._TIMEOUT_PENALTY_DELTA

        return self.get_state(), np.array([reward_0, reward_1]), self.done