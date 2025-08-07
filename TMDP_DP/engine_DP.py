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

    def __init__(self, max_steps: int, grid_size: int, push_distance: int, rewards: dict):
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
        
        
        # --- Assign rewards from config to instance variables ---

        try:
            self.step_penalty = rewards['step_penalty']
            
            self.push_reward_delta = rewards['push_reward_delta']
            self.push_penalty_delta = rewards['push_penalty_delta']
            self.push_but_not_adjacent_penalty_delta = rewards['push_but_not_adjacent_penalty_delta']
            self.both_push_penalty_delta = rewards['both_push_penalty_delta']
            
            self.collision_penalty_delta = rewards['collision_penalty_delta']
            self.out_of_bounds_penalty_delta = rewards['out_of_bounds_penalty_delta']
            
            self.coin_reward_delta = rewards['coin_reward_delta']
            self.coin_steal_penalty_delta = rewards['coin_steal_penalty_delta']
            self.contested_coin_penalty_delta = rewards['contested_coin_penalty_delta']
            
            self.win_reward = rewards['win_reward']
            self.loss_penalty = rewards['loss_penalty']
            self.draw_penalty = rewards['draw_penalty']
            
            self.timeout_penalty_delta = rewards['timeout_penalty_delta']
            self.timeout_lead_bonus_delta = rewards['timeout_lead_bonus_delta']
            self.timeout_trail_penalty_delta = rewards['timeout_trail_penalty_delta']
        
        except KeyError as e: 
            raise KeyError(
                        f"The required reward key {e} was not found. "
                        "Please ensure all reward values are defined under "
                        "'environment_settings.params.rewards' in your config.yaml file."
                    )
        
        # --- Action Space Definition ---
        self.available_move_actions = np.array([0, 1, 2, 3])  # 0:Up, 1:Right, 2:Down, 3:Left
        self.available_push_actions = np.array([0, 1])        # 0:No push, 1:Push
        
        # Create all combined (move, push) actions, identical for both players
        grid_move, grid_push = np.meshgrid(self.available_move_actions, self.available_push_actions)
        self.combined_actions = np.column_stack([grid_move.ravel(), grid_push.ravel()])

        # --- Stochasticity Parameters ---
        self.player_0_execution_prob = 0.8
        self.player_1_execution_prob = 0.8
        
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
                tuple: A tuple containing the reward deltas and only the positive and negative rewards
                    (full_reward_delta_0, full_reward_delta_1, negative_reward_delta_0, negative_reward_delta_1, positive_reward_delta_0, positive_reward_delta_1) for this event.
            """
            # Determine which coin attributes to use based on the index
            if coin_index == 0:
                if not self.coin0_available:
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Coin already gone, no rewards
                coin_pos = self.coin_0_pos
            else: # coin_index == 1
                if not self.coin1_available:
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Coin already gone, no rewards
                coin_pos = self.coin_1_pos

            full_reward_delta_0, full_reward_delta_1 = 0.0, 0.0
            
            positive_reward_delta_0, positive_reward_delta_1 = 0.0, 0.0
            
            negative_reward_delta_0, negative_reward_delta_1 = 0.0, 0.0
            
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
                
                full_reward_delta_0 += self.coin_reward_delta
                positive_reward_delta_0 += self.coin_reward_delta
                
                full_reward_delta_1 += self.coin_steal_penalty_delta
                negative_reward_delta_1 += self.coin_steal_penalty_delta
                
            elif p1_on_c and not p0_on_c:
                update_collection_state(coin_index == 0, p0_collects=False, p1_collects=True)
                
                full_reward_delta_1 += self.coin_reward_delta
                positive_reward_delta_1 += self.coin_reward_delta
                
                
                full_reward_delta_0 += self.coin_steal_penalty_delta
                negative_reward_delta_0 += self.coin_steal_penalty_delta
                
            elif p0_on_c and p1_on_c: # Contested coin
                
                full_reward_delta_0 += self.contested_coin_penalty_delta
                negative_reward_delta_0 += self.contested_coin_penalty_delta
                
                full_reward_delta_1 += self.contested_coin_penalty_delta
                negative_reward_delta_1 += self.contested_coin_penalty_delta
                
            return full_reward_delta_0, full_reward_delta_1, positive_reward_delta_0, positive_reward_delta_1, negative_reward_delta_0, negative_reward_delta_1 


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

    def step(self, actions: tuple) -> tuple:
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
                - A numpy array of cumulated rewards [full_reward_0, full_reward_1] - accounts for positive and negative reward and is used for decision making.
                - A numpy array of cumulated rewards [positive_reward_0, positive_reward_1] - accounts only for positive reward and is used purely for analysis.
                - A numpy array of cumulated rewards [negative_reward_0, negative_reward_1] - accounts only for negative reward and is used purely for analysis.
                - A done flag (bool) indicating if the episode has ended.
        """
        
        # Unpack actions
        action_0, action_1 = actions
        # Increment step count
        self.step_count += 1
        
        # Set reward for step
        full_reward_0, full_reward_1 = self.step_penalty, self.step_penalty
        
        positive_reward_0, positive_reward_1 = 0, 0
        
        negative_reward_0, negative_reward_1 = self.step_penalty, self.step_penalty
        
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
                
                full_reward_0 += self.both_push_penalty_delta
                negative_reward_0 += self.both_push_penalty_delta
                
                full_reward_1 += self.both_push_penalty_delta
                negative_reward_1 += self.both_push_penalty_delta
            else: 
                
                full_reward_0 += self.push_but_not_adjacent_penalty_delta
                negative_reward_0 += self.both_push_penalty_delta
                
                full_reward_1 += self.push_but_not_adjacent_penalty_delta
                negative_reward_1 += self.both_push_penalty_delta
                
        elif actual_push_0:  # Player 0 pushes
            if players_are_adjacent:
                push_direction = original_pos_1 - original_pos_0
                self.player_1_pos = np.clip(original_pos_1 + self.push_distance * push_direction, 0, self.grid_size - 1)
                
                full_reward_0 += self.push_reward_delta
                positive_reward_0 += self.push_reward_delta
                
                full_reward_1 += self.push_penalty_delta
                negative_reward_1 += self.push_penalty_delta
                
                pushed_1_this_step = True
            else:
                
                full_reward_0 += self.push_but_not_adjacent_penalty_delta
                negative_reward_0 += self.push_but_not_adjacent_penalty_delta
                
        elif actual_push_1:  # Player 1 pushes
            if players_are_adjacent:
                push_direction = original_pos_0 - original_pos_1
                self.player_0_pos = np.clip(original_pos_0 + self.push_distance * push_direction, 0, self.grid_size - 1)
                
                full_reward_1 += self.push_reward_delta
                positive_reward_1 += self.push_reward_delta
                
                full_reward_0 += self.push_penalty_delta
                negative_reward_0 += self.push_penalty_delta
                
                pushed_0_this_step = True
            else:
                full_reward_1 += self.push_but_not_adjacent_penalty_delta
                negative_reward_1 += self.push_but_not_adjacent_penalty_delta
        # --- Apply Movement ---
        
        new_pos_0 = self.player_0_pos.copy()
        new_pos_1 = self.player_1_pos.copy()

        if not pushed_0_this_step:
            # Calculate candidate position
            candidate_pos = original_pos_0 + self._deltas[actual_move_0]
            # Resolve out of bounds movement
            if np.any(candidate_pos < 0) or np.any(candidate_pos >= self.grid_size):
                
                full_reward_0 += self.out_of_bounds_penalty_delta
                negative_reward_0 += self.out_of_bounds_penalty_delta
                
            new_pos_0 = np.clip(candidate_pos, 0, self.grid_size - 1)
            
        if not pushed_1_this_step:
            # Calculate candidate position 
            candidate_pos = original_pos_1 + self._deltas[actual_move_1]
            # Resolve out of bounds movement
            if np.any(candidate_pos < 0) or np.any(candidate_pos >= self.grid_size):
                
                full_reward_1 += self.out_of_bounds_penalty_delta
                negative_reward_1 += self.out_of_bounds_penalty_delta
                
            new_pos_1 = np.clip(candidate_pos, 0, self.grid_size - 1)
        
        # Resolve collisions: if agents move to the same spot, they bounce back.
        if not np.array_equal(new_pos_0, new_pos_1):
            self.player_0_pos = new_pos_0
            self.player_1_pos = new_pos_1
        else:
            
            full_reward_0 += self.collision_penalty_delta
            negative_reward_0 += self.collision_penalty_delta
            
            full_reward_1 += self.collision_penalty_delta
            negative_reward_1 += self.collision_penalty_delta
        
        # --- Coin collection logic ---
        
        # Coin 1
        full_reward_delta_0, full_reward_delta_1, positive_reward_delta_0, positive_reward_delta_1, negative_reward_delta_0, negative_reward_delta_1  = self._resolve_coin_collection(0)
        
        full_reward_0 += full_reward_delta_0
        negative_reward_0 += negative_reward_delta_0
        positive_reward_0 += positive_reward_delta_0
        
        full_reward_1 += full_reward_delta_1
        negative_reward_1 += negative_reward_delta_1
        positive_reward_1 += positive_reward_delta_1

        # Coin 2
        full_reward_delta_0, full_reward_delta_1, positive_reward_delta_0, positive_reward_delta_1, negative_reward_delta_0, negative_reward_delta_1 = self._resolve_coin_collection(1)
        
        full_reward_0 += full_reward_delta_0
        negative_reward_0 += negative_reward_delta_0
        positive_reward_0 += positive_reward_delta_0
        
        full_reward_1 += full_reward_delta_1
        negative_reward_1 += negative_reward_delta_1
        positive_reward_1 += positive_reward_delta_1

        # --- Game End Conditions (overwrite step rewards) ---

        p0_wins = self.player_0_collected_coin0 and self.player_0_collected_coin1
        p1_wins = self.player_1_collected_coin0 and self.player_1_collected_coin1
        
        # Draw condition: both coins are gone, and no one has two.
        # NOTE: This includes scenarios where each player has one coin.
        is_draw = (not self.coin0_available and not self.coin1_available) and not p0_wins and not p1_wins

        if p0_wins:
            full_reward_0, full_reward_1, self.done = self.win_reward, self.loss_penalty, True
            positive_reward_0, negative_reward_1, self.done = self.win_reward, self.loss_penalty, True
        elif p1_wins:
            full_reward_0, full_reward_1, self.done = self.loss_penalty, self.win_reward, True
            positive_reward_1, negative_reward_0, self.done = self.win_reward, self.loss_penalty, True
        elif is_draw:
            full_reward_0, full_reward_1, self.done = self.draw_penalty, self.draw_penalty, True
            negative_reward_0, negative_reward_1, self.done = self.draw_penalty, self.draw_penalty, True
            
        # Check for timeout
        if not self.done and self.step_count >= self.max_steps:
            self.done = True
            p0_coins = int(self.player_0_collected_coin0) + int(self.player_0_collected_coin1)
            p1_coins = int(self.player_1_collected_coin0) + int(self.player_1_collected_coin1)
            if p0_coins > p1_coins:
                full_reward_0 += self.timeout_lead_bonus_delta
                positive_reward_0 += self.timeout_lead_bonus_delta
                
                full_reward_1 += self.timeout_trail_penalty_delta
                negative_reward_1 += self.timeout_trail_penalty_delta
                
            elif p1_coins > p0_coins:
                full_reward_1 += self.timeout_lead_bonus_delta
                positive_reward_1 += self.timeout_lead_bonus_delta
                
                full_reward_0 += self.timeout_trail_penalty_delta
                negative_reward_0 += self.timeout_trail_penalty_delta
                
            else:
                full_reward_0 += self.timeout_penalty_delta
                negative_reward_0 += self.timeout_penalty_delta
                
                full_reward_1 += self.timeout_penalty_delta
                negative_reward_1 += self.timeout_penalty_delta
                
        return self.get_state(), np.array([full_reward_0, full_reward_1]), np.array([positive_reward_0, positive_reward_1]), np.array([negative_reward_0, negative_reward_1]), self.done