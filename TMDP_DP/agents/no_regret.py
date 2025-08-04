import numpy as np
from numpy.random import choice
import copy
from tqdm.auto import tqdm
from typing import Optional

from .base import LearningAgent

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
ValueFunction = np.ndarray


class RegretMatching_TD_Agent(LearningAgent):
    """
    An advanced regret-matching agent that uses a learned state-value function
    (V(s)) to calculate non-myopic, one-step-lookahead regret. This gives it
    the foresight that the simple RegretMatchingAgent lacks.

    It maintains both a V-table for long-term value and a regret table for
    adaptive policy selection, making it a very strong and general baseline.

    Attributes:
        n_states (int): The total number of states in the environment.
        action_space (np.ndarray): The set of actions available to this agent.
        player_id (int): The agent's identifier (0, 1).
        cumulative_regrets (np.ndarray): A table of shape (n_states, num_actions)
                                         storing the regret for each state-action pair.
        r_lookup (np.ndarray): A pre-computed table mapping (s, a_self, a_opp) -> r.
                               This is used to get counterfactual rewards.
        learning_rate (float): Alpha, for the TD update of the V-function.
        V (ValueFunction): The agent's state-value function table, V(s).
        s_prime_lookup (np.ndarray): Table mapping (s, a_self, a_opp) -> s'.
        (Other attributes are similar to the simple RegretMatchingAgent)
    """
    def __init__(self, action_space: np.ndarray, opponent_action_space: np.ndarray,
                 n_states: int, player_id: int, env, gamma: float, learning_rate: float,
                 initial_V_value: float = 0.0):

        # --- Core Agent Parameters ---
        self.n_states = n_states
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.opponent_action_space = opponent_action_space
        self.num_opponent_actions = len(opponent_action_space)
        self.player_id = player_id
        self.gamma = gamma
        self.alpha = learning_rate

        # --- Learning Tables ---
        self.cumulative_regrets = np.zeros((self.n_states, self.num_actions))
        self.V = self._setup_value_function(initial_V_value)

        # --- Model Pre-computation ---
        self.env_snapshot = copy.deepcopy(env)
        self.env_snapshot.player_0_execution_prob = 1.0
        self.env_snapshot.player_1_execution_prob = 1.0
        
        # This agent needs both reward and next-state lookups.
        self.s_prime_lookup, self.r_lookup = self._precompute_lookups()

    def _setup_value_function(self, initial_V_value: float) -> ValueFunction:
        """ Initializes the state-value function V(s). """
        V = np.full(self.n_states, initial_V_value, dtype=float)
        for s in tqdm(range(self.n_states), desc="Initializing V-function for RegretMatching_TD"):
            if self._is_terminal_state(s):
                V[s] = 0
        return V

    def _is_terminal_state(self, obs: State) -> bool:
        """ Checks if a given state is terminal. """
        _, base_coll = self.env_snapshot.grid_size**2, 2
        state_copy = obs
        c_r2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll)
        p0_wins = c_b1 and c_b2
        p1_wins = c_r1 and c_r2
        coins_gone = (c_b1 or c_r1) and (c_b2 or c_r2)
        is_draw = coins_gone and not p0_wins and not p1_wins
        return p0_wins or p1_wins or is_draw

    def _precompute_lookups(self) -> tuple:
        """ Builds lookup tables for next states (s') and rewards (r). """
        s_prime_lookup = np.zeros((self.n_states, self.num_actions, self.num_opponent_actions), dtype=int)
        r_lookup = np.zeros((self.n_states, self.num_actions, self.num_opponent_actions), dtype=float)

        desc_str = f"Pre-computing lookups for RegretMatching_TD_Agent (Player {self.player_id})"
        for s in tqdm(range(self.n_states), desc=desc_str):
            try:
                self._reset_sim_env_to_state(s)
            except (IndexError, ValueError):
                continue

            for a_self_exec in range(self.num_actions):
                for a_opp_exec in range(self.num_opponent_actions):
                    current_env_state = self.env_snapshot.get_state()
                    action_pair = (a_self_exec, a_opp_exec) if self.player_id == 0 else (a_opp_exec, a_self_exec)

                    s_prime, rewards_vec, _ = self.env_snapshot.step(action_pair)
                    s_prime_lookup[s, a_self_exec, a_opp_exec] = s_prime
                    r_lookup[s, a_self_exec, a_opp_exec] = rewards_vec[self.player_id]
                    self._reset_sim_env_to_state(current_env_state)

        return s_prime_lookup, r_lookup

    def _reset_sim_env_to_state(self, obs: State):
        """ Resets the internal simulation environment to a specific state. """
        self.env_snapshot.reset()
        base_pos, base_coll = self.env_snapshot.grid_size**2, 2
        state_copy = obs
        c_r2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_r1 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b2 = bool(state_copy % base_coll); state_copy //= base_coll
        c_b1 = bool(state_copy % base_coll); state_copy //= base_coll
        p1_flat = state_copy % base_pos; state_copy //= base_pos
        p0_flat = state_copy
        self.env_snapshot.player_0_pos = np.array([p0_flat % self.env_snapshot.grid_size, p0_flat // self.env_snapshot.grid_size])
        self.env_snapshot.player_1_pos = np.array([p1_flat % self.env_snapshot.grid_size, p1_flat // self.env_snapshot.grid_size])
        self.env_snapshot.player_0_collected_coin0, self.env_snapshot.player_0_collected_coin1 = c_b1, c_b2
        self.env_snapshot.player_1_collected_coin0, self.env_snapshot.player_1_collected_coin1 = c_r1, c_r2
        self.env_snapshot.coin0_available = not (c_b1 or c_r1)
        self.env_snapshot.coin1_available = not (c_b2 or c_r2)

    def get_policy(self, obs: State) -> Policy:
        """ Computes the policy for a given state based on positive cumulative regrets. """
        regrets = self.cumulative_regrets[obs]
        positive_regrets = np.maximum(0, regrets)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            return positive_regrets / regret_sum
        else:
            return np.ones(self.num_actions) / self.num_actions

    def act(self, obs: State, env=None) -> Action:
        """ Selects an action by sampling from the regret-based policy. """
        policy = self.get_policy(obs)
        return choice(self.action_space, p=policy)

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards: Optional[tuple[Reward, Reward]]):
        """
        Performs a two-part update:
        1. Updates the state-value function V(s) via Temporal Difference learning.
        2. Updates the cumulative regrets using the new V(s) to provide foresight.
        """
        if rewards is None:
            raise ValueError("RegretMatching_TD_Agent requires rewards for its update.")

        if self.player_id == 0:
            _, opponent_action = actions
            self_reward, _ = rewards
        else:
            opponent_action, _ = actions
            _, self_reward = rewards

        # --- Part 1: Update the V-function using TD(0) ---
        if not self._is_terminal_state(obs):
            td_target = self_reward + self.gamma * self.V[new_obs]
            self.V[obs] = (1 - self.alpha) * self.V[obs] + self.alpha * td_target

        # --- Part 2: Update Cumulative Regrets with Foresight ---
        
        # 1. Calculate the utility of the action that was actually taken.
        # This utility is non-myopic: r + gamma * V(s').
        actual_utility = self_reward + self.gamma * self.V[new_obs]
        
        # 2. Calculate counterfactual utilities for all other actions.
        for a_i in self.action_space:
            # Find what the immediate reward and next state WOULD have been.
            counterfactual_r = self.r_lookup[obs, a_i, opponent_action]
            counterfactual_s_prime = self.s_prime_lookup[obs, a_i, opponent_action]
            
            # Calculate the utility of this counterfactual action.
            counterfactual_utility = counterfactual_r + self.gamma * self.V[counterfactual_s_prime]
            
            # Update regret for action a_i.
            regret = counterfactual_utility - actual_utility
            self.cumulative_regrets[obs, a_i] += regret

    def update_epsilon(self, new_epsilon_agent: float, new_epsilon_lower_k_level: Optional[float]):
        """ This agent does not use epsilon-greedy exploration. """
        pass