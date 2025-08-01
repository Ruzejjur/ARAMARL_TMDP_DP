# --- START OF FILE agents/nash_agent.py ---

import numpy as np
from numpy.random import choice
from tqdm.notebook import tqdm
import os
import logging
from typing import Optional

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
ValueFunction = np.ndarray

# Import the base agent class and the agents module to dynamically create trainers
from .base import BaseAgent
import agents

class NashEquilibriumAgent(BaseAgent):
    """
    An offline agent that plays an approximate Nash Equilibrium strategy.

    This agent's initialization is a 'train-on-demand' process. It checks for a
    pre-computed policy file. If the file doesn't exist, it launches a full
    self-play training session using a specified no-regret agent (e.g.,
    RegretMatching_TD_Agent) to generate and save the policy. If the file
    exists, it simply loads it.
    """
    def __init__(self, action_space: np.ndarray, n_states: int, env,
                 policy_path: str, training_episodes: int, training_agent: dict):
        """
        Initializes the agent by either loading or generating the Nash policy.

        Args:
            action_space (np.ndarray): The set of available actions.
            n_states (int): The total number of states in the environment.
            env: The environment instance, needed for training.
            policy_path (str): The file path to save/load the .npy policy table.
            training_episodes (int): The number of self-play episodes to run if training.
            training_agent (dict): A config dictionary specifying the 'class' and
                                   'params' of the agent to use for self-play training.
        """
        self.action_space = action_space
        self.policy_path = policy_path

        if os.path.exists(self.policy_path):
            logging.info(f"Found existing Nash policy. Loading from: {self.policy_path}")
            self.policy_table = np.load(self.policy_path)
        else:
            logging.warning(
                f"No Nash policy found at '{self.policy_path}'. "
                "Starting self-play training process. This may take a long time."
            )
            # Trigger the training process
            self.policy_table = self._train_and_generate_policy(
                n_states, env, training_episodes, training_agent
            )

    def _train_and_generate_policy(self, n_states, env, episodes, agent_config):
        """ The self-play training loop to generate the Nash policy. """
        
        # --- 1. Instantiate the trainer agents ---
        # Dynamically create the agent specified in the config
        try:
            TrainerAgentClass = getattr(agents, agent_config['class'])
        except AttributeError:
            raise ValueError(f"Training agent class '{agent_config['class']}' not found.")

        common_params = {
            'n_states': n_states,
            'action_space': self.action_space,
            'opponent_action_space': self.action_space, # Playing against itself
            'env': env
        }

        p1_trainer_params = {**common_params, **agent_config['params'], 'player_id': 0}
        p2_trainer_params = {**common_params, **agent_config['params'], 'player_id': 1}
        
        p1 = TrainerAgentClass(**p1_trainer_params)
        p2 = TrainerAgentClass(**p2_trainer_params)

        # --- 2. Run the self-play loop ---
        policy_sum_table = np.zeros((n_states, len(self.action_space)))
        
        for episode in tqdm(range(episodes), desc="Nash Equilibrium Self-Play Training"):
            env.reset()
            obs = env.get_state()
            done = False
            
            while not done:
                # Accumulate Player 1's policy at each step
                policy_sum_table[obs] += p1.get_policy(obs)
                
                # Agents act and update
                a1 = p1.act(obs, env)
                a2 = p2.act(obs, env)
                s_new, rewards, done = env.step((a1, a2))
                
                p1.update(obs, (a1, a2), s_new, rewards)
                p2.update(obs, (a1, a2), s_new, rewards)
                
                obs = s_new
        
        # --- 3. Normalize and save the policy ---
        logging.info("Self-play training finished. Normalizing and saving policy...")
        
        state_sums = np.sum(policy_sum_table, axis=1, keepdims=True)
        num_actions = len(self.action_space)
        
        # For unvisited states, default to a uniform random policy
        default_policy = np.full_like(policy_sum_table, 1.0 / num_actions)
        
        nash_policy_table = np.divide(
            policy_sum_table, state_sums, out=default_policy, where=state_sums != 0
        )
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.policy_path), exist_ok=True)
        np.save(self.policy_path, nash_policy_table)
        logging.info(f"Nash policy successfully saved to: {self.policy_path}")
        
        return nash_policy_table

    def act(self, obs: State, env=None) -> Action:
        """ Selects an action by sampling from the pre-computed policy. """
        policy_for_obs = self.policy_table[obs]
        return choice(self.action_space, p=policy_for_obs)

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards: Optional[tuple[Reward, Reward]]):
        pass # This agent does not learn online.
    
    def update_epsilon(self, new_epsilon_agent: float, new_epsilon_lower_k_level: Optional[float]):
        pass # This agent does not use epsilon.