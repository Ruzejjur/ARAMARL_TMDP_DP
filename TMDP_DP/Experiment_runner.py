"""
This module serves as the main entry point for running reinforcement learning
experiments. It orchestrates the entire process, from loading configurations
to initializing agents and the environment, running the simulation, and finally
logging and plotting the results.

The runner is designed to be flexible, using a YAML configuration file to define
all experimental parameters. This allows for easy modification of agent types,
hyperparameters, and environment settings without changing the core code.
"""

import yaml
import numpy as np
from tqdm.auto import tqdm
import os
import logging 
import inspect
from datetime import datetime
import shutil
import sys
from pathlib import Path 
from typing import Union, Optional

from utils.trajectory_log_schema import TRAJECTORY_LOG_COLUMN_MAP


# Import your custom modules
from engine_DP import CoinGame
import agents
from utils.exploration_schedule_utils import linear_epsilon_decay
from utils.plot_utils import plot_reward_per_episode_series, plot_result_ration

# Directing log messages of level INFO and higher to the console.
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     stream=sys.stdout  # Explicitly direct logs to standard output
# )

def load_config(config_file_path: str) -> dict:
    """
    Loads and returns the YAML configuration file.

    This file contains all the necessary settings for the agents and the
    environment.

    Args:
        config_file_path (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration settings.
    """
    
    
    logging.info(f"Loading configuration from: {config_file_path}")
    
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)
    
    logging.info("Configuration loaded successfully. Starting experiment...")
    
def add_file_handler_to_logger(results_path: str):
    """Adds a file handler to the root logger to save logs to a specified directory."""
    log_filename = os.path.join(results_path, 'experiment_log.log')
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Create a formatter (so the file logs match the console logs)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    logger.addHandler(file_handler)
    logging.info(f"Logging is now also being saved to: {log_filename}")
    
def create_agent(agent_config: dict, common_params: dict) -> agents.BaseAgent:
    """
    Factory function to create an agent instance from its configuration.

    This function dynamically instantiates an agent class based on the name
    provided in the config. It uses Python's `inspect` module to ensure that
    all required parameters for the agent's constructor are present in the
    configuration, raising helpful errors if they are not.

    Args:
        agent_config (dict): A dictionary slice from the main config containing
                             the specific agent's 'class' and 'params'.
        common_params (dict): A dictionary of parameters that are common to all
                              agents (e.g., n_states, action_space).

    Returns:
        An instance of the specified agent class.

    Raises:
        ValueError: If the agent class specified in the config is not found in
                    the `agents` module.
        TypeError: If the configuration is missing required parameters for the
                   agent's constructor.
    """
    agent_class_name = agent_config['class']
    agent_params = agent_config.get('params', {})
    
    # Combine common parameters (like n_states) with agent-specific ones
    full_params = {**common_params, **agent_params}
    
    # Dynamically get the agent class from the `agents` module
    try:
        AgentClass = getattr(agents, agent_class_name)
    except AttributeError:
        raise ValueError(f"Agent class '{agent_class_name}' not found in the 'agents' module.")
    
    # --- Parameter Validation using 'inspect' ---
    # This ensures that the configuration provides all necessary arguments for the agent.
    sig = inspect.signature(AgentClass.__init__)
    
    # Find all parameters in the agent's constructor that DO NOT have a default value.
    required_params = {
        param.name for param in sig.parameters.values()
        if param.name != 'self' and param.default == inspect.Parameter.empty
    }
            
    # Check if any required parameters are missing from our configuration.
    provided_params = set(full_params.keys())
    missing_params = required_params - provided_params
    
    if missing_params:
        # If parameters are missing, raise a specific and helpful error.
        raise TypeError(
            f"Error creating agent '{agent_class_name}'. "
            f"The configuration is missing the following required parameters: {list(missing_params)}. "
            "Please add them to the 'params' section in your config*.yaml for this agent."
        )
        
    # Filter the provided parameters to only include those expected by the constructor.
    # This prevents errors from passing unexpected arguments (e.g., 'epsilon_decay_agent').
    valid_params = {k: v for k, v in full_params.items() if k in sig.parameters}
    
    return AgentClass(**valid_params)

def run_single_episode(env: CoinGame, p0: agents.BaseAgent, p1: agents.BaseAgent, experiment_num: int, episode_num: int, log_trajectory: bool = False) -> tuple:
    """
    Runs a single episode of the game from start to finish.

    Args:
        env (CoinGame): The game environment instance.
        p0 (BaseAgent): The agent instance for player 1.
        p1 (BaseAgent): The agent instance for player 2.
        experiment_num (int): The current experiment run number (for logging).
        episode_num (int): The current episode number (for logging).
        log_trajectory (bool): If True, records a detailed log of each step for
                               later visualization.

    Returns:
        A tuple containing:
        - episode_cumulative_full_reward_p0 (float): Total reward for player 1 in the episode.
        - episode_cumulative_full_reward_p1 (float): Total reward for player 2 in the episode.
        - episode_cumulative_positive_reward_p0 (float): Only positive rewards for player 1 in the episode.
        - episode_cumulative_positive_reward_p1 (float): Only positive rewards for player 2 in the episode.
        - episode_cumulative_negative_reward_p0 (float): Only negative rewards for player 1 in the episode.
        - episode_cumulative_negative_reward_p1 (float): Only negative rewards for player 2 in the episode.
        - episode_cumulative_only_step_reward_p0 (float): Only step rewards for player 1 in the episode.
        - episode_cumulative_only_step_reward_p1 (float): Only step rewards for player 2 in the episode.
        - episode_cumulative_full_reward_without_coin_p0 (float): Total reward without coin for player 1 in the episode.
        - episode_cumulative_full_reward_without_coin_p1 (float): Total reward without coin for player 2 in the episode.
        - trajectory_log (list): A log of states, actions, and rewards, if enabled.
    """
    env.reset()
    obs = env.get_state()
    done = False
    
    episode_cumulative_full_reward_p0 = 0
    episode_cumulative_full_reward_p1 = 0

    episode_cumulative_positive_reward_p0 = 0
    episode_cumulative_positive_reward_p1 = 0
    
    episode_cumulative_negative_reward_p0 = 0
    episode_cumulative_negative_reward_p1 = 0
    
    episode_cumulative_only_step_reward_p0 = 0
    episode_cumulative_only_step_reward_p1 = 0
    
    episode_cumulative_full_reward_without_coin_p0 = 0
    episode_cumulative_full_reward_without_coin_p1 = 0
    
    episode_cumulative_full_reward_without_step_p0 = 0
    episode_cumulative_full_reward_without_step_p1 = 0
    
    trajectory_log = []
    
    # Initialising variable for result of the game from perspective of player 1
    result_p0 = None
    # Initialising step counter
    step_num = 0
         
    if log_trajectory:
        
        # Use -1 as a placeholder for unavailable data
        p0_pos = env.player_0_pos
        p1_pos = env.player_1_pos
        coin1_pos = env.coin_0_pos if env.coin0_available else [-1, -1]
        coin2_pos = env.coin_1_pos if env.coin1_available else [-1, -1]
        
        # Log step details if enabled.
        if log_trajectory:
            # Pre-allocate a numpy array for the initial state row
            initial_row = np.full(len(TRAJECTORY_LOG_COLUMN_MAP), -1.0) # Use -1 as placeholder for None values
            
            p0_pos = env.player_0_pos
            p1_pos = env.player_1_pos
            coin1_pos = env.coin_0_pos if env.coin0_available else [-1, -1]
            coin2_pos = env.coin_1_pos if env.coin1_available else [-1, -1]
            #p0_action = -1.0 # Stays initialised to -1.0  
            #p1_action = -1.0 # Stays initialised to -1.0 

            initial_row[TRAJECTORY_LOG_COLUMN_MAP['experiment_num']] = experiment_num
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['episode_num']] = episode_num
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['step_num']] = step_num
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_old_row']] = -1.0 # Stays initialised to -1.0 
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_old_col']] = -1.0 # Stays initialised to -1.0      
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_new_row']] = p0_pos[0]
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_new_col']] = p0_pos[1]
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['p1_loc_new_row']] = p1_pos[0]
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['p1_loc_new_col']] = p1_pos[1]
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['coin1_row']] = coin1_pos[0]
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['coin1_col']] = coin1_pos[1]
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['coin2_row']] = coin2_pos[0]
            initial_row[TRAJECTORY_LOG_COLUMN_MAP['coin2_col']] = coin2_pos[1]
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_action_move']] = -1.0 # Stays initialised to -1
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_action_push']] = -1.0 # Stays initialised to -1
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p1_action_move']] = -1.0 # Stays initialised to -1.0 
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p1_action_push']] = -1.0 # Stays initialised to -1.0 
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_reward']] = -1.0 # Stays initialised to -1.0 
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p1_reward']] = -1.0 # Stays initialised to -1.0 
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p0_cum_reward']] = -1.0 # Stays initialised to -1.0 
            # initial_row[TRAJECTORY_LOG_COLUMN_MAP['p1_cum_reward']] = -1.0 # Stays initialised to -1.0 
            
            trajectory_log.append(initial_row)
            
    while not done:
        
        step_num += 1
        # Get actions from both agents.
        a1 = p0.act(obs=obs, env=env)
        a2 = p1.act(obs=obs, env=env)
        
        # Store current locations for logging before the step.
        p0_loc_old, p1_loc_old = env.player_0_pos.copy(), env.player_1_pos.copy()

        # Execute actions in the environment.
        (s_new, full_rewards, positive_rewards, negative_rewards,
        only_step_rewards,full_rewards_without_coin, full_rewards_without_step,
        result_p0, done) = env.step((a1, a2))
        
        # Allow learning agents to update their internal models/policies.
        if isinstance(p0, agents.LearningAgent):
            p0.update(obs=obs, actions=(a1, a2), new_obs=s_new, rewards=(full_rewards[0], full_rewards[1]))
        if isinstance(p1, agents.LearningAgent):
            p1.update(obs=obs, actions=(a1, a2), new_obs=s_new, rewards=(full_rewards[0], full_rewards[1]))
        
        # Advance the state
        obs = s_new
        
        # Accumulate rewards.   
        episode_cumulative_full_reward_p0 += full_rewards[0]
        episode_cumulative_full_reward_p1 += full_rewards[1]
        
        episode_cumulative_positive_reward_p0 += positive_rewards[0]
        episode_cumulative_positive_reward_p1 += positive_rewards[1]
        
        episode_cumulative_negative_reward_p0 += negative_rewards[0]
        episode_cumulative_negative_reward_p1 += negative_rewards[1]
        
        episode_cumulative_only_step_reward_p0 += only_step_rewards[0]
        episode_cumulative_only_step_reward_p1 += only_step_rewards[1]
        
        episode_cumulative_full_reward_without_coin_p0 += full_rewards_without_coin[0]
        episode_cumulative_full_reward_without_coin_p1 += full_rewards_without_coin[1]
        
        episode_cumulative_full_reward_without_step_p0 += full_rewards_without_step[0]
        episode_cumulative_full_reward_without_step_p1 += full_rewards_without_step[1]
        
        if log_trajectory:
            log_row = np.full(len(TRAJECTORY_LOG_COLUMN_MAP), -1.0) # Pre-allocate
            p0_pos = env.player_0_pos
            p1_pos = env.player_1_pos
            coin1_pos = env.coin_0_pos if env.coin0_available else [-1, -1]
            coin2_pos = env.coin_1_pos if env.coin1_available else [-1, -1]
            p0_action = env.combined_actions[a1]
            p1_action = env.combined_actions[a2]

            log_row[TRAJECTORY_LOG_COLUMN_MAP['experiment_num']] = experiment_num
            log_row[TRAJECTORY_LOG_COLUMN_MAP['episode_num']] = episode_num
            log_row[TRAJECTORY_LOG_COLUMN_MAP['step_num']] = step_num
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_old_row']] = p0_loc_old[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_old_col']] = p0_loc_old[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_loc_old_row']] = p1_loc_old[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_loc_old_col']] = p1_loc_old[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_new_row']] = p0_pos[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_loc_new_col']] = p0_pos[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_loc_new_row']] = p1_pos[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_loc_new_col']] = p1_pos[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['coin1_row']] = coin1_pos[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['coin1_col']] = coin1_pos[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['coin2_row']] = coin2_pos[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['coin2_col']] = coin2_pos[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_action_move']] = p0_action[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_action_push']] = p0_action[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_action_move']] = p1_action[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_action_push']] = p1_action[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_reward']] = full_rewards[0]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_reward']] = full_rewards[1]
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p0_cum_reward']] = episode_cumulative_full_reward_p0
            log_row[TRAJECTORY_LOG_COLUMN_MAP['p1_cum_reward']] = episode_cumulative_full_reward_p1
            
            trajectory_log.append(log_row)
            
    assert result_p0 is not None, "result_p0 cannot be None. The episode should return a result for player 0 value of [-2,-1,0,1] after its finished."
        
    episode_result_p0 = result_p0
        
    return (episode_cumulative_full_reward_p0, episode_cumulative_full_reward_p1,
            episode_cumulative_positive_reward_p0, episode_cumulative_positive_reward_p1,
            episode_cumulative_negative_reward_p0, episode_cumulative_negative_reward_p1,
            episode_cumulative_only_step_reward_p0, episode_cumulative_only_step_reward_p1,
            episode_cumulative_full_reward_without_coin_p0, episode_cumulative_full_reward_without_coin_p1,
            episode_cumulative_full_reward_without_step_p0, episode_cumulative_full_reward_without_step_p1,
            episode_result_p0,
            trajectory_log)

def run_experiment(config_file_path: str, base_output_dir: Optional[str] = None, log_trajectory: bool = False) -> str:
    """
    Main function to run the entire set of experiments based on the config.

    This function handles the main experiment loop, including setting up
    logging directories, initializing the environment and agents for each run,
    looping through episodes, and saving the final results and plots.

    Args:
        config_file_path (str): Path to the YAML experiment configuration file.
        log_trajectory (bool): If True, a detailed log is saved.
        base_output_dir (Path, optional): The base directory where the results
            folder for this specific run will be created. If None, it falls
            back to the path specified in the config file. Defaults to None.


    Returns:
        str: The path to the directory where results for this run were saved.
    """
    
    # --- Setup ---
    # Load full experiment configuration
    config = load_config(config_file_path=config_file_path)
    
    exp_settings = config['experiment_settings']
    env_settings = config['environment_settings']
    agent_configs = config['agent_settings']
    
    # Create a unique, timestamped directory for this experiment's results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = exp_settings.get('name', 'experiment')
    run_dir_name = f"{experiment_name}_{timestamp}"
    
    # Determine the root directory for results.
    if base_output_dir:
        # If a base directory is provided by the parallel runner, USE IT.
        root_results_dir = base_output_dir
    else:
        # This is the FALLBACK for running this script directly.
        # It reads the base path from the config file.
        logging.warning(
            "No base_output_dir provided. Falling back to 'results_dir' from config file."
        )
        root_results_dir = Path(exp_settings['results_dir'])

    # The final, unique path for this specific experiment's results.
    results_path = Path(root_results_dir) / run_dir_name
    os.makedirs(results_path, exist_ok=True)
    
    # Setup logging
    add_file_handler_to_logger(str(results_path))
    
    logging.info("Results for this run will be saved in: %s", results_path)
    
    # Initialize the single environment instance.
    env = CoinGame(**env_settings['params'])
    n_episodes = exp_settings['num_episodes']
    
    # --- Conditional Epsilon Decay Schedule Setup ---
    # Initialize all schedules to None. They will only be created if the
    # corresponding configuration section exists for the agent.

    epsilon_agent_schedule_p0 = None
    epsilon_agent_schedule_p1 = None
    epsilon_lower_k_level_schedule_p0 = None
    epsilon_lower_k_level_schedule_p1 = None

    # Player 1 Schedules
    if 'epsilon_decay_agent' in agent_configs['player_1']:
        try:
            p0_decay_config = agent_configs['player_1']['epsilon_decay_agent']
            p0_params_config = agent_configs['player_1']['params']
            if p0_decay_config.get('type') == 'linear':
                epsilon_agent_schedule_p0 = linear_epsilon_decay(
                    p0_params_config['epsilon'], p0_decay_config['end'], n_episodes
                )
            elif p0_decay_config.get('type') == 'no_decay':
                epsilon_agent_schedule_p0 = np.full(n_episodes, p0_params_config['epsilon'])
        except KeyError as e:
            raise KeyError(f"Missing key {e} in 'epsilon_decay_agent' or 'params' for player_1.")

    if 'epsilon_decay_internal_opponent_model' in agent_configs['player_1']:
        try:
            p0_internal_decay_config = agent_configs['player_1']['epsilon_decay_internal_opponent_model']
            p0_params_config = agent_configs['player_1']['params']
            if p0_internal_decay_config.get('type') == 'linear':
                epsilon_lower_k_level_schedule_p0 = linear_epsilon_decay(
                    p0_params_config['lower_level_k_epsilon'], p0_internal_decay_config['end'], n_episodes
                )
            elif p0_internal_decay_config.get('type') == 'no_decay':
                epsilon_lower_k_level_schedule_p0 = epsilon_agent_schedule_p0 = np.full(n_episodes, p0_params_config['lower_level_k_epsilon'])
        except KeyError as e:
            raise KeyError(f"Missing key {e} in 'epsilon_decay_internal_opponent_model' or 'params' for player_1.")

    # Player 2 Schedules
    if 'epsilon_decay_agent' in agent_configs['player_2']:
        try:
            p1_decay_config = agent_configs['player_2']['epsilon_decay_agent']
            p1_params_config = agent_configs['player_2']['params']
            if p1_decay_config.get('type') == 'linear':
                epsilon_agent_schedule_p1 = linear_epsilon_decay(
                    p1_params_config['epsilon'], p1_decay_config['end'], n_episodes
                )
            elif p1_decay_config.get('type') == 'no_decay':
                epsilon_agent_schedule_p1 = np.full(n_episodes, p1_params_config['epsilon'])
        except KeyError as e:
            raise KeyError(f"Missing key {e} in 'epsilon_decay_agent' or 'params' for player_2.")

    if 'epsilon_decay_internal_opponent_model' in agent_configs['player_2']:
        try:
            p1_internal_decay_config = agent_configs['player_2']['epsilon_decay_internal_opponent_model']
            p1_params_config = agent_configs['player_2']['params']
            if p1_internal_decay_config.get('type') == 'linear':
                epsilon_lower_k_level_schedule_p1 = linear_epsilon_decay(
                    p1_params_config['lower_level_k_epsilon'], p1_internal_decay_config['end'], n_episodes
                )
            elif p1_internal_decay_config.get('type') == 'no_decay':
                epsilon_lower_k_level_schedule_p1 = epsilon_agent_schedule_p1 = np.full(n_episodes, p1_params_config['lower_level_k_epsilon'])
        except KeyError as e:
            raise KeyError(f"Missing key {e} in 'epsilon_decay_internal_opponent_model' or 'params' for player_2.")
    
    # --- Data Logging Initialisation ---
    
    # Reward log list initalisation
    full_rewards_p0 = []
    full_rewards_p1 = []
    
    positive_rewards_p0 = []
    positive_rewards_p1 = []
    
    negative_rewards_p0 = []
    negative_rewards_p1 = []
    
    only_step_rewards_p0 = []
    only_step_rewards_p1 = []
    
    full_rewards_without_coin_p0 = []
    full_rewards_without_coin_p1 = []
    
    full_rewards_without_step_p0 = []
    full_rewards_without_step_p1 = []
    
    # Trajectory log initialisation
    trajectory_logs_all_experiments = []
    
    # Win, loss and draw array for player 1 initalisation
    game_result_p0 = []

    # --- Experiment Loop (for multiple independent runs) ---
    for experiment_num in tqdm(range(exp_settings['num_runs']), desc="Experiment runs"):
        # Check for run_seed
        seed = exp_settings.get('run_seed')
        if seed is not None:
            # Add experiment_num to the base seed for different, but reproducible runs
            current_seed = seed + experiment_num
            np.random.seed(current_seed)
            
            logging.info(
                "Running experiment %d/%d with seed: %d",
                experiment_num + 1,
                exp_settings['num_runs'],
                current_seed
            )
        else:
            # Using logging.warning is also better for consistency.
            logging.warning(
                "'run_seed' not found in config. Experiment %d/%d will run with a non-reproducible random seed.",
                experiment_num + 1,
                exp_settings['num_runs']
            )

        # Trajectory log for this experiment
        trajectory_log_single_experiment = []
        
        # --- Agent Initialization ---
            
        if env_settings['params']['enable_push']:
            action_space_to_use = np.arange(len(env.combined_actions))
            logging.info("Experiment running with MOVE and PUSH actions.")
        else:
            # The first 4 actions in combined_actions are the move-only ones.
            action_space_to_use = np.arange(4) 
            logging.info("Experiment running with MOVE-ONLY actions (PUSH is disabled).")
        
        common_agent_params = {
            'n_states': env.n_states,
            'action_space': action_space_to_use,
            'opponent_action_space': action_space_to_use,
            'grid_size': env_settings['params']['grid_size'],
            'env': env 
        }
        
        p0_params = {'class': agent_configs['player_1']['class'],
            'params': {**agent_configs['player_1'].get('params', {})
            ,'player_id': 0}}
        
        p1_params = {'class': agent_configs['player_2']['class'],
                    'params': {**agent_configs['player_2'].get('params', {}),
                    'player_id': 1}}
        
        # Add coin location if agents is of type ManhattanAgent
        if "ManhattanAgent" in p0_params['class']:
            p0_params['params'].update({'coin_location': np.array([env.coin_0_pos,env.coin_1_pos])})
        if "ManhattanAgent" in p1_params['class']:
            p1_params['params'].update({'coin_location': np.array([env.coin_0_pos,env.coin_1_pos])})
        
        p0 = None
        p1 = None

        # Check if either agent is an offline solver
        offline_solver_classes = ['MDP_DP_Agent_PerfectModel', 'TMDP_DP_Agent_PerfectModel']
        p0_is_offline_solver = agent_configs['player_1']['class'] in offline_solver_classes
        p1_is_offline_solver = agent_configs['player_2']['class'] in offline_solver_classes
        
        if not p0_is_offline_solver and not p1_is_offline_solver:
            p0 = create_agent(p0_params, common_agent_params)
            
            p1 = create_agent(p1_params, common_agent_params)
        
        # If one of the agents is of class DPAgent_PerfectModel
        else:
            if p0_is_offline_solver:

                p1 = create_agent(p1_params, common_agent_params)
            
                p0_params['params'].update({'opponent': p1})
                
                p0 = create_agent(p0_params, common_agent_params)
                
            elif p1_is_offline_solver:
                
                p0 = create_agent(p0_params, common_agent_params)
                
                p1_params['params'].update({'opponent': p0})
                
                p1 = create_agent(p1_params, common_agent_params)
        
        assert isinstance(p0, agents.BaseAgent) and isinstance(p1, agents.BaseAgent), "Agents must always be of type BaseAgent."
        
        # Single experiment rewards initalisation
        run_full_rewards_p0 = []
        run_full_rewards_p1 = []
        
        run_positive_rewards_p0 = []
        run_positive_rewards_p1 = []
        
        run_negative_rewards_p0 = []
        run_negative_rewards_p1 = []
        
        run_only_step_rewards_p0 = []
        run_only_step_rewards_p1 = []
        
        run_full_rewards_without_coin_p0 = []
        run_full_rewards_without_coin_p1 = []
        
        run_full_rewards_without_step_p0 = []
        run_full_rewards_without_step_p1 = []
        
        run_game_result_p0 = []
        
        # --- Episode Loop ---
        for episode_num in tqdm(range(n_episodes), desc=f"Epoch for experiment {experiment_num+1}", leave=False):
            
            (episode_cumulative_full_reward_p0, episode_cumulative_full_reward_p1,
            cumulative_positive_episode_reward_p0, cumulative_positive_episode_reward_p1,
            cumulative_negative_episode_reward_p0, cumulative_negative_episode_reward_p1,
            cumulative_only_step_reward_p0, cumulative_only_step_reward_p1,
            cumulative_full_reward_without_coin_p0, cumulative_full_reward_without_coin_p1,
            cumulative_full_reward_without_step_p0, cumulative_full_reward_without_step_p1,
            episode_game_result_p0,
            trajectory) = run_single_episode(env, p0, p1, experiment_num, episode_num, log_trajectory=log_trajectory)
            
            # Append the cumulative reward to array of cumulative results
            run_full_rewards_p0.append(episode_cumulative_full_reward_p0)
            run_full_rewards_p1.append(episode_cumulative_full_reward_p1)
            
            run_positive_rewards_p0.append(cumulative_positive_episode_reward_p0)
            run_positive_rewards_p1.append(cumulative_positive_episode_reward_p1)
            
            run_negative_rewards_p0.append(cumulative_negative_episode_reward_p0)
            run_negative_rewards_p1.append(cumulative_negative_episode_reward_p1)
            
            run_only_step_rewards_p0.append(cumulative_only_step_reward_p0)
            run_only_step_rewards_p1.append(cumulative_only_step_reward_p1)
            
            run_full_rewards_without_coin_p0.append(cumulative_full_reward_without_coin_p0)
            run_full_rewards_without_coin_p1.append(cumulative_full_reward_without_coin_p1)
            
            run_full_rewards_without_step_p0.append(cumulative_full_reward_without_step_p0)
            run_full_rewards_without_step_p1.append(cumulative_full_reward_without_step_p1)
            
            # Append the trajectory log for this epoch to experiment log 
            if log_trajectory:
                trajectory_log_single_experiment.extend(trajectory)
            
            # Append game result of the player 1 to game result player1 log 
            run_game_result_p0.append(episode_game_result_p0)
            
            #Update epsilon for agent and possible lower k-level models
            
            # Player 1
            if isinstance(p0, agents.LearningAgent):
                new_epsilon_agent_p0 = agent_configs['player_1']['params']['epsilon'] if not p0_is_offline_solver else 1 # Default to initial if appropriate 
                new_epsilon_lower_k_p0 = None      # Default to None
                
                if epsilon_agent_schedule_p0 is not None:
                    new_epsilon_agent_p0 = epsilon_agent_schedule_p0[episode_num]
                if epsilon_lower_k_level_schedule_p0 is not None:
                    new_epsilon_lower_k_p0 = epsilon_lower_k_level_schedule_p0[episode_num]
                    
                p0.update_epsilon(new_epsilon_agent_p0, new_epsilon_lower_k_p0)

            # Player 2
            if isinstance(p1, agents.LearningAgent):
                new_epsilon_agent_p1 = agent_configs['player_2']['params']['epsilon'] if not p1_is_offline_solver else 1 # Default to initial if appropriate 
                new_epsilon_lower_k_p1 = None

                if epsilon_agent_schedule_p1 is not None:
                    new_epsilon_agent_p1 = epsilon_agent_schedule_p1[episode_num]
                if epsilon_lower_k_level_schedule_p1 is not None:
                    new_epsilon_lower_k_p1 = epsilon_lower_k_level_schedule_p1[episode_num]

                p1.update_epsilon(new_epsilon_agent_p1, new_epsilon_lower_k_p1)
            
        if log_trajectory:
            # Append the trajectory logs of this experiment to the total logs
            trajectory_logs_all_experiments.extend(trajectory_log_single_experiment)
        
        full_rewards_p0.append(run_full_rewards_p0)
        full_rewards_p1.append(run_full_rewards_p1)
        
        positive_rewards_p0.append(run_positive_rewards_p0)
        positive_rewards_p1.append(run_positive_rewards_p1)
    
        negative_rewards_p0.append(run_negative_rewards_p0)
        negative_rewards_p1.append(run_negative_rewards_p1)
        
        only_step_rewards_p0.append(run_only_step_rewards_p0)
        only_step_rewards_p1.append(run_only_step_rewards_p1)
        
        full_rewards_without_coin_p0.append(run_full_rewards_without_coin_p0)
        full_rewards_without_coin_p1.append(run_full_rewards_without_coin_p1)
        
        full_rewards_without_step_p0.append(run_full_rewards_without_step_p0)
        full_rewards_without_step_p1.append(run_full_rewards_without_step_p1)
        
        game_result_p0.append(run_game_result_p0)

    # --- Plotting and Saving results ---
    
    # Plotting cumulative full rewards 
    plot_name = experiment_name + '_full_rewards'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Cumulative full rewards (positive + negative) per episode'
    
    plot_reward_per_episode_series(full_rewards_p0, full_rewards_p1, plot_title,
         moving_average_window_size=config['plotting_settings']['plot_moving_average_window_size'],
         episode_series_x_axis_plot_range=config['plotting_settings']['reward_time_series_x_axis_plot_range'], 
         dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting cumulative positive rewards 
    plot_name = experiment_name + '_positive_rewards'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Cumulative positive rewards per episode'
    
    plot_reward_per_episode_series(positive_rewards_p0, positive_rewards_p1, plot_title,
         moving_average_window_size=config['plotting_settings']['plot_moving_average_window_size'],
         episode_series_x_axis_plot_range=config['plotting_settings']['reward_time_series_x_axis_plot_range'],  
         dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting cumulative negative rewards 
    plot_name = experiment_name + '_negative_rewards'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Cumulative negative rewards per episode'
    
    plot_reward_per_episode_series(negative_rewards_p0, negative_rewards_p1, plot_title,
         moving_average_window_size=config['plotting_settings']['plot_moving_average_window_size'],
         episode_series_x_axis_plot_range=config['plotting_settings']['reward_time_series_x_axis_plot_range'],  
         dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting cumulative only step rewards 
    plot_name = experiment_name + '_only_step_rewards'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Cumulative only step rewards per episode'
    
    plot_reward_per_episode_series(only_step_rewards_p0, only_step_rewards_p1, plot_title,
         moving_average_window_size=config['plotting_settings']['plot_moving_average_window_size'],
         episode_series_x_axis_plot_range=config['plotting_settings']['reward_time_series_x_axis_plot_range'],  
         dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting cumulative full rewards without coin 
    plot_name = experiment_name + '_full_rewards_without_coin'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Cumulative full rewards without coin per episode'
    
    plot_reward_per_episode_series(full_rewards_without_coin_p0, full_rewards_without_coin_p1, plot_title,
         moving_average_window_size=config['plotting_settings']['plot_moving_average_window_size'],
         episode_series_x_axis_plot_range=config['plotting_settings']['reward_time_series_x_axis_plot_range'],  
         dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting cumulative full rewards without step 
    plot_name = experiment_name + '_full_rewards_without_step'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Cumulative full rewards without step per episode'
    
    plot_reward_per_episode_series(full_rewards_without_step_p0, full_rewards_without_step_p1, plot_title,
         moving_average_window_size=config['plotting_settings']['plot_moving_average_window_size'],
         episode_series_x_axis_plot_range=config['plotting_settings']['reward_time_series_x_axis_plot_range'],  
         dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting evolving win ratio 
    plot_name = experiment_name + '_win_ratio'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Win ratio for DM'
    
    plot_result_ration(result_series=game_result_p0,
                       episode_range_to_eval=config['plotting_settings']['episode_range_to_eval'],
                       plot_title=plot_title,
                       result_type_to_plot="win",
                       episode_series_x_axis_plot_range=config['plotting_settings']['game_result_ratio_x_axis_plot_range'],  
                       dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting evolving loss ratio 
    plot_name = experiment_name + '_loss_ratio'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Loss ratio for DM'
    
    plot_result_ration(result_series=game_result_p0,
                       episode_range_to_eval=config['plotting_settings']['episode_range_to_eval'],
                       plot_title=plot_title,
                       result_type_to_plot="loss",
                       episode_series_x_axis_plot_range=config['plotting_settings']['game_result_ratio_x_axis_plot_range'],  
                       dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Plotting evolving draw ratio 
    plot_name = experiment_name + '_draw_ratio'
    plot_path = os.path.join(results_path, plot_name)
    plot_title = 'Draw ratio for DM'
    
    plot_result_ration(result_series=game_result_p0,
                       episode_range_to_eval=config['plotting_settings']['episode_range_to_eval'],
                       plot_title=plot_title,
                       result_type_to_plot="draw",
                       episode_series_x_axis_plot_range=config['plotting_settings']['game_result_ratio_x_axis_plot_range'],  
                       dir=plot_path)
    logging.info("Plot saved to %s.png", plot_path)
    
    # Save trajectory logs if enabled.
    if log_trajectory:
        # Define structured array
        trajectory_dtype = np.dtype([
            ('experiment_num', np.int16), ('episode_num', np.int16), ('step_num', np.int16),
            ('p0_loc_old_row', np.int8), ('p0_loc_old_col', np.int8),
            ('p1_loc_old_row', np.int8), ('p1_loc_old_col', np.int8),
            ('p0_loc_new_row', np.int8), ('p0_loc_new_col', np.int8),
            ('p1_loc_new_row', np.int8), ('p1_loc_new_col', np.int8),
            ('coin1_row', np.int8), ('coin1_col', np.int8),
            ('coin2_row', np.int8), ('coin2_col', np.int8),
            ('p0_action_move', np.int8), ('p0_action_push', np.int8),
            ('p1_action_move', np.int8), ('p1_action_push', np.int8),
            ('p0_reward', np.float32), ('p1_reward', np.float32),
            ('p0_cum_reward', np.float32), ('p1_cum_reward', np.float32)
        ])

        # Convert list of logs to structured array
        # Ensure each row is a tuple to match the dtype structure
        save_array = np.array(
            [tuple(row) for row in trajectory_logs_all_experiments], 
            dtype=trajectory_dtype
        )

        # Save the array as a compressed file
        traj_path = os.path.join(results_path, 'trajectory_log.npz')
        np.savez_compressed(traj_path, trajectory_logs_all_experiments=save_array) # I suggest a simpler key like 'trajectory'
        
        logging.info(f"Trajectory log saved to {traj_path}. Shape: {save_array.shape}")
    
    # Save accumulated reward data for each experiment
    
    # Saving full rewards
    p0_full_rewards_path = os.path.join(results_path, 'full_rewards_per_episode_p0.npz')
    np.savez_compressed(p0_full_rewards_path, full_rewards_p0 = np.array(full_rewards_p0))
    logging.info("Full rewards for player 1 saved to %s.", p0_full_rewards_path)
    
    p1_full_rewards_path = os.path.join(results_path, 'full_rewards_per_episode_p1.npz')
    np.savez_compressed(p1_full_rewards_path, full_rewards_p1 = np.array(full_rewards_p1))
    logging.info("Full rewards for player 2 saved to %s.", p1_full_rewards_path)
    
    # Saving positive rewards
    p0_positive_rewards_path = os.path.join(results_path, 'positive_rewards_per_episode_p0.npz')
    np.savez_compressed(p0_positive_rewards_path, positive_rewards_p0 = np.array(positive_rewards_p0))
    logging.info("Positive rewards for player 1 saved to %s.", p0_positive_rewards_path)
    
    p1_positive_rewards_path = os.path.join(results_path, 'positive_rewards_per_episode_p1.npz')
    np.savez_compressed(p1_positive_rewards_path, positive_rewards_p1 = np.array(positive_rewards_p1))
    logging.info("Positive rewards for player 2 saved to %s.", p1_positive_rewards_path)
    
    # Saving negative rewards
    p0_negative_rewards_path = os.path.join(results_path, 'negative_rewards_per_episode_p0.npz')
    np.savez_compressed(p0_negative_rewards_path, negative_rewards_p0 = np.array(negative_rewards_p0))
    logging.info("Negative rewards for player 1 saved to %s.", p0_negative_rewards_path)
    
    p1_negative_rewards_path = os.path.join(results_path, 'negative_rewards_per_episode_p1.npz')
    np.savez_compressed(p1_negative_rewards_path, negative_rewards_p1 = np.array(negative_rewards_p1))
    logging.info("Negative rewards for player 2 saved to %s.", p1_negative_rewards_path)
    
    # Saving only step rewards
    p0_only_step_rewards_path = os.path.join(results_path, 'only_step_rewards_per_episode_p0.npz')
    np.savez_compressed(p0_only_step_rewards_path, only_step_rewards_p0 = np.array(only_step_rewards_p0))
    logging.info("Only step rewards for player 1 saved to %s.", p0_only_step_rewards_path)
    
    p1_only_step_rewards_path = os.path.join(results_path, 'only_step_rewards_per_episode_p1.npz')
    np.savez_compressed(p1_only_step_rewards_path, only_step_rewards_p1 = np.array(only_step_rewards_p1))
    logging.info("Only step rewards for player 2 saved to %s.", p1_only_step_rewards_path)
    
    # Saving full rewards without coins
    p0_full_rewards_without_coin_path = os.path.join(results_path, 'full_rewards_without_coin_per_episode_p0.npz')
    np.savez_compressed(p0_full_rewards_without_coin_path, full_rewards_without_coin_p0 = np.array(full_rewards_without_coin_p0))
    logging.info("Full rewards without coin for player 1 saved to %s.", p0_full_rewards_without_coin_path)
    
    p1_full_rewards_without_coin_path = os.path.join(results_path, 'full_rewards_without_coin_per_episode_p1.npz')
    np.savez_compressed(p1_full_rewards_without_coin_path, full_rewards_without_coin_p1 = np.array(full_rewards_without_coin_p1))
    logging.info("Full rewards without coin for player 2 saved to %s.", p1_full_rewards_without_coin_path)
    
    # Saving full rewards without step
    p0_full_rewards_without_step_path = os.path.join(results_path, 'full_rewards_without_step_per_episode_p0.npz')
    np.savez_compressed(p0_full_rewards_without_step_path, full_rewards_without_step_p0 = np.array(full_rewards_without_step_p0))
    logging.info("Full rewards without step for player 1 saved to %s.", p0_full_rewards_without_step_path)
    
    p1_full_rewards_without_step_path = os.path.join(results_path, 'full_rewards_without_step_per_episode_p1.npz')
    np.savez_compressed(p1_full_rewards_without_step_path, full_rewards_without_step_p1 = np.array(full_rewards_without_step_p1))
    logging.info("Full rewards without step for player 2 saved to %s.", p1_full_rewards_without_step_path)
    
    # Save game results of player 1 for each experiment
    game_result_p0_path = os.path.join(results_path, 'game_result_episode_p0.npz')
    np.savez_compressed(game_result_p0_path, game_result_p0 = np.array(game_result_p0))
    logging.info("Game results for player 1 saved to %s.", game_result_p0_path)
    
    # Save the configuration file used for this run for full reproducibility.
    shutil.copy(config_file_path, results_path)
    
    # The saved file will have its original name inside the results_path directory.
    saved_config_path = os.path.join(results_path, os.path.basename(config_file_path))
    logging.info("Configuration file copied for reproducibility to %s", saved_config_path)
    
    
    return str(results_path)
    
    
