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
from tqdm.notebook import tqdm
import os
import inspect
from datetime import datetime

# Import your custom modules
from engine_DP import CoinGame
import agents
from utils.exploration_schedule_utils import linear_epsilon_decay
from utils.plot_utils import plot

def load_config(config_path: str) -> dict:
    """
    Loads and returns the YAML configuration file.

    This file contains all the necessary settings for the agents and the
    environment.

    Args:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration settings.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
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
            "Please add them to the 'params' section in your config.yaml for this agent."
        )
        
    # Filter the provided parameters to only include those expected by the constructor.
    # This prevents errors from passing unexpected arguments (e.g., 'epsilon_decay_agent').
    valid_params = {k: v for k, v in full_params.items() if k in sig.parameters}
    
    return AgentClass(**valid_params)

def run_single_episode(env: CoinGame, p1: agents.BaseAgent, p2: agents.BaseAgent, experiment_num: int, episode_num: int, log_trajectory: bool = False) -> tuple:
    """
    Runs a single episode of the game from start to finish.

    Args:
        env (CoinGame): The game environment instance.
        p1 (BaseAgent): The agent instance for player 1.
        p2 (BaseAgent): The agent instance for player 2.
        experiment_num (int): The current experiment run number (for logging).
        episode_num (int): The current episode number (for logging).
        log_trajectory (bool): If True, records a detailed log of each step for
                               later visualization.

    Returns:
        A tuple containing:
        - episode_rewards_p1 (float): Total reward for player 1 in the episode.
        - episode_rewards_p2 (float): Total reward for player 2 in the episode.
        - trajectory_log (list): A log of states, actions, and rewards, if enabled.
    """
    env.reset()
    obs = env.get_state()
    done = False
    
    episode_rewards_p1 = 0
    episode_rewards_p2 = 0
    
    trajectory_log = []
    
    # Log the initial state if trajectory logging is enabled.
    if log_trajectory:
         trajectory_log.append({
            'p0_loc_old': ["None", "None"],
            'p1_loc_old': ["None", "None"],
            'p0_loc_new': env.player_0_pos.copy(),
            'p1_loc_new': env.player_1_pos.copy(),
            'coin1': env.coin_0_pos.copy() if env.coin0_available else None,
            'coin2': env.coin_1_pos.copy() if env.coin1_available else None, 
            'p0_action': ["None", "None"],
            'p1_action': ["None", "None"],
            'p0_reward': None,
            'p1_reward': None,
            'experiment': experiment_num,
            'epoch': episode_num
        })

    while not done:
        
        # Get actions from both agents.
        a1 = p1.act(obs=obs, env=env)
        a2 = p2.act(obs=obs, env=env)
        
        # Store current locations for logging before the step.
        p1_loc_old, p2_loc_old = env.player_0_pos.copy(), env.player_1_pos.copy()

        # Execute actions in the environment.
        s_new, rewards, done = env.step((a1, a2))
        
        # Allow learning agents to update their internal models/policies.
        if isinstance(p1, agents.LearningAgent):
            p1.update(obs=obs, actions=(a1, a2), new_obs=s_new, rewards=(rewards[0], rewards[1]))
        if isinstance(p2, agents.LearningAgent):
            p2.update(obs=obs, actions=(a1, a2), new_obs=s_new, rewards=(rewards[0], rewards[1]))
        
        # Advance the state
        obs = s_new
        
        # Accumulate rewards.   
        episode_rewards_p1 += rewards[0]
        episode_rewards_p2 += rewards[1]
        
        # Log step details if enabled.
        if log_trajectory:
            trajectory_log.append({
                'p0_loc_old': p1_loc_old,
                'p1_loc_old': p2_loc_old,
                'p0_loc_new': env.player_0_pos.copy(),
                'p1_loc_new': env.player_1_pos.copy(),
                'coin1': env.coin_0_pos.copy() if env.coin0_available else None,
                'coin2': env.coin_1_pos.copy() if env.coin1_available else None, 
                'p0_action': env.combined_actions[a1],
                'p1_action': env.combined_actions[a2],
                'p0_reward': rewards[0],
                'p1_reward': rewards[1],
                'experiment': experiment_num,
                'epoch': episode_num
            })


    return episode_rewards_p1, episode_rewards_p2, trajectory_log

def run_experiment(config:dict, log_trajectory: bool = False) -> str:
    """
    Main function to run the entire set of experiments based on the config.

    This function handles the main experiment loop, including setting up
    logging directories, initializing the environment and agents for each run,
    looping through episodes, and saving the final results and plots.

    Args:
        config (dict): The full experiment configuration dictionary.
        log_trajectory (bool): If True, a detailed log of the last episode of
                               each run is saved for animation.

    Returns:
        str: The path to the directory where results for this run were saved.
    """
    
    # --- Setup ---
    exp_settings = config['experiment_settings']
    env_settings = config['environment_settings']
    agent_configs = config['agent_settings']
    
    # Create a unique, timestamped directory for this experiment's results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = exp_settings.get('name', 'experiment')
    run_dir_name = f"{experiment_name}_{timestamp}"
    results_path = os.path.join(exp_settings['results_dir'], run_dir_name)
    os.makedirs(results_path, exist_ok=True)
    print(f"Results for this run will be saved in: {results_path}")
    
    # Initialize the single environment instance.
    env = CoinGame(**env_settings['params'])
    
    # Setup Agent Epsilon Decay Schedules for both players if they are learning agents.
    n_episodes = exp_settings['num_episodes']
    
    epsilon_agent_config_p1 = agent_configs['player_1']["params"]
    epsilon_agent_schedule_p1 = linear_epsilon_decay(epsilon_agent_config_p1['epsilon'],
                                                     agent_configs['player_1']['epsilon_decay_agent']['end'], n_episodes)
    
    epsilon_agent_config_p2 = agent_configs['player_2']['params']
    epsilon_agent_schedule_p2 = linear_epsilon_decay(epsilon_agent_config_p2['epsilon'],
                                                     agent_configs['player_2']['epsilon_decay_agent']['end'], n_episodes)
    
    # Setup Lower K level Epsilon Decay Schedules for both players if they implement k-level hierarchy.
    
    epsilon_lower_k_level_config_p1 = agent_configs['player_1']["params"]
    epsilon_lower_k_level_schedule_p1 = linear_epsilon_decay(epsilon_lower_k_level_config_p1['epsilon_lower_k_levels'],
                                                             agent_configs['player_1']['epsilon_decay_inernal_opponent_model']['end'], n_episodes)
    
    epsilon_lower_k_level_config_p2 = agent_configs['player_2']['params']
    epsilon_lower_k_level_schedule_p2 = linear_epsilon_decay(epsilon_lower_k_level_config_p2['epsilon_lower_k_levels'],
                                                             agent_configs['player_2']['epsilon_decay_inernal_opponent_model']['end'], n_episodes)
            
    # --- Data Logging Initialisation ---
    all_rewards_p1 = []
    all_rewards_p2 = []
    trajectory_logs_all_experiments = []

    # --- Experiment Loop (for multiple independent runs) ---
    for experiment_num in tqdm(range(exp_settings['num_runs']), desc="Experiment runs"):
        np.random.seed(experiment_num) # for reproducibility
        
        # Trajectory log for this experiment
        trajectory_log_single_experiment = []
        
        # --- Agent Initialization ---
        common_agent_params = {
            'n_states': env.n_states,
            'action_space': np.array(range(len(env.combined_actions))),
            'opponent_action_space': np.array(range(len(env.combined_actions))),
            'grid_size': env_settings['params']['grid_size'],
            'env': env 
        }
        
        p1_params = {'class': agent_configs['player_1']['class'],
                    'params': {**agent_configs['player_1'].get('params', {})
                    ,'player_id': 0}}
        
        p1 = create_agent(p1_params, common_agent_params)
        
        p2_params = {'class': agent_configs['player_2']['class'],
                     'params': {**agent_configs['player_2'].get('params', {}),
                     'player_id': 1}}
        
        p2 = create_agent(p2_params, common_agent_params)
        
        # Single experiment rewards initalisation
        run_rewards_p1 = []
        run_rewards_p2 = []
            
        # --- Episode Loop ---
        for episode_num in tqdm(range(n_episodes), desc=f"Epoch for experiment {experiment_num+1}", leave=False):
            
            rew1, rew2, trajectory = run_single_episode(env, p1, p2, experiment_num, episode_num, log_trajectory=log_trajectory)
            
            # Append the trajectory log for this epoch to experiment log 
            trajectory_log_single_experiment.append(trajectory)
            
            run_rewards_p1.append(rew1)
            run_rewards_p2.append(rew2)
            
            # Extract new epsilon from schedules for agent and internal agent model
            new_epsilon_agent_p1 = epsilon_agent_schedule_p1[episode_num]
            new_epsilon_agent_p2 = epsilon_agent_schedule_p1[episode_num]
            
            new_epsilon_lower_k_p1 = epsilon_lower_k_level_schedule_p1[episode_num]
            new_epsilon_lower_k_p2 = epsilon_lower_k_level_schedule_p2[episode_num]
            
            # Update epsilon for learning agents.
            if isinstance(p1, agents.LearningAgent):
                p1.update_epsilon(new_epsilon_agent_p1, new_epsilon_lower_k_p1)
            if isinstance(p2, agents.LearningAgent):
                p2.update_epsilon(new_epsilon_agent_p2, new_epsilon_lower_k_p2)
            
        if log_trajectory:
            # Append the trajectory logs of this experiment to the total logs
            trajectory_logs_all_experiments.append(trajectory_log_single_experiment)
        
        all_rewards_p1.append(run_rewards_p1)
        all_rewards_p2.append(run_rewards_p2)

    # --- Plotting and Saving results ---
    plot_path = os.path.join(results_path, experiment_name)
    plot(all_rewards_p1, all_rewards_p2, 
         moving_average_window_size=config['plotting_settings']['moving_average_window'], 
         dir=plot_path)
    print(f"Plot saved to {plot_path}.png")
    
    # Save trajectory logs if enabled.
    if log_trajectory:
        traj_path = os.path.join(results_path, 'trajectory_log.npy')
        save_array = np.array(trajectory_logs_all_experiments, dtype=object) # Converting into object so the array cah be saved 
        np.save(traj_path, save_array, allow_pickle=True)
    
    # Save accumulated reward data for each experiment
    print("Saving rewards of both players.")
    np.save(os.path.join(results_path, 'rewards_p1.npy'), np.array(all_rewards_p1))
    np.save(os.path.join(results_path, 'rewards_p2.npy'), np.array(all_rewards_p2))

    
    # Save the configuration file used for this run for full reproducibility.
    config_path = os.path.join(results_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")
    
    
    return results_path
    
    
