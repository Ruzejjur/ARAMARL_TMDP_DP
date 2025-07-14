import yaml
import numpy as np
from tqdm.notebook import tqdm
import os
import inspect
import json
from datetime import datetime

# Import your custom modules
from engine_DP import CoinGame
import agents
from utils.exploration_schedule_utils import linear_epsilon_decay
from utils.plot_utils import plot


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert NumPy arrays to Python lists
        if isinstance(obj, np.integer):
            return int(obj) # Convert NumPy integers to Python integers
        if isinstance(obj, np.floating):
            return float(obj) # Convert NumPy floats to Python floats
        # Let the base class default method raise the TypeError for other types
        return json.JSONEncoder.default(self, obj)

def load_config(config_path):
    """Loads and returns the YAML configuration file containing agent and environment setup."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def create_agent(agent_config, common_params):
    """
    Factory function to create an agent instance from its configuration.
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
        
    # Filter only the parameters required by the agent's constructor
    sig = inspect.signature(AgentClass.__init__)
    valid_params = {k: v for k, v in full_params.items() if k in sig.parameters}

    return AgentClass(**valid_params)

def run_single_episode(env, p1, p2, experiment_num, episode_num, log_trajectory=False):
    """Runs a single episode of the game and returns the rewards."""
    env.reset()
    obs = env.get_state()
    done = False
    
    episode_rewards_p1 = 0
    episode_rewards_p2 = 0
    
    trajectory_log = []
    
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
        
        # Get agents actions
        a1 = p1.act(obs=obs, env=env)
        a2 = p2.act(obs=obs, env=env)
        
        # Log state for visualization before the step
        p1_loc_old, p2_loc_old = env.player_0_pos.copy(), env.player_1_pos.copy()

        # Transition to next time step
        s_new, rewards, done = env.step((a1, a2))
        
        # Agents update their Q/Value functions
        p1.update(obs=obs, actions=(a1, a2), new_obs=s_new, rewards=(rewards[0], rewards[1]))
        p2.update(obs=obs, actions=(a1, a2), new_obs=s_new, rewards=(rewards[0], rewards[1]))
        
        # Set the current state to the new state
        obs = s_new
        
        # Add the observed reward to the episode reward of both agents 
        # * Note: The rewards are observed simultaneously
        episode_rewards_p1 += rewards[0]
        episode_rewards_p2 += rewards[1]
        
        # Log state in single episode
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

def run_experiment(config, log_trajectory=False):
    """
    Main function to run the entire set of experiments based on the config.
    """
    
    # --- Setup ---
    exp_settings = config['experiment_settings']
    env_settings = config['environment_settings']
    agent_configs = config['agent_settings']
    
    # --- Create a unique directory for this experiment run ---
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = exp_settings.get('name', 'experiment')
    run_dir_name = f"{experiment_name}_{timestamp}"
    results_path = os.path.join(exp_settings['results_dir'], run_dir_name)
    os.makedirs(results_path, exist_ok=True)
    print(f"Results for this run will be saved in: {results_path}")
    
    # Initialize Environment
    env = CoinGame(**env_settings['params'])
    
    # Setup Epsilon Decay Schedules for both players
    n_episodes = exp_settings['num_episodes']
    
    epsilon_config_p1 = agent_configs['player_1']["params"]
    epsilon_schedule_p1 = linear_epsilon_decay(epsilon_config_p1['epsilon'], agent_configs['player_1']['epsilon_decay']['end'], n_episodes)
    
    epsilon_config_p2 = agent_configs['player_2']['params']
    epsilon_schedule_p2 = linear_epsilon_decay(epsilon_config_p2['epsilon'], agent_configs['player_2']['epsilon_decay']['end'], n_episodes)
    
    # --- Data Logging ---
    all_rewards_p1 = []
    all_rewards_p2 = []
    
    if log_trajectory:
        # Total trajectory logs for all experiments
        trajectory_logs_all_experiments = []

    # --- Experiment Loop ---
    for experiment_num in tqdm(range(exp_settings['num_runs']), desc="Experiment runs"):
        np.random.seed(experiment_num) # for reproducibility
        
        # Trajectory log for this episode
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
        
        run_rewards_p1 = []
        run_rewards_p2 = []
            
        # --- Episode Loop ---
        for episode_num in tqdm(range(n_episodes), desc=f"Epoch for experiment {experiment_num+1}", leave=False):

            rew1, rew2, trajectory = run_single_episode(env, p1, p2, experiment_num, episode_num, log_trajectory=log_trajectory)
            
            # Append the trajectory log for this epoch to experiment log 
            trajectory_log_single_experiment.append(trajectory)
            
            run_rewards_p1.append(rew1)
            run_rewards_p2.append(rew2)

            # Update epsilon for both agents
            new_epsilon_p1 = epsilon_schedule_p1[episode_num]
            new_epsilon_p2 = epsilon_schedule_p2[episode_num]
            
            p1.update_epsilon(new_epsilon_p1)
            p2.update_epsilon(new_epsilon_p2)
            
        if log_trajectory:
            # Append the trajectory logs of this experiment to the total logs
            trajectory_logs_all_experiments.append(trajectory_log_single_experiment)
        
        all_rewards_p1.append(run_rewards_p1)
        all_rewards_p2.append(run_rewards_p2)

    # --- Plotting and Saving ---
    plot_path = os.path.join(results_path, experiment_name)
    plot(all_rewards_p1, all_rewards_p2, 
         moving_average_window_size=config['plotting_settings']['moving_average_window'], 
         dir=plot_path)
    print(f"Plot saved to {plot_path}.png")
    
    # --- Results ---
    
    # Save trajectory of each game
    if log_trajectory:
        traj_path = os.path.join(results_path, 'trajectory_log.json')
        with open(traj_path, 'w') as f:
            # We save the log for the last experiment run
            json.dump(trajectory_logs_all_experiments, f, indent=4, cls=NumpyArrayEncoder)
        print(f"Trajectory log saved to {traj_path}")
    
    # Save rewards of both players
    print("Saving rewards of both players.")
    np.save(os.path.join(results_path, 'rewards_p1.npy'), np.array(all_rewards_p1))
    np.save(os.path.join(results_path, 'rewards_p2.npy'), np.array(all_rewards_p2))

    
    # Save configuration file
    config_path = os.path.join(results_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")
    
    
    return trajectory_logs_all_experiments if log_trajectory else None
    
    
