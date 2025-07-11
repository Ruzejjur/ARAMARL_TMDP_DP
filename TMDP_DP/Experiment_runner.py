import yaml
import numpy as np
from tqdm.notebook import tqdm
import os

# Import your custom modules
from engine_DP import CoinGame
import agents
from utils.exploration_schedule_utils import linear_epsilon_decay
from utils.plot_utils import plot, animate_trajectory_from_log

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
    import inspect
    sig = inspect.signature(AgentClass.__init__)
    valid_params = {k: v for k, v in full_params.items() if k in sig.parameters}

    return AgentClass(**valid_params)

def run_single_episode(env, p1, p2, experiment_num, episode_num, log_trajectory=False):
    """Runs a single episode of the game and returns the rewards."""
    env.reset()
    s = env.get_state()
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
        a1 = p1.act(obs=s, env=env)
        a2 = p2.act(obs=s, env=env)
        
        # Log state for visualization before the step
        p1_loc_old, p2_loc_old = env.player_0_pos.copy(), env.player_1_pos.copy()

        # Transition to next time step
        s_new, rewards, done = env.step((a1, a2))
        
        # Agents update their Q/Value functions
        p1.update(obs=s, actions=(a1, a2), new_obs=s_new, rewards=(rewards[0], rewards[1]))
        p2.update(obs=s, actions=(a2, a1), new_obs=s_new, rewards=(rewards[1], rewards[0]))
        
        # Set the current state to the new state
        s = s_new
        
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

def run_experiment(config):
    """
    Main function to run the entire set of experiments based on the config.
    """
    
    # --- Setup ---
    exp_settings = config['experiment_settings']
    env_settings = config['environment_settings']
    agent_configs = config['agent_settings']
    learning_params = config['learning_params']
    
    # Create results directory if it doesn't exist
    os.makedirs(exp_settings['results_dir'], exist_ok=True)
    
    # Initialize Environment
    env = CoinGame(**env_settings['params'])
    
    # Setup Epsilon Decay Schedule
    n_episodes = exp_settings['num_episodes']
    epsilon_config = learning_params['epsilon_decay']
    epsilon_schedule = linear_epsilon_decay(epsilon_config['start'], epsilon_config['end'], n_episodes)
    
    # --- Data Logging ---
    all_rewards_p1 = []
    all_rewards_p2 = []
    final_trajectory_log = []

    # --- Experiment Loop ---
    for experiment_num in tqdm(range(exp_settings['num_runs']), desc="Experiment runs"):
        np.random.seed(experiment_num) # for reproducibility
        
        # --- Agent Initialization ---
        common_agent_params = {
            'n_states': env.n_states,
            'action_space': np.array(range(len(env.combined_actions))),
            'opponent_action_space': np.array(range(len(env.combined_actions))),
            'grid_size': env_settings['params']['grid_size'],
            'gamma': learning_params['gamma'],
            'learning_rate': learning_params['learning_rate'],
            'epsilon': epsilon_schedule[0],
            'env': env 
        }
        
        p1 = create_agent({**agent_configs['player_1'], 'params': {**agent_configs['player_1'].get('params', {}), 'player_id': 0}}, common_agent_params)
        p2 = create_agent({**agent_configs['player_2'], 'params': {**agent_configs['player_2'].get('params', {}), 'player_id': 1}}, common_agent_params)
        
        run_rewards_p1 = []
        run_rewards_p2 = []

        # --- Episode Loop ---
        for episode_num in tqdm(range(n_episodes), desc=f"Epoch for experiment {experiment_num+1}", leave=False):
            # Check if this is the last episode of the last run to log trajectory for animation
            log_traj_flag = (experiment_num == exp_settings['num_runs'] - 1) and (episode_num == n_episodes - 1)
            
            rew1, rew2, trajectory = run_single_episode(env, p1, p2, experiment_num, episode_num, log_trajectory=log_traj_flag)
            
            run_rewards_p1.append(rew1)
            run_rewards_p2.append(rew2)
            
            if log_traj_flag:
                final_trajectory_log = trajectory

            # Update epsilon for both agents
            new_epsilon = epsilon_schedule[episode_num]
            p1.update_epsilon(new_epsilon)
            p2.update_epsilon(new_epsilon)
            
        all_rewards_p1.append(run_rewards_p1)
        all_rewards_p2.append(run_rewards_p2)

    # --- Plotting and Saving ---
    plot_path = os.path.join(exp_settings['results_dir'], exp_settings['name'])
    plot(all_rewards_p1, all_rewards_p2, 
         moving_average_window_size=config['plotting_settings']['moving_average_window'], 
         dir=plot_path)
    print(f"Plot saved to {plot_path}.png")
    
    # --- Animation ---
    if final_trajectory_log:
        print("Generating animation for the last episode of the last run...")
        animate_trajectory_from_log(final_trajectory_log, grid_size=env.grid_size, fps=2)
        print("Animation saved as trajectory.mp4")