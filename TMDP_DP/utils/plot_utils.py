import numpy as np

# Also ffmpeg is required for video saving
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

def moving_average(array, moving_average_window_size=3):
    """
    Compute the right-aligned moving average of a 1D array.

    Parameters:
        array (array-like): Input array containing numerical data.
        moving_average_window_size (int, optional): Window size for the moving average. Defaults to 3.

    Returns:
        ndarray: Array of moving averages with length len(array) - moving_average_window_size + 1.
    """
    # Compute cumulative sum of the input array as float
    ret = np.cumsum(array, dtype=float)
    
    # Subtract the cumulative sum shifted by 'moving_average_window_size' to get the sum over each window
    ret[moving_average_window_size:] = ret[moving_average_window_size:] - ret[:-moving_average_window_size]
    
    # Divide by window size to obtain the moving average
    return ret[moving_average_window_size - 1:] / moving_average_window_size

def plot_reward_per_episode_series(r0ss, r1ss, plot_title, moving_average_window_size, time_series_range, dir=None):
    """
    Plot smoothed reward trajectories for two agents over multiple experiments.

    Parameters:
        r0ss (list of arrays): Rewards for Agent A across experiments.
        r1ss (list of arrays): Rewards for Agent B across experiments.
        plot_title (str): Title of the plot.
        moving_average_window_size(int): Size of right-aligned moving average window.
        time_series_range (list/tuple): A list [xmin, xmax] to set the x-axis view.
        dir (str, optional): If provided, saves the plot to 'dir.png'.

    Returns:
        None
    """
    # Apply 'ggplot' style
    plt.style.use('ggplot')
    # Create figure and axis explicitly d
    fig, ax = plt.subplots()

    number_of_experiment = len(r0ss)

    for i in range(number_of_experiment):
        # Calculate the moving average for the full series
        p0_reward_series = moving_average(r0ss[i], moving_average_window_size)
        p1_reward_series = moving_average(r1ss[i], moving_average_window_size)
        
        x_axis = np.arange(moving_average_window_size - 1, len(r0ss[i]))
        ax.plot(x_axis, p0_reward_series, 'b', alpha=0.05)
        ax.plot(x_axis, p1_reward_series, 'r', alpha=0.05)

    p0_average_reward_series = moving_average(np.mean(r0ss, axis=0), moving_average_window_size)
    p1_average_reward_series = moving_average(np.mean(r1ss, axis=0), moving_average_window_size)
    
    x_axis_avg = np.arange(moving_average_window_size - 1, len(np.mean(r0ss, axis=0)))
    ax.plot(x_axis_avg, p0_average_reward_series, 'b', alpha=0.5)
    ax.plot(x_axis_avg, p1_average_reward_series, 'r', alpha=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative reward per episode')
    ax.set_title(plot_title)
    
    ax.set_xlim(time_series_range)

    custom_lines = [Line2D([0], [0], color='b', label='DM'),
                    Line2D([0], [0], color='r', label='Adversary')]
    ax.legend(handles=custom_lines)

    if dir is not None:
        # Set figure background to transparent
        #fig.patch.set_alpha(0.0) # jpeg does not support transparency
        fig.savefig(
            f"{dir}.jpg",
            format='jpeg',
            dpi=1200,
            bbox_inches='tight',
            pil_kwargs={
                "quality": 90, # values above 95 remove compresion
                "optimize": True,
                "progressive": False
            }
        )    
        
def plot_result_ration(result_series, plot_title, result_type_to_plot, time_series_range, dir=None):
    """
    Plot result ration trajectories for an agent over multiple experiments.

    Parameters:
        result_series (list of arrays): A list where each element is a 1D numpy array
                                        of game results for one experiment run.
        plot_title (str): Title of the plot.
        result_type_to_plot (str): One of 'win', 'loss', 'draw', or 'timeout'.
        time_series_range (list/tuple): A list [xmin, xmax] to set the x-axis view.
        dir (str, optional): If provided, saves the plot to 'dir.png'.

    Returns:
        None
    """
    # Apply 'ggplot' style
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    # Map the result type string to the integer code used in the data
    result_code_map = {
        "win": 1,
        "loss": -1,
        "draw": 0,
        "timeout": -2
    }
    
    if result_type_to_plot not in result_code_map:
        raise ValueError("Invalid result type. Use 'win', 'loss', 'draw', or 'timeout'.")
    
    result_code = result_code_map[result_type_to_plot]

    num_experiments = len(result_series)
    num_episodes = len(result_series[0])

    # Create an array of episode numbers [1, 2, ..., N] for the ratio calculation
    episode_indices = np.arange(1, num_episodes + 1)
    
    # Store the calculated ratios for each experiment to compute the mean later
    all_ratios = []

    for i in range(num_experiments):
        # Create a binary array: 1 if the result matches the type, 0 otherwise
        binary_results = (np.array(result_series[i]) == result_code).astype(int)
        
        # Calculate the cumulative sum of the specific result
        cumulative_results = np.cumsum(binary_results)
        
        # Calculate the ratio of that result type over the episodes
        result_ratio = cumulative_results / episode_indices
        all_ratios.append(result_ratio)
        
        # Plot the ratio for the individual run with high transparency
        ax.plot(episode_indices, result_ratio, 'b', alpha=0.05)

    # Calculate the mean ratio across all experiments
    mean_ratio_series = np.mean(all_ratios, axis=0)
    
    # Plot the averaged line with lower transparency
    ax.plot(episode_indices, mean_ratio_series, 'b', alpha=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel(f'{result_type_to_plot.capitalize()} Ratio')
    ax.set_title(plot_title)
    
    ax.set_xlim(time_series_range)

    # Simplified legend for a single agent's metric
    custom_lines = [Line2D([0], [0], color='b', label=f'DM {result_type_to_plot} ratio')]
    ax.legend(handles=custom_lines)

    if dir is not None:
        fig.savefig(
            f"{dir}.jpg",
            format='jpeg',
            dpi=1200,
            bbox_inches='tight',
            pil_kwargs={
                "quality": 90,
                "optimize": True,
                "progressive": False
            }
        )
        
def animate_trajectory_from_log(trajectory_log, grid_size=4, fps=4, dpi=100):
    """
    Animate a trajectory from an enriched trajectory log that includes actions and rewards.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_aspect('equal')
    ax.grid(True)

    title = ax.set_title("")
    
    info_text = fig.text(0.02, 0.5, "", ha='left', va='center', fontsize=10)
    
    move_action_codes = ["Down", "Right", "Up", "Left"]
    action_push_codes = ["No push", "Push"]

    # Agent and coin markers
    dm_dot, = ax.plot([], [], 'bo', label='DM (Blue)')
    adv_dot, = ax.plot([], [], 'ro', label='Adv (Red)')
    coin1_dot, = ax.plot([], [], 'y*', label='Coin 1', markersize=15)
    coin2_dot, = ax.plot([], [], 'g*', label='Coin 2', markersize=15)

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    def init():
        dm_dot.set_data([], [])
        adv_dot.set_data([], [])
        coin1_dot.set_data([], [])
        coin2_dot.set_data([], [])
        title.set_text("")
        return dm_dot, adv_dot, coin1_dot, coin2_dot, title

    def update(frame):
        state = trajectory_log[frame]

        
        if isinstance(state['p0_loc_new'], (list, np.ndarray)) and len(state['p0_loc_new']) == 2:
            dm_dot.set_data([state['p0_loc_new'][1]], [state['p0_loc_new'][0]])
        else:
            dm_dot.set_data([], [])

        if isinstance(state['p1_loc_new'], (list, np.ndarray)) and len(state['p1_loc_new']) == 2:
            adv_dot.set_data([state['p1_loc_new'][1]], [state['p1_loc_new'][0]])
        else:
            adv_dot.set_data([], [])

        if isinstance(state['coin1'], (list, np.ndarray)) and len(state['coin1']) == 2:
            coin1_dot.set_data([state['coin1'][1]], [state['coin1'][0]])
        else:
            coin1_dot.set_data([-10], [-10])  # hide

        if isinstance(state['coin2'], (list, np.ndarray)) and len(state['coin2']) == 2:
            coin2_dot.set_data([state['coin2'][1]], [state['coin2'][0]])
        else:
            coin2_dot.set_data([-10], [-10])  # hide

        title.set_text(
            f"Exp {state['experiment'] + 1} | Ep {state['epoch'] + 1} | Step {frame}"
        )
        
       
        if(np.array_equal(state['p0_action'], ["None", "None"])): 
            DM_action_string = ["None", "None"]
        else: 
            DM_action_string = f"DM Action: [{move_action_codes[state['p0_action'][0]]}, {action_push_codes[state['p0_action'][1]]}]"
            
        if(np.array_equal(state['p1_action'], ["None", "None"])): 
            Adv_action_string = ["None", "None"]
        else: 
            Adv_action_string = f"Adv Action: [{move_action_codes[state['p1_action'][0]]}, {action_push_codes[state['p1_action'][1]]}]"
            
        # Update a text box the grid for step and reward info
        info_text.set_text(
            (
                f"Prev. location DM: [{state['p0_loc_old'][0]}, {state['p0_loc_old'][1]}]\n"
                f"Curr. location DM: [{state['p0_loc_new'][0]}, {state['p0_loc_new'][1]}]\n"
                f"{DM_action_string}\n"
                f"DM reward: {state['p0_reward']}\n"
                f"\n"  # This adds a visual separation
                f"Prev. location Adv: [{state['p1_loc_old'][0]}, {state['p1_loc_old'][1]}]\n"
                f"Curr. location Adv: [{state['p1_loc_new'][0]}, {state['p1_loc_new'][1]}]\n"
                f"{Adv_action_string}\n"
                f"Adv reward: {state['p1_reward']}"
            )
        )

        return dm_dot, adv_dot, coin1_dot, coin2_dot, title

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(trajectory_log),
        init_func=init,
        blit=False,
        repeat=False
    )
    ani.save("trajectory.mp4", writer="ffmpeg", fps=fps, dpi=dpi)
    
    # Render and save the last frame as PNG with transparent background
    # update(len(trajectory_log)) 
    # fig.patch.set_alpha(0.0)  # Make background transparent
    # fig.savefig("Coin_game_example.png", dpi=1000, transparent=False, bbox_inches='tight')
    
    plt.close(fig)