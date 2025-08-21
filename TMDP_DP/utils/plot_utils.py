import numpy as np

# Also ffmpeg is required for video saving
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import mplcursors

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

def plot_reward_per_episode_series(reward_series_p0, reward_series_p1, plot_title, moving_average_window_size, episode_series_x_axis_plot_range, dir=None, plot_bands=False):
    """
    Plot smoothed reward trajectories for two agents over multiple experiments.

    Parameters:
        reward_series_p0 (list of arrays): Rewards for Agent A across experiments.
        reward_series_p1 (list of arrays): Rewards for Agent B across experiments.
        plot_title (str): Title of the plot.
        moving_average_window_size(int): Size of right-aligned moving average window.
        episode_series_x_axis_plot_range (list/tuple): A list [xmin, xmax] to set the x-axis view.
        dir (str, optional): If provided, saves the plot to 'dir.png'.
        plot_bands (bool, optional): If True, plots a 95% confidence interval band instead of individual runs. Defaults to False.
    """
    # Apply 'ggplot' style
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    number_of_experiment = len(reward_series_p0)
    
    # Common x-axis for all plots
    x_axis = np.arange(moving_average_window_size - 1, len(reward_series_p0[0]))
    
    # Store smoothed series for all runs to calculate bands later
    smoothed_p0_series = []
    smoothed_p1_series = []

    for i in range(number_of_experiment):
        # Calculate the moving average for the full series
        p0_reward_series = moving_average(reward_series_p0[i], moving_average_window_size)
        p1_reward_series = moving_average(reward_series_p1[i], moving_average_window_size)
        
        smoothed_p0_series.append(p0_reward_series)
        smoothed_p1_series.append(p1_reward_series)

        if not plot_bands:
            # If not plotting bands, plot individual low-opacity lines
            ax.plot(x_axis, p0_reward_series, 'b', alpha=0.05)
            ax.plot(x_axis, p1_reward_series, 'r', alpha=0.05)

    if plot_bands:
        # Calculate percentiles on the smoothed data
        lower_bound_p0 = np.percentile(smoothed_p0_series, 5, axis=0)
        upper_bound_p0 = np.percentile(smoothed_p0_series, 95, axis=0)
        ax.fill_between(x_axis, lower_bound_p0, upper_bound_p0, color='b', alpha=0.2)
        
        # Plot invisible lines for the cursor
        ax.plot(x_axis, lower_bound_p0, color='b', alpha=0)
        ax.plot(x_axis, upper_bound_p0, color='b', alpha=0)
        
        lower_bound_p1 = np.percentile(smoothed_p1_series, 5, axis=0)
        upper_bound_p1 = np.percentile(smoothed_p1_series, 95, axis=0)
        
        # Plot invisible lines for the cursor
        ax.plot(x_axis, lower_bound_p1, color='r', alpha=0)
        ax.plot(x_axis, upper_bound_p1, color='r', alpha=0)
        
        # Plot the shaded confidence interval band
        ax.fill_between(x_axis, lower_bound_p1, upper_bound_p1, color='r', alpha=0.2)
        
    # Calculate and plot the mean of the smoothed series
    p0_average_reward_series = np.mean(smoothed_p0_series, axis=0)
    p1_average_reward_series = np.mean(smoothed_p1_series, axis=0)
    
    ax.plot(x_axis, p0_average_reward_series, 'b', alpha=0.8, linewidth=1.5, label='DM Mean')
    ax.plot(x_axis, p1_average_reward_series, 'r', alpha=0.8, linewidth=1.5, label='Adversary Mean')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative reward per episode')
    ax.set_title(plot_title)
    ax.set_xlim(episode_series_x_axis_plot_range)
    
    # Force the x-axis to use integer ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.legend()

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
        
    # Enable hover tooltips
    cursor = mplcursors.cursor(ax.lines, hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        x, y = sel.target
        label = sel.artist.get_label()
        sel.annotation.set_text(f"{label}\nEpisode={int(round(float(x)))}\nReward={float(y):.4f}")

    plt.show()
        
def plot_result_ration(result_series, episode_range_to_eval, plot_title, result_type_to_plot, episode_series_x_axis_plot_range, dir=None, plot_bands=False):
    """
    Plot result ration trajectories for an agent over multiple experiments.

    Parameters:
        result_series (list of arrays): A list where each element is a 1D numpy array
                                        of game results for one experiment run.
        episode_range_to_eval(list/tuple): A list or tuple with range for which to calculate the running game result ratio.
        plot_title (str): Title of the plot.
        result_type_to_plot (str): One of 'win', 'loss', 'draw', or 'timeout'.
        episode_series_x_axis_plot_range (list/tuple): A list [xmin, xmax] to set the x-axis view.
        dir (str, optional): If provided, saves the plot to 'dir.png'.
        plot_bands (bool, optional): If True, plots a 95% confidence interval band instead of individual runs. Defaults to False.
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

    result_series_subset = [result[episode_range_to_eval[0]:episode_range_to_eval[1]] for result in result_series]
    
    num_experiments = len(result_series_subset)
    num_episodes = len(result_series_subset[0])

    # Create an array of episode numbers [1, 2, ..., N] for the ratio calculation
    episode_indices = np.arange(1, num_episodes + 1)
    
    # For plotting labels only
    plot_x_indices = episode_indices + episode_range_to_eval[0]
    
    # Store the calculated ratios for each experiment to compute the mean later
    all_ratios = []

    for i in range(num_experiments):
        # Create a binary array: 1 if the result matches the type, 0 otherwise
        binary_results = (np.array(result_series_subset[i]) == result_code).astype(int)
        
        # Calculate the cumulative sum of the specific result
        cumulative_results = np.cumsum(binary_results)
        
        # Calculate the ratio of that result type over the episodes
        result_ratio = cumulative_results / episode_indices
        all_ratios.append(result_ratio)
        
        if not plot_bands:
            # Plot the ratio for the individual run with high transparency
            ax.plot(plot_x_indices, result_ratio, 'b', alpha=0.05)

    # Calculate the mean ratio across all experiments
    mean_ratio_series = np.mean(all_ratios, axis=0)
    
    # Plot the averaged line with lower transparency
    ax.plot(plot_x_indices, mean_ratio_series, 'b', alpha=0.8, linewidth=1.5, label=f'DM Mean {result_type_to_plot.capitalize()} Ratio')

    if plot_bands:
        # Calculate the 5th and 95th percentiles for the confidence interval
        lower_bound = np.percentile(all_ratios, 5, axis=0)
        upper_bound = np.percentile(all_ratios, 95, axis=0)
        
        # Plot invisible lines for the cursor to attach to
        ax.plot(plot_x_indices, lower_bound, color='b', alpha=0)
        ax.plot(plot_x_indices, upper_bound, color='b', alpha=0)
        
        # Plot the shaded confidence interval band
        ax.fill_between(plot_x_indices, lower_bound, upper_bound, color='b', alpha=0.2)

    ax.set_xlabel('Episode')
    ax.set_ylabel(f'{result_type_to_plot.capitalize()} Ratio')
    ax.set_title(plot_title)
    ax.set_xlim(episode_series_x_axis_plot_range)
    
    # Force the x-axis to use integer ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Simplified legend for a single agent's metric
    # custom_lines = [Line2D([0], [0], color='b', label=f'DM {result_type_to_plot} ratio')]
    # ax.legend(handles=custom_lines)
    
    ax.legend()

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
        
    # Enable interactive hover if running live
    cursor = mplcursors.cursor(ax.lines, hover=True)
    @cursor.connect("add")
    def _on_add(sel):
        x, y = sel.target
        label = sel.artist.get_label()
        sel.annotation.set_text(f"Episode={int(round(x))}\nRatio={y:.4f}")

    plt.show()
        
def animate_trajectory_from_log(trajectory_episode_array, grid_size=4, fps=4, dpi=100):
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
    push_action_codes = ["No push", "Push"]

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
        info_text.set_text("")
        return dm_dot, adv_dot, coin1_dot, coin2_dot, title, info_text

    def update(frame):
        state = trajectory_episode_array[frame]

        dm_col, dm_row = state['p0_loc_new_col'], state['p0_loc_new_row']
        adv_col, adv_row = state['p1_loc_new_col'], state['p1_loc_new_row']
        
        dm_dot.set_data([dm_col], [dm_row])
        adv_dot.set_data([adv_col], [adv_row])

        # Check for if coin location is non None
        if state['coin1_row'] != -1:
            coin1_dot.set_data([state['coin1_col']], [state['coin1_row']])
        else:
            coin1_dot.set_data([-10], [-10]) # hide

        if state['coin2_row'] != -1:
            coin2_dot.set_data([state['coin2_col']], [state['coin2_row']])
        else:
            coin2_dot.set_data([-10], [-10]) # hide

        exp_num = int(state['experiment_num'])
        ep_num = int(state['episode_num'])
        step_num = int(state['step_num'])
        title.set_text(f"Exp {exp_num + 1} | Ep {ep_num + 1} | Step {step_num}")
        
        # Handle sentinel values for actions/locations
        p0_old_row, p0_old_col = int(state['p0_loc_old_row']), int(state['p0_loc_old_col'])
        p1_old_row, p1_old_col = int(state['p1_loc_old_row']), int(state['p1_loc_old_col'])
        p0_loc_old_str = f"[{p0_old_col}, {p0_old_row}]" if p0_old_row != -1 else "None"
        p1_loc_old_str = f"[{p1_old_col}, {p1_old_row}]" if p1_old_row != -1 else "None"

        p0_move, p0_push = int(state['p0_action_move']), int(state['p0_action_push'])
        p1_move, p1_push = int(state['p1_action_move']), int(state['p1_action_push'])
        p0_action_str = f"DM Action: [{move_action_codes[p0_move]}, {push_action_codes[p0_push]}]" if p0_move != -1 else "DM Action: None"
        p1_action_str = f"Adv Action: [{move_action_codes[p1_move]}, {push_action_codes[p1_push]}]" if p1_move != -1 else "Adv Action: None"
        
        p0_reward = state['p0_reward']
        p1_reward = state['p1_reward']
        p0_reward_str = f"DM reward: {p0_reward:.3f}" if p0_reward != -1 else "DM reward: None"
        p1_reward_str = f"Adv reward: {p1_reward:.3f}" if p0_reward != -1 else "Adv reward: None"
        
        p0_cum_reward = state['p0_cum_reward']
        p1_cum_reward = state['p1_cum_reward']
        p0_cum_reward_str = f"DM cumulative reward: {p0_cum_reward:.3f}" if p0_cum_reward != -1 else "DM reward: None"
        p1_cum_reward_str = f"Adv cumulative reward: {p1_cum_reward:.3f}" if p1_cum_reward != -1 else "Adv reward: None"
        
        # Update a text box the grid for step and reward info
        info_text.set_text(
            f"Prev. location DM: {p0_loc_old_str}\n"
            f"Curr. location DM: [{int(dm_col)}, {int(dm_row)}]\n"
            f"{p0_action_str}\n"
            f"{p0_reward_str}\n"
            f"{p0_cum_reward_str}\n"
            f"\n"
            f"Prev. location Adv: {p1_loc_old_str}\n"
            f"Curr. location Adv: [{int(adv_col)}, {int(adv_row)}]\n"
            f"{p1_action_str}\n"
            f"{p1_reward_str}\n"
            f"{p1_cum_reward_str}\n"
        )

        return dm_dot, adv_dot, coin1_dot, coin2_dot, title, info_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=trajectory_episode_array.shape[0],
        init_func=init,
        blit=False, # Blit can be tricky with text, False is safer
        repeat=False
    )
    ani.save("trajectory.mp4", writer="ffmpeg", fps=fps, dpi=dpi)
    
    # Render and save the last frame as PNG with transparent background
    # update(len(trajectory_episode_array)) 
    # fig.patch.set_alpha(0.0)  # Make background transparent
    # fig.savefig("Coin_game_example.png", dpi=1000, transparent=False, bbox_inches='tight')
    
    plt.close(fig)