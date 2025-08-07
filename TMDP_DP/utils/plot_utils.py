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

def plot_reward_per_episode_series(r0ss, r1ss, plot_title, moving_average_window_size=1000, dir=None):
    """
    Plot smoothed reward trajectories for two agents over multiple experiments.

    Parameters:
        r0ss (list of arrays): Rewards for Agent A across experiments.
        r1ss (list of arrays): Rewards for Agent B across experiments.
        dir (str, optional): If provided, saves the plot to 'dir.png'.

    Returns:
        None
    """
    # Apply 'ggplot' style
    plt.style.use('ggplot')
    # Create figure and axis explicitly d
    fig, ax = plt.subplots()

    N_EXP = len(r0ss)

    for i in range(N_EXP):
        ax.plot(moving_average(r0ss[i], moving_average_window_size), 'b', alpha=0.05)
        ax.plot(moving_average(r1ss[i], moving_average_window_size), 'r', alpha=0.05)

    ax.plot(moving_average(np.mean(r0ss, axis=0), moving_average_window_size), 'b', alpha=0.5)
    ax.plot(moving_average(np.mean(r1ss, axis=0), moving_average_window_size), 'r', alpha=0.5)

    ax.set_xlabel('episode')
    ax.set_ylabel('Cumulative reward per episode')
    ax.set_title(plot_title)

    custom_lines = [Line2D([0], [0], color='b', label='DM'),
                    Line2D([0], [0], color='r', label='Adversary')]
    ax.legend(handles=custom_lines)

    if dir is not None:
        # Set figure background to transparent
        fig.patch.set_alpha(0.0)
        fig.savefig(f"{dir}.png", transparent=False, bbox_inches='tight')
        
        
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