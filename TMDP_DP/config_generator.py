"""
Generates a suite of .yaml configuration files for experiments.

This script automates the creation of configuration files based on a set of
pre-defined agent profiles and a list of desired matchups. It uses a base
template for common settings (environment, rewards, plotting) and injects
the specific agent configurations for each player in a given matchup.

This approach minimizes manual configuration, reduces errors, and makes it
easy to modify parameters across all experiments in a centralized way.

Usage:
  # Generate all defined configs into the 'configs/generated/' directory
  python config_generator.py --output-dir configs/generated/
"""

import yaml
from pathlib import Path
import copy
import argparse
from typing import Any
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

# =============================================================================
# 1. DEFINE THE BASE CONFIGURATION
# All settings that are THE SAME across all experiment files.
# =============================================================================
BASE_CONFIG: dict[str, Any] = {
    "experiment_settings": {
        # The 'name' will be generated automatically for each matchup.
        "num_runs": 6,
        "num_episodes": 20000,
        "run_seed": 42,
        "results_dir": "results/article_configs/push_false" # Used only if base output is not defined
                                                            #(as in case of the CoinGame_runner.py and CoinGame_parallel_runner.py)
    },
    "environment_settings": {
        "class": "CoinGame",
        "params": {
            "grid_size": 7,
            "max_steps": 'Inf',
            "enable_push": False,
            "push_distance": 0,
            "action_execution_probabilities": [0.7, 0.7],
            "rewards": {
                "step_penalty": [-0.1, -0.1],
                "out_of_bounds_penalty_delta": [0.0, 0.0],
                "collision_penalty_delta": [0.0, 0.0],
                "push_reward_delta": [0.0, 0.0],
                "push_penalty_delta": [0.0, 0.0],
                "push_but_not_adjacent_penalty_delta": [0.0, 0.0],
                "both_push_penalty_delta": [0.0, 0.0],
                "coin_reward_delta": [0.0, 0.0],
                "coin_steal_penalty_delta": [0.0, 0.0],
                "contested_coin_penalty_delta": [0.0, 0.0],
                "win_reward": [10.0, 10.0],
                "loss_penalty": [-10.0, -10.0],
                "draw_penalty": [0.0, 0.0],
                "timeout_penalty_delta": [0.0, 0.0],
                "timeout_lead_bonus_delta": [0.0, 0.0],
                "timeout_trail_penalty_delta": [0.0, 0.0]
            }
        }
    },
    "plotting_settings": {
        "plot_moving_average_window_size": 500,
        "reward_time_series_x_axis_plot_range": [0, 19999],
        "episode_range_to_eval": [19899, 19999],
        "game_result_ratio_x_axis_plot_range": [19899, 19999]
    }
}

# =============================================================================
# 2. DEFINE THE AGENT PROFILES
# A blueprint for each unique agent class. This is where you would modify
# an agent's parameters (e.g., gamma, learning_rate) for all experiments.
# =============================================================================
AGENT_PROFILES = {
    "IndQLearningAgent": {
        "class": "IndQLearningAgent",
        "params": {
            "gamma": 0.95,
            "learning_rate": 0.7,
            "epsilon": 0.05,
            "initial_Q_value": 0.0
        },
        "epsilon_decay_agent": {"type": "no_decay", "end": 0}
    },
    "LevelKQAgent": {
        "class": "LevelKQAgent",
        "params": {
            "gamma": 0.95,
            "learning_rate": 0.7,
            "epsilon": 0.05,
            "k": 1,
            "lower_level_k_epsilon": 0,
            "initial_Q_value": 0.0
        },
        "epsilon_decay_agent": {"type": "no_decay", "end": 0}
    },
    "LevelK_MDP_DP_Agent_Stationary": {
        "class": "LevelK_MDP_DP_Agent_Stationary",
        "params": {
            "gamma": 0.95,
            "epsilon": 0.05,
            "k": 1,
            "lower_level_k_epsilon": 0,
            "initial_V_value": 0.0
        },
        "epsilon_decay_agent": {"type": "no_decay", "end": 0}
    },
    "LevelK_MDP_DP_Agent_Dynamic": {
        "class": "LevelK_MDP_DP_Agent_Dynamic",
        "params": {
            "gamma": 0.95,
            "epsilon": 0.05,
            "k": 1,
            "lower_level_k_epsilon": 0,
            "initial_V_value": 0.0
        },
        "epsilon_decay_agent": {"type": "no_decay", "end": 0}
    },
    "LevelK_TMDP_DP_Agent_Stationary": {
        "class": "LevelK_TMDP_DP_Agent_Stationary",
        "params": {
            "gamma": 0.95,
            "epsilon": 0.05,
            "k": 1,
            "lower_level_k_epsilon": 0,
            "initial_V_value": 0.0
        },
        "epsilon_decay_agent": {"type": "no_decay", "end": 0}
    },
    "LevelK_TMDP_DP_Agent_Dynamic": {
        "class": "LevelK_TMDP_DP_Agent_Dynamic",
        "params": {
            "gamma": 0.95,
            "epsilon": 0.05,
            "k": 1,
            "lower_level_k_epsilon": 0,
            "initial_V_value": 0.0
        },
        "epsilon_decay_agent": {"type": "no_decay", "end": 0}
    },
    "ManhattanAgent": { # Matchup with this agent is only viable when push is set to True!
        "class": "ManhattanAgent",
        # Heuristic agents might have no parameters defined in the config.
        # An empty 'params' is fine, as the create_agent factory handles it.
    },
    "ManhattanAgent_Passive": {
        "class": "ManhattanAgent_Passive",
    },
    "MDP_DP_Agent_PerfectModel": {
        "class": "MDP_DP_Agent_PerfectModel",
        "params": {
            "gamma": 0.95,
            "initial_V_value": 0.0,
            "termination_criterion": 0.00001,
            "value_iteration_max_num_of_iter": 1000
        }
    },
    "TMDP_DP_Agent_PerfectModel": {
        "class": "TMDP_DP_Agent_PerfectModel",
        "params": {
            "gamma": 0.95,
            "initial_V_value": 0.0,
            "termination_criterion": 0.00001,
            "value_iteration_max_num_of_iter": 1000
        }
    }
}

# =============================================================================
# 3. DEFINE THE EXPERIMENT MATCHUPS
# A list of tuples, where each tuple is (player_1_class, player_2_class).
# The script will generate one .yaml file for each tuple.
# =============================================================================
MATCHUPS = [
    # LevelK_MDP_DP_Dynamic vs. X
    ("LevelK_MDP_DP_Agent_Dynamic", "IndQLearningAgent"),
    ("LevelK_MDP_DP_Agent_Dynamic", "LevelKQAgent"),
    ("LevelK_MDP_DP_Agent_Dynamic", "ManhattanAgent_Passive"),
    # ("LevelK_MDP_DP_Agent_Dynamic", "ManhattanAgent"), # This matchup is only viable when push is set to True!
    ("LevelK_MDP_DP_Agent_Dynamic", "LevelK_MDP_DP_Agent_Dynamic"),

    # LevelK_MDP_DP_Stationary vs. X
    ("LevelK_MDP_DP_Agent_Stationary", "IndQLearningAgent"),
    ("LevelK_MDP_DP_Agent_Stationary", "LevelKQAgent"),
    ("LevelK_MDP_DP_Agent_Stationary", "ManhattanAgent_Passive"),
    # ("LevelK_MDP_DP_Agent_Stationary", "ManhattanAgent"), # This matchup is only viable when push is set to True!
    ("LevelK_MDP_DP_Agent_Stationary", "LevelK_MDP_DP_Agent_Stationary"),
    ("LevelK_MDP_DP_Agent_Stationary", "LevelK_MDP_DP_Agent_Dynamic"),

    # LevelK_TMDP_DP_Dynamic vs. X
    ("LevelK_TMDP_DP_Agent_Dynamic", "IndQLearningAgent"),
    ("LevelK_TMDP_DP_Agent_Dynamic", "LevelKQAgent"),
    ("LevelK_TMDP_DP_Agent_Dynamic", "ManhattanAgent_Passive"),
    # ("LevelK_TMDP_DP_Agent_Dynamic", "ManhattanAgent"), # This matchup is only viable when push is set to True!
    ("LevelK_TMDP_DP_Agent_Dynamic", "LevelK_MDP_DP_Agent_Dynamic"),
    ("LevelK_TMDP_DP_Agent_Dynamic", "LevelK_TMDP_DP_Agent_Dynamic"),
    
    # LevelK_TMDP_DP_Stationary vs. X
    ("LevelK_TMDP_DP_Agent_Stationary", "IndQLearningAgent"),
    ("LevelK_TMDP_DP_Agent_Stationary", "LevelKQAgent"),
    ("LevelK_TMDP_DP_Agent_Stationary", "ManhattanAgent_Passive"),
    # ("LevelK_TMDP_DP_Agent_Stationary", "ManhattanAgent"), # This matchup is only viable when push is set to True!
    ("LevelK_TMDP_DP_Agent_Stationary", "LevelK_MDP_DP_Agent_Stationary"),
    ("LevelK_TMDP_DP_Agent_Stationary", "LevelK_TMDP_DP_Agent_Dynamic"),
    ("LevelK_TMDP_DP_Agent_Stationary", "LevelK_TMDP_DP_Agent_Stationary"),
    
    # LevelKQAgent vs. X
    ("LevelKQAgent", "IndQLearningAgent"),

    # Offline Solvers vs. Heuristics
    ("MDP_DP_Agent_PerfectModel", "ManhattanAgent_Passive"),
    # ("MDP_DP_Agent_PerfectModel", "ManhattanAgent"), # This matchup is only viable when push is set to True!
    ("TMDP_DP_Agent_PerfectModel", "ManhattanAgent_Passive"),
    # ("TMDP_DP_Agent_PerfectModel", "ManhattanAgent"), # This matchup is only viable when push is set to True!
]

def generate_configs(output_dir: Path):
    """
    Generates and saves .yaml config files for all defined matchups.

    Args:
        output_dir (Path): The directory where the generated .yaml files will be saved.
    """
    logging.info(f"Generating configuration files in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_count = 0
    for p1_name, p2_name in MATCHUPS:
        # Start with a clean copy of the base configuration
        # deepcopy is crucial to prevent modifications from affecting subsequent loops
        config = copy.deepcopy(BASE_CONFIG)

        # Generate the unique experiment name
        experiment_name = f"{p1_name}_vs_{p2_name}"
        config["experiment_settings"]["name"] = experiment_name
        
        # Add a descriptive comment
        comment = f"Auto-generated config for {p1_name} vs. {p2_name}."
        if "comment" not in config:
            config["comment"] = comment
        
        # Look up and inject the agent profiles
        try:
            player_1_profile = AGENT_PROFILES[p1_name]
            player_2_profile = AGENT_PROFILES[p2_name]
        except KeyError as e:
            logging.error(f"Agent profile '{e.args[0]}' not found in AGENT_PROFILES. Skipping this matchup.")
            continue

        config["agent_settings"] = {
            "player_1": player_1_profile,
            "player_2": player_2_profile
        }

        # Construct the output file path and save the file
        output_filename = f"config_{experiment_name}.yaml"
        output_path = output_dir / output_filename

        with open(output_path, 'w') as f:
            # sort_keys=False preserves the nice ordering from our dictionaries
            yaml.dump(config, f, sort_keys=False, indent=2)
        
        generated_count += 1
        logging.info(f"  -> Saved {output_filename}")
        
    logging.info(f"\nSuccessfully generated {generated_count} configuration files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate .yaml configuration files for a suite of experiments."
    )
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="The directory where the generated .yaml files will be saved."
    )
    args = parser.parse_args()
    
    generate_configs(args.output_dir)