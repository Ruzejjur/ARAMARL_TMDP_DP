"""
Executes a single experiment based on a specified configuration file.

This script is the "worker" in the experimental setup. It is typically
invoked by a parallel runner script (like 'CoinGame_parallel_runner.py'),
but can also be run directly for testing a single configuration.

It takes a path to a .yaml config file and an optional flag to control
whether detailed trajectory data should be logged.

Usage Examples:
  # Run a single experiment with default settings (trajectory logging ON)
  python CoinGame_runner.py ./configs/my_experiment.yaml

  # Run a single experiment and disable trajectory logging
  python CoinGame_runner.py ./configs/my_experiment.yaml --no-log-trajectory
"""


import logging
import sys
from pathlib import Path
import argparse

from experiment_runner import run_experiment

# This configures the root logger for immediate console output.
# The 'force=True' argument is crucial to ensure this configuration is applied,
# even if other modules (like in a Jupyter environment) have already touched the logging system.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run a single experiment using a specified YAML configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help message
    )
    
    # Define the required configuration file argument
    parser.add_argument(
        "config_file", 
        type=Path,
        help="The path to the .yaml configuration file for the experiment."
    )

    # Define results directory from the parallel runner argument
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="The base directory where final experiment results should be saved."
    )
    
    # Define an optional flag to control trajectory logging
    parser.add_argument(
        "--no-log-trajectory",
        action="store_false",
        dest="log_trajectory",
        help="Disable trajectory logging. By default, trajectories are logged."
    )
    # Set the default value for the destination 'log_trajectory'
    parser.set_defaults(log_trajectory=True)

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    results_path = run_experiment(
        config_file_path=str(args.config_file), 
        log_trajectory=args.log_trajectory,
        base_output_dir=args.result_dir
    )
    print("="*50)
    print("Experiment finished successfully.")
    print(f"Results have been saved in: {results_path}")
    print("="*50)