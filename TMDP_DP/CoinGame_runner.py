import logging
import sys

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
    # Use the config file specified from the command line, 
    # or default to 'configs/config.yaml'
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    else:
        raise ValueError("Config path was not provided.")
    
    # Set log_trajectory to True if you want to save animations later
    results_path = run_experiment(config_file_path, log_trajectory=True)
    
    print("="*50)
    print("Experiment finished successfully.")
    print(f"Results have been saved in: {results_path}")
    print("="*50)