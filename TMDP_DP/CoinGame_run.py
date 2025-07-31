import logging
import sys

# Directing log messages of level INFO and higher to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,  # Explicitly direct logs to standard output
    force=True
)

from experiment_runner import run_experiment


if __name__ == "__main__":
    # Use the config file specified from the command line, 
    # or default to 'configs/config.yaml'
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    else:
        config_file_path = 'configs/config.yaml'
    
    # Set log_trajectory to True if you want to save animations later
    results_path = run_experiment(config_file_path, log_trajectory=True)
    
    print("="*50)
    print("Experiment finished successfully.")
    print(f"Results have been saved in: {results_path}")
    print("="*50)