import sys
from experiment_runner import load_config, run_experiment

if __name__ == "__main__":
    # Use the config file specified from the command line, 
    # or default to 'configs/config.yaml'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'configs/config.yaml'

    print(f"Loading configuration from: {config_file}")
    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_file}'")
        sys.exit(1)

    print("Configuration loaded successfully. Starting experiment...")
    
    # Set log_trajectory to True if you want to save animations later
    results_path = run_experiment(config, log_trajectory=True)
    
    print("="*50)
    print("Experiment finished successfully.")
    print(f"Results have been saved in: {results_path}")
    print("="*50)