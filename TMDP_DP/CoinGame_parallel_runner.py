import subprocess
import os
import logging
import sys

# Directing log messages of level INFO and higher to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,  # Explicitly direct logs to standard output
    force=True
)

def run_parallel_experiments():
    """
    Runs multiple reinforcement learning experiments in parallel using different
    configuration files.
    """
    # --- List your configuration files here ---
    
    config_files = [
        'configs/config_k1_vs_k2.yaml',
        'configs/config_k2_vs_k1.yaml',
        'configs/config_k1_vs_passive_manhattan.yaml',
        'configs/config_k2_vs_aggressive_manhattan.yaml',
        # Add as many config files as you need
    ]

    processes = []

    # --- Launch all experiments as subprocesses ---
    for config_path in config_files:
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found, skipping: {config_path}")
            continue

        # Create a descriptive log file name from the config file name
        config_name = os.path.splitext(os.path.basename(config_path))[0]

        # The command to execute. Using sys.executable ensures we use the same
        # python interpreter that is running this script.
        command = [sys.executable, 'CoinGame_runner.py', config_path]

        # Launch the subprocess
        try:
            process = subprocess.Popen(command)
            processes.append(process)
            logging.info(f"Started experiment for '{config_path}'.")
        except FileNotFoundError:
            logging.error("'CoinGame_runner.py' not found. Make sure this script is in the same directory.")
            # Close the file handle before exiting
            return
        except Exception as e:
            logging.error(f"An unexpected error occurred while launching '{config_path}': {e}")


    logging.info("-" * 50)
    logging.info("All experiments have been launched. Waiting for them to complete...")
    logging.info("-" * 50)

    # --- Wait for all processes to finish ---
    for i, process in enumerate(processes):
        process.wait()  # This blocks until the specific process has finished
        
        config_name = os.path.splitext(os.path.basename(config_files[i]))[0]
        
        if process.returncode == 0:
            logging.info(f"Finished experiment: {config_name} (Exit code: 0)")
        else:
            logging.info(f"FAILED experiment: {config_name} (Exit code: {process.returncode}). Check log for details.")


    logging.info("-" * 50)
    logging.info("All experiments have been completed.")
    logging.info(f"Results are saved in their respective timestamped folders inside 'results/'.")
    logging.info("-" * 50)

if __name__ == "__main__":
    run_parallel_experiments()