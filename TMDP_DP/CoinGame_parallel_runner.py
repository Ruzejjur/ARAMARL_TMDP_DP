import subprocess
import os
import sys
from datetime import datetime

def run_parallel_experiments():
    """
    Runs multiple reinforcement learning experiments in parallel using different
    configuration files.

    Each experiment is launched as a separate process, and its output is logged
    to a unique file for easier monitoring and debugging.
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
            print(f" Warning: Config file not found, skipping: {config_path}")
            continue

        # Create a descriptive log file name from the config file name
        config_name = os.path.splitext(os.path.basename(config_path))[0]

        # The command to execute. Using sys.executable ensures we use the same
        # python interpreter that is running this script.
        command = [sys.executable, 'CoinGame_run.py', config_path]

        # Launch the subprocess
        try:
            process = subprocess.Popen(command)
            processes.append(process)
            print(f"Started experiment for '{config_path}'.")
        except FileNotFoundError:
            print("Error: 'CoinGame_run.py' not found. Make sure this script is in the same directory.")
            # Close the file handle before exiting
            return
        except Exception as e:
            print(f"An unexpected error occurred while launching '{config_path}': {e}")


    print("-" * 50)
    print("All experiments have been launched. Waiting for them to complete...")
    print("-" * 50)

    # --- Wait for all processes to finish ---
    for i, process in enumerate(processes):
        process.wait()  # This blocks until the specific process has finished
        
        config_name = os.path.splitext(os.path.basename(config_files[i]))[0]
        
        if process.returncode == 0:
            print(f"Finished experiment: {config_name} (Exit code: 0)")
        else:
            print(f"FAILED experiment: {config_name} (Exit code: {process.returncode}). Check log for details.")


    print("-" * 50)
    print("All experiments have been completed.")
    print(f"Results are saved in their respective timestamped folders inside 'results/'.")
    print("-" * 50)

if __name__ == "__main__":
    run_parallel_experiments()