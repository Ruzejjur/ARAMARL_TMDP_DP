
"""
Orchestrates running multiple experiments defined by .yaml files in parallel.

This script acts as a master controller. It discovers experiment configuration
files in a specified directory and then spawns multiple "worker" processes,
each running the 'CoinGame_runner.py' script with a different configuration.

It uses a thread pool to manage concurrency, captures the standard output and
error for each individual run into log files, and provides a final summary of
successful and failed jobs.

Usage Examples:
  # Run all configs in a directory using default CPU count
  python CoinGame_parallel_runner.py ./path/to/configs -o ./my_run_results

  # Run recursively and save results to a custom output directory
  python CoinGame_parallel_runner.py ./path/to/configs -r -o ./my_run_results

  # Run with a specific number of parallel jobs (e.g., 8)
  python CoinGame_parallel_runner.py ./path/to/configs --jobs 8
"""

import subprocess
import os
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent

def setup_logging(run_directory: Path):
    """Configures the root logger to output to console and a file."""
    
    # Define the log file path.
    log_file = run_directory / "parallel_runner.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level of messages to handle.

    # Clear any existing handlers to avoid duplicate logs.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter to define the log message format.
    formatter = logging.Formatter('%(asctime)s - [PARALLEL_RUNNER] - %(levelname)s - %(message)s')

    # Create a handler to write to the console (stdout).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a handler to write to the log file.
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def run_one(config_path: Path, run_directory: Path) -> tuple[str, int]:
    """
    Executes a single experiment subprocess and captures its output.

    This function is the "worker" task that is submitted to the thread pool.
    It takes a single configuration file, runs 'CoinGame_runner.py' in a
    separate process, and waits for it to complete.

    Args:
        config_path: Path to the .yaml configuration file.
        run_directory: The unique top-level directory for this entire parallel run.

    Returns:
        A tuple containing:
            - str: The clean name of the configuration (e.g., 'config_A').
            - int: The exit code of the subprocess (0 for success).

    Side Effects:
        - Creates the log directory if it doesn't exist.
        - Writes the standard output and error of the subprocess to
          '{config_name}.out' and '{config_name}.err' files, respectively.
    """
    # Extract the final part of the path without extension
    config_name = Path(config_path).stem
    
    subprocess_log_dir = run_directory / "run_logs"
    subprocess_log_dir.mkdir(parents=True, exist_ok=True)
    
    stdout_path = subprocess_log_dir / f"{config_name}.out"
    stderr_path = subprocess_log_dir / f"{config_name}.err"

    worker_script_path = SCRIPT_DIR / "CoinGame_runner.py"
    
    # Define path for individual run results
    individual_run_result_dir = run_directory / "run_results"
    
    # Run single experiment  
    cmd = [
        sys.executable, 
        str(worker_script_path), 
        str(config_path), 
        "--result-dir", 
        str(individual_run_result_dir)
    ]
    
    logging.info(f"Starting: {config_name}")
    
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err)
        return_code = proc.wait()
        
    return config_name, return_code

def run_parallel_experiments(config_directory: Path, run_directory: Path, jobs: int | None = None, recursive=False):
    """
    Discovers and runs all experiments in parallel using a thread pool.

    Args:
        config_directory: Path to the directory containing .yaml config files.
        run_directory: Path to the top-level directory for saving results.
        jobs: The maximum number of experiments to run at once. If None,
              it defaults to the system's CPU count.
        recursive: If True, search for .yaml files in subdirectories as well.
    """
    
    # Set default number of jobs to number of threads
    # NOTE: os.cup_count() may return None in some specific cases, this is why we add  (os.cpu_count() or 1)
    default_jobs = max(1, (os.cpu_count() or 1))
    
    jobs = jobs or default_jobs
    
    # Load (possibly recursively) all config_file_path files in a directory
    if recursive:
        config_file_paths = sorted(p.resolve() for p in config_directory.rglob("*.yaml"))
    else:
        config_file_paths = sorted(p.resolve() for p in config_directory.glob("*.yaml"))

    if not config_file_paths:
        logging.error(f"No .yaml configs found in: {config_directory}")
        return

    # Don’t spawn more workers than tasks
    jobs = max(1, min(jobs, len(config_file_paths)))  
    
    logging.info(f"Discovered {len(config_file_paths)} configs. Running with jobs={jobs}.")

    # Run with at most `jobs` concurrent processes
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_one, config_file_path, run_directory): config_file_path for config_file_path in config_file_paths}
        
        for future in as_completed(futures):
            config_file_path = futures[future]
            try:
                name, return_code = future.result()
                if return_code == 0:
                    logging.info(f"FINISHED: {name} (0)")
                    ok += 1
                else:
                    log_file_path = run_directory / "run_logs" / f"{name}.err"
                    logging.error(f"FAILED: {name} (exit {return_code}) – see {log_file_path}.")
                    fail += 1
                    
            except Exception as e:
                logging.exception(f"Exception while running {config_file_path}: {e}")
                fail += 1

    logging.info(f"Done. Success: {ok}, Failed: {fail}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run a series of experiments from .yaml configuration files in parallel."
    )
    parser.add_argument(
        "config_directory", 
        type=Path,
        help="The path to the directory containing the .yaml configuration files."
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("results"), # Sensible default
        help="Base directory to save the unique run folder."
    )
    
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        help="Number of parallel jobs to run. Defaults to system CPU count or JOBS env var."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="If set, search for .yaml files in subdirectories recursively."
    )
    args = parser.parse_args()
    
    # Generate unique run directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_run_name = f"parallel_run_{timestamp}"
    unique_run_dir = args.output / unique_run_name

    
    # Call our setup function with the output directory provided by the user.
    setup_logging(unique_run_dir)

    # Logic to determine the number of jobs
    num_jobs = args.jobs  # Prioritize the command-line flag
    if num_jobs is None:
        jobs_env = os.getenv("JOBS")
        if jobs_env and jobs_env.isdigit():
            num_jobs = int(jobs_env)
            
    run_parallel_experiments(
        config_directory=args.config_directory, 
        run_directory=unique_run_dir, 
        jobs=num_jobs, 
        recursive=args.recursive
    )