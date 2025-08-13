import subprocess
import os
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PARALLEL_RUNNER] - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

CONFIG_DIR = Path("/Users/ruzejjur/Github/ARAMARL_TMDP_DP/TMDP_DP/configs/article_configs/push_true")
DEFAULT_JOBS = max(1, (os.cpu_count() or 1))  # sensible default

def run_one(config_path: str) -> tuple[str, int]:
    """Run a single config to completion; return (name, exitcode)."""
    cfg_name = Path(config_path).stem
    
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{cfg_name}.out"
    stderr_path = log_dir / f"{cfg_name}.err"

    cmd = [sys.executable, "CoinGame_runner.py", config_path]
    logging.info(f"Starting: {cfg_name}")
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err)
        rc = proc.wait()
    return cfg_name, rc

def run_parallel_experiments(jobs: int | None = None, recursive=False):
    jobs = jobs or DEFAULT_JOBS
    if recursive:
        config_files = sorted(str(p.resolve()) for p in CONFIG_DIR.rglob("*.yaml"))
    else:
        config_files = sorted(str(p.resolve()) for p in CONFIG_DIR.glob("*.yaml"))

    if not config_files:
        logging.error(f"No .yaml configs found in: {CONFIG_DIR}")
        return

    jobs = max(1, min(jobs, len(config_files)))  # don’t spawn more workers than tasks
    logging.info(f"Discovered {len(config_files)} configs. Running with jobs={jobs}.")

    # Run with at most `jobs` concurrent processes
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_one, cfg): cfg for cfg in config_files}
        for fut in as_completed(futs):
            cfg = futs[fut]
            try:
                name, rc = fut.result()
                if rc == 0:
                    logging.info(f"Finished: {name} (0)")
                    ok += 1
                else:
                    logging.error(f"FAILED: {name} (exit {rc}) – see results/logs/{name}.err")
                    fail += 1
            except Exception as e:
                logging.exception(f"Exception while running {cfg}: {e}")
                fail += 1

    logging.info(f"Done. Success: {ok}, Failed: {fail}")

if __name__ == "__main__":
    # Optionally: read JOBS from env or CLI arg
    jobs_env = os.getenv("JOBS")
    jobs = int(jobs_env) if jobs_env and jobs_env.isdigit() else None
    run_parallel_experiments(jobs=jobs, recursive=False)