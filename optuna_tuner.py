import os
import re
import subprocess
import optuna
import time
from multiprocessing import Process, Queue
from typing import List

# --- CONFIGURATION FOR PARALLEL EXECUTION ON A SINGLE GPU ---
N_PARALLEL_TRIALS = 2
GPU_IDS: List[int] = [0, 0]
TOTAL_TRIALS_TO_RUN = 50 # Total trials to run

def run_trial(trial: optuna.trial.Trial, gpu_id: int) -> float:
    """
    This function runs a single training process for one Optuna trial on a specific GPU.
    """
    lr = trial.params["lr"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python",
        "main.py",
        "--config=gmappo_plus",
        "--env-config=replenishment",
        "with",
        f"lr={lr}",
        "t_max=20000",
        "use_cuda=True",
        "debug_mode=False" # This is still correct to keep logs clean
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"--- TRIAL {trial.number} (GPU {gpu_id}) FAILED ---")
        print(result.stderr)
        raise optuna.exceptions.TrialPruned()

    # --- CORRECTED REGEX ---
    # Look for the "new test result : " line printed when the model improves.
    matches = re.findall(r"new test result : ([\-\d\.]+)", result.stdout)

    if matches:
        # The script might print this line multiple times if the model keeps improving.
        # We take the last (and best) one.
        scores = [float(m) for m in matches]
        best_score = scores[-1]
        print(f"--- TRIAL {trial.number} (GPU {gpu_id}) SUCCESS --- Scores: {scores}, Best: {best_score}")
        return best_score
    else:
        # If the model never improves, it never prints the key.
        # We can treat this as a failed trial and prune it.
        print(f"--- TRIAL {trial.number} (GPU {gpu_id}) NO IMPROVEMENT ---")
        print("Could not find 'new test result :'. Pruning.")
        raise optuna.exceptions.TrialPruned()

def objective_worker(study: optuna.study.Study, gpu_queue: Queue):
    """
    A worker process that runs trials.
    """
    gpu_id = gpu_queue.get()
    while True:
        try:
            # Ask the study for a new trial with suggested hyperparameters
            trial = study.ask({
                "lr": optuna.distributions.LogUniformDistribution(1e-6, 1e-4)
            })
        except Exception:
            break
        try:
            result = run_trial(trial, gpu_id)
            study.tell(trial, result)
        except optuna.exceptions.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as e:
            print(f"An unexpected error occurred in trial {trial.number}: {e}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
    gpu_queue.put(gpu_id)

if __name__ == "__main__":
    # Correctly create the storage and study for parallel execution
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(
        storage=storage,
        study_name="gmappo_tuning_parallel",
        direction="maximize"
    )

    gpu_queue = Queue()
    for gpu_id in GPU_IDS:
        gpu_queue.put(gpu_id)

    processes = []
    print(f" Starting {N_PARALLEL_TRIALS} parallel trials on GPUs: {GPU_IDS}")

    for _ in range(N_PARALLEL_TRIALS):
        p = Process(target=objective_worker, args=(study, gpu_queue))
        p.start()
        processes.append(p)

    # Monitor progress until all trials are accounted for
    while study.n_trials < TOTAL_TRIALS_TO_RUN:
        time.sleep(1)
        print(f"Completed {study.n_trials}/{TOTAL_TRIALS_TO_RUN} trials...", end="\r")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print(f"\n\n--- STUDY COMPLETE ---")
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        print("No trials completed successfully.")
    else:
        best_trial = study.best_trial
        print("\n--- BEST TRIAL ---")
        print(f"  Value (Best test score): {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")