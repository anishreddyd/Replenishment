import os
import re
import subprocess
import time
from multiprocessing import Process, Queue, Lock
import optuna
from typing import List

# --- CONFIGURATION ---
N_PARALLEL_TRIALS = 2  # Number of trials to run in parallel
GPU_IDS: List[int] = [0, 0]  # List of GPU IDs to use for the parallel trials
TOTAL_TRIALS_TO_RUN = 10  # Total number of trials to run for the entire study

# --- LOCK FOR PRINTING TO PREVENT GARBLED OUTPUT ---
print_lock = Lock()


def run_trial(trial: optuna.trial.Trial, gpu_id: int) -> float:
    """
    Runs a single training process for one Optuna trial on a specific GPU.
    """
    lr = trial.params["lr"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with print_lock:
        print(f"🚀 Starting Trial {trial.number} on GPU {gpu_id} with lr={lr:.6e}...")

    cmd = [
        "python",
        "main.py",
        "--config=gmappo_plus",
        "--env-config=replenishment",
        "with",
        f"lr={lr}",
        "t_max=1000000",  # Using a reasonable number of timesteps
        "use_cuda=True",
        "debug_mode=False"  # Keep logs clean for production runs
    ]

    # Execute the training script
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Check if the subprocess crashed
    if result.returncode != 0:
        with print_lock:
            print(f"--- ❌ TRIAL {trial.number} (GPU {gpu_id}) FAILED (Exit Code: {result.returncode}) ---")
            print("--- STDERR ---")
            print(result.stderr)
            print("---------------------------------------------------------")
        raise optuna.exceptions.TrialPruned()

    # Regex to find all instances of the test result log
    matches = re.findall(r"new test result : ([\-\d\.]+)", result.stdout)

    if matches:
        # The script might print this line multiple times. We take the last (best) one.
        scores = [float(m) for m in matches]
        best_score = scores[-1]
        with print_lock:
            print(f"--- ✅ TRIAL {trial.number} (GPU {gpu_id}) SUCCESS --- Final Score: {best_score}")
        return best_score
    else:
        # If the key is never found, the model likely didn't improve. Prune the trial.
        with print_lock:
            print(f"--- ⚠️ TRIAL {trial.number} (GPU {gpu_id}) NO IMPROVEMENT ---")
            print("Could not find 'new test result :'. Pruning.")
            # Optional: Print stdout for debugging mismatches
            # print("--- STDOUT ---")
            # print(result.stdout)
            print("---------------------------------------------------------")
        raise optuna.exceptions.TrialPruned()


def objective_worker(study: optuna.study.Study, gpu_queue: Queue, trials_processed_queue: Queue):
    """
    A worker process that continuously fetches a GPU, asks for a trial, runs it, and reports back.
    """
    while True:
        # Wait for an available GPU
        gpu_id = gpu_queue.get()
        if gpu_id is None:  # Sentinel value to stop the worker
            break

        try:
            # FIX: Use the new FloatDistribution for logarithmic sampling
            trial = study.ask({"lr": optuna.distributions.FloatDistribution(1e-6, 1e-4, log=True)})
        except Exception:
            # If asking for a trial fails, the study might be over.
            gpu_queue.put(gpu_id)
            break

        try:
            result = run_trial(trial, gpu_id)
            study.tell(trial, result)
        except optuna.exceptions.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as e:
            with print_lock:
                print(f"An unexpected error occurred in trial {trial.number}: {e}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        finally:
            # Return the GPU to the queue for another worker to use
            gpu_queue.put(gpu_id)
            trials_processed_queue.put(1)  # Signal that one trial is done


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="gmappo_tuning_parallel",
        direction="maximize"
    )

    gpu_queue = Queue()
    for gpu_id in GPU_IDS:
        gpu_queue.put(gpu_id)

    trials_processed_queue = Queue()
    processes = []

    print(f"🔬 Starting study with {N_PARALLEL_TRIALS} parallel workers on GPUs: {GPU_IDS}")

    for _ in range(N_PARALLEL_TRIALS):
        p = Process(target=objective_worker, args=(study, gpu_queue, trials_processed_queue))
        p.start()
        processes.append(p)

    # Monitor progress until the target number of trials is reached
    completed_trials = 0
    while completed_trials < TOTAL_TRIALS_TO_RUN:
        trials_processed_queue.get()  # Wait for a trial to finish
        completed_trials += 1
        print(f"📈 Progress: {completed_trials}/{TOTAL_TRIALS_TO_RUN} trials completed...", end="\r")

    # Stop all worker processes
    for _ in range(N_PARALLEL_TRIALS):
        gpu_queue.put(None)  # Sentinel value to signal workers to stop

    for p in processes:
        p.join()

    print(f"\n\n--- 🏆 STUDY COMPLETE 🏆 ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print(f"Total Trials: {len(study.trials)}")
    print(f"  ✅ Completed: {len(complete_trials)}")
    print(f"   pruned: {len(pruned_trials)}")

    if not complete_trials:
        print("\nNo trials completed successfully.")
    else:
        best_trial = study.best_trial
        print("\n--- ⭐ BEST TRIAL ---")
        print(f"  Value (Best Score): {best_trial.value:.4f}")
        print("  Parameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value:.6e}")