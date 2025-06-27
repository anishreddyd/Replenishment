import os
import re
import subprocess
import time
from multiprocessing import Process, Queue, Lock
import optuna
from typing import List

# --- CONFIGURATION ---
N_PARALLEL_TRIALS = 1
GPU_IDS: List[int] = [0]
TOTAL_TRIALS_TO_RUN = 1

print_lock = Lock()


def run_trial(trial: optuna.trial.Trial, gpu_id: int) -> float:
    lr = trial.params["lr"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    with print_lock:
        print(f"🚀 Starting Trial {trial.number} on GPU {gpu_id} with lr={lr:.6e}...")

    cmd = [
        "python",
        "main.py",
        "--config=gmappo_plus",
        "--env-config=replenishment",
        "with",
        f"lr={lr}",
        "t_max=30000",
        "use_cuda=True",
        "debug_mode=True"
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )

    full_output = []
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        print(line)
        full_output.append(line)

    process.stdout.close()
    returncode = process.wait()

    full_stdout_str = "\n".join(full_output)

    if returncode != 0:
        with print_lock:
            print(f"--- ❌ TRIAL {trial.number} (GPU {gpu_id}) FAILED (Exit Code: {returncode}) ---")
        raise optuna.exceptions.TrialPruned()

    matches = re.findall(r"new test result : ([\-\d\.]+)", full_stdout_str)

    if matches:
        scores = [float(m) for m in matches]
        best_score = scores[-1]
        with print_lock:
            print(f"--- ✅ TRIAL {trial.number} (GPU {gpu_id}) SUCCESS --- Final Score: {best_score}")
        return best_score
    else:
        with print_lock:
            print(f"--- ⚠️ TRIAL {trial.number} (GPU {gpu_id}) KEY MISMATCH ---")
            print("Could not find 'new test result :'. Pruning.")
        raise optuna.exceptions.TrialPruned()

def objective_worker(study: optuna.study.Study, gpu_queue: Queue):
    """
    Worker process that runs trials. It no longer needs the processed queue.
    """
    while True:
        # Check if the study is finished before asking for a new trial
        # This prevents starting extra trials
        if len(study.get_trials(deepcopy=False)) >= TOTAL_TRIALS_TO_RUN:
            break
        gpu_id = gpu_queue.get()
        if gpu_id is None:
            break
        try:
            trial = study.ask({"lr": optuna.distributions.FloatDistribution(1e-6, 1e-4, log=True)})
        except Exception:
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
            gpu_queue.put(gpu_id)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="gmappo_tuning_final",
        direction="maximize"
    )
    gpu_queue = Queue()
    for gpu_id in GPU_IDS:
        gpu_queue.put(gpu_id)

    processes = []
    print(f"🔬 Starting study with {N_PARALLEL_TRIALS} parallel workers on GPUs: {GPU_IDS}")
    for _ in range(N_PARALLEL_TRIALS):
        p = Process(target=objective_worker, args=(study, gpu_queue))
        p.start()
        processes.append(p)

    # --- FIX: Monitor the study object directly instead of a separate queue ---
    while len(study.get_trials(deepcopy=False)) < TOTAL_TRIALS_TO_RUN:
        # Print progress using the study's trial count
        n_trials = len(study.get_trials(deepcopy=False))
        print(f"📈 Progress: {n_trials}/{TOTAL_TRIALS_TO_RUN} trials registered...", end="\r")
        time.sleep(1) # Wait a moment to avoid spamming and allow the study to update

    # Stop all worker processes
    for _ in range(N_PARALLEL_TRIALS):
        gpu_queue.put(None)
    for p in processes:
        p.join()
    print(f"\n\n--- 🏆 STUDY COMPLETE 🏆 ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print(f"Total Trials: {len(study.trials)}")
    print(f"  ✅ Completed: {len(complete_trials)}")
    print(f"  pruned: {len(pruned_trials)}")
    if complete_trials:
        best_trial = study.best_trial
        print("\n--- ⭐ BEST TRIAL ---")
        print(f"  Value (Best Score): {best_trial.value:.4f}")
        print("  Parameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value:.6e}")
    else:
        print("\nNo trials completed successfully.")