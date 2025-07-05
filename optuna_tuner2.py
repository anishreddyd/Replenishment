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
TOTAL_TRIALS_TO_RUN = 2  # Increased trials are now feasible with pruning
CLEANUP_TIMEOUT_SECONDS = 15

print_lock = Lock()


def run_trial(trial: optuna.trial.Trial, gpu_id: int) -> float:
    # --- Get hyperparameters from the trial object ---
    mini_epochs = trial.params["mini_epochs"]
    entropy_coef = trial.params["entropy_coef"]
    gae_lambda = trial.params["gae_lambda"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    with print_lock:
        print(f"🚀 Starting Trial {trial.number} on GPU {gpu_id} with:")
        print(f"   mini_epochs={mini_epochs}, entropy_coef={entropy_coef:.4e}, gae_lambda={gae_lambda:.3f}")

    cmd = [
        "python", "main.py", "--config=gmappo_plus", "--env-config=replenishment",
        "--capture=no",
        "with",
        "lr=4.5e-05",  # Use the stable LR we found
        f"mini_epochs={mini_epochs}",
        f"entropy_coef={entropy_coef}",
        f"gae_lambda={gae_lambda}",
        "t_max=10000",
        "use_cuda=True",
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )

    full_output = []
    final_score = -float('inf')

    # --- UPDATED: Real-time parsing for pruning ---
    for step, line in enumerate(iter(process.stdout.readline, '')):
        line = line.strip()
        print(line)
        full_output.append(line)

        # Check for intermediate value report
        match = re.search(r"INTERMEDIATE_VAL_RETURN: ([\-\d\.]+)", line)
        if match:
            intermediate_value = float(match.group(1))

            # Report the score to Optuna
            trial.report(intermediate_value, step)

            # Check if the trial should be pruned
            if trial.should_prune():
                with print_lock:
                    print(f"--- ✂️ TRIAL {trial.number} PRUNED at step {step} ---")
                process.terminate()  # Kill the subprocess
                raise optuna.exceptions.TrialPruned()

    process.stdout.close()
    returncode = process.wait()
    full_stdout_str = "\n".join(full_output)

    if returncode != 0 and returncode != -15:  # -15 is the termination signal
        with print_lock:
            print(f"--- ❌ TRIAL {trial.number} FAILED (Exit Code: {returncode}) ---")
        raise optuna.exceptions.TrialPruned()

    # Find the final test score for successful trials
    final_matches = re.findall(r"new test result : ([\-\d\.]+)", full_stdout_str)
    if final_matches:
        final_score = float(final_matches[-1])
        with print_lock:
            print(f"--- ✅ TRIAL {trial.number} SUCCESS --- Final Score: {final_score}")
        return final_score
    else:
        # If a trial finishes without ever reporting a score (rare), prune it.
        with print_lock:
            print(f"--- ⚠️ TRIAL {trial.number} FINISHED W/O SCORE --- Pruning.")
        raise optuna.exceptions.TrialPruned()


def objective_worker(study: optuna.study.Study, gpu_queue: Queue):
    # This function remains the same as the previous version
    while True:
        if len(study.get_trials(deepcopy=False)) >= TOTAL_TRIALS_TO_RUN:
            break
        gpu_id = gpu_queue.get()
        if gpu_id is None: break
        try:
            trial = study.ask({
                "mini_epochs": optuna.distributions.CategoricalDistribution([8, 16, 20]),
                "entropy_coef": optuna.distributions.FloatDistribution(1e-4, 5e-2, log=True),
                "gae_lambda": optuna.distributions.FloatDistribution(0.9, 0.99),
            })
        except Exception:
            gpu_queue.put(gpu_id)
            break
        try:
            study.tell(trial, run_trial(trial, gpu_id))
        except optuna.exceptions.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as e:
            with print_lock:
                print(f"An unexpected error occurred in trial {trial.number}: {e}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        finally:
            gpu_queue.put(gpu_id)


if __name__ == "__main__":
    # --- UPDATED: Instantiate and add the pruner to the study ---
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,  # The first check will be after the first report
        reduction_factor=4,
        min_early_stopping_rate=0
    )

    study = optuna.create_study(
        study_name="gmappo_performance_tuning_pruning",
        direction="maximize",
        pruner=pruner  # Add the pruner here
    )

    # ... (the rest of the __main__ block remains the same) ...
    gpu_queue = Queue()
    for gpu_id in GPU_IDS: gpu_queue.put(gpu_id)
    processes = []
    print(f"🔬 Starting study with {N_PARALLEL_TRIALS} parallel workers on GPUs: {GPU_IDS}")
    for _ in range(N_PARALLEL_TRIALS):
        p = Process(target=objective_worker, args=(study, gpu_queue))
        p.start()
        processes.append(p)
    while len(study.get_trials(deepcopy=False)) < TOTAL_TRIALS_TO_RUN:
        print(f"📈 Progress: {len(study.get_trials(deepcopy=False))}/{TOTAL_TRIALS_TO_RUN} trials registered...",
              end="\r")
        time.sleep(1)
    for _ in range(N_PARALLEL_TRIALS): gpu_queue.put(None)
    for p in processes:
        p.join(timeout=CLEANUP_TIMEOUT_SECONDS)
        if p.is_alive():
            print(f"Worker process {p.pid} did not exit cleanly. Terminating.")
            p.terminate()

    print(f"\n\n--- 🏆 STUDY COMPLETE 🏆 ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print(f"Total Trials: {len(study.trials)}")
    print(f"  ✅ Completed: {len(complete_trials)}")
    print(f"  ✂️ Pruned: {len(pruned_trials)}")
    if complete_trials:
        best_trial = study.best_trial
        print("\n--- ⭐ BEST TRIAL ---")
        print(f"  Value (Best Score): {best_trial.value:.4f}")
        print("  Parameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("\nNo trials completed successfully.")