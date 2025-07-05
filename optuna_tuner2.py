import os
import re
import subprocess
import time
import optuna
from typing import List
import wandb

# --- CONFIGURATION ---
# These are no longer needed for a sequential run, but kept for clarity
# N_PARALLEL_TRIALS = 1
# GPU_IDS: List[int] = [0]
TOTAL_TRIALS_TO_RUN = 15  # Set for your full study
CLEANUP_TIMEOUT_SECONDS = 20


def objective(trial: optuna.trial.Trial) -> float:
    """
    This function encapsulates a single trial run.
    """
    # --- Get hyperparameters for this trial ---
    mini_epochs = trial.suggest_categorical("mini_epochs", [8, 16, 20])
    entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 5e-2, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    env = os.environ.copy()
    # If you have multiple GPUs, you can still specify which one to use
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n🚀 Starting Trial {trial.number} with:")
    print(f"   mini_epochs={mini_epochs}, entropy_coef={entropy_coef:.4e}, gae_lambda={gae_lambda:.3f}")

    cmd = [
        "python",
        "main.py",
        "--config=gmappo_plus",
        "--env-config=replenishment",
        "--capture=no",  # This is critical to prevent hanging
        "with",
        "lr=4.5e-05",  # Use the stable LR we found
        f"mini_epochs={mini_epochs}",
        f"entropy_coef={entropy_coef}",
        f"gae_lambda={gae_lambda}",
        "t_max=500000",
        "use_cuda=True",
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )

    full_output = []
    # Read output line-by-line to enable real-time pruning
    for step, line in enumerate(iter(process.stdout.readline, '')):
        line = line.strip()
        print(line)
        full_output.append(line)

        # Check for the intermediate score to report to the pruner
        match = re.search(r"INTERMEDIATE_VAL_RETURN: ([\-\d\.]+)", line)
        if match:
            intermediate_value = float(match.group(1))
            trial.report(intermediate_value, step)

            # Ask the pruner if this trial should be stopped early
            if trial.should_prune():
                print(f"--- ✂️ TRIAL {trial.number} PRUNED at step {step} ---")
                process.terminate()
                raise optuna.exceptions.TrialPruned()

    process.stdout.close()

    # Wait for the process to finish, but with a timeout
    try:
        returncode = process.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"--- ⚠️ TRIAL {trial.number} HUNG after finishing. Terminating. ---")
        process.terminate()
        process.wait()
        raise optuna.exceptions.TrialPruned()

    full_stdout_str = "\n".join(full_output)

    # Check for errors (-15 is the code for a successful termination/prune)
    if returncode not in [0, -15]:
        print(f"--- ❌ TRIAL {trial.number} FAILED (Exit Code: {returncode}) ---")
        raise optuna.exceptions.TrialPruned()

    # Parse the final score from a successful run
    final_matches = re.findall(r"new test result : ([\-\d\.]+)", full_stdout_str)
    if final_matches:
        final_score = float(final_matches[-1])
        print(f"--- ✅ TRIAL {trial.number} SUCCESS --- Final Score: {final_score}")
        # Explicitly finish the wandb run associated with this trial
        if wandb.run is not None:
            wandb.finish()
        return final_score
    else:
        print(f"--- ⚠️ TRIAL {trial.number} FINISHED W/O FINAL SCORE --- Pruning.")
        if wandb.run is not None:
            wandb.finish()
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(
        study_name="gmappo_performance_tuning_sequential",
        direction="maximize",
        pruner=pruner
    )

    # The main loop is now a simple, direct call to study.optimize
    print(f"🔬 Starting sequential study for {TOTAL_TRIALS_TO_RUN} trials...")
    study.optimize(objective, n_trials=TOTAL_TRIALS_TO_RUN)

    # --- This block will now be executed reliably ---
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