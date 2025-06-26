import os
import re
import subprocess
import optuna


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    cmd = [
        "python",
        "main.py",
        "--config=gmappo_plus",
        "--env-config=replenishment",
        "with",
        f"lr={lr}",
        "t_max=100000"
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    match = re.search(r"val return : ([\\-\\d\\.]+)", result.stdout)
    if not match:
        match = re.search(r"new best val result : ([\\-\\d\\.]+)", result.stdout)
    if match:
        return float(match.group(1))
    if result.returncode != 0:
        raise RuntimeError(f"main.py failed:\n{result.stdout}")
    return -float("inf")


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best LR:", study.best_params["lr"])
