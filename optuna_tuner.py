import os
import re
import subprocess
import optuna


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-4)
    cmd = [
        "python",
        "main.py",
        "--config=gmappo_plus",
        "--env-config=replenishment",
        f"lr={lr}",
        "t_max=100000",
        "use_wandb=False",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    match = re.search(r"val return : ([\-\d\.]+)", result.stdout)
    if match:
        return float(match.group(1))
    return -float("inf")


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best LR:", study.best_params["lr"])
