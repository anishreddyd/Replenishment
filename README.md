# MABIM
MABIM: Multi-Agent Benchmark for Inventory Management. 
Replenishment environment for OR and RL algorithms


## Contents

| Folder      | Description                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------- |
| [ReplenishmentEnv](ReplenishmentEnv)      | Replenishment env source code.                                      |
| [ReplenishmentEnv/config](ReplenishmentEnv/config)      | Config for building the env.                          |
| [ReplenishmentEnv/data](ReplenishmentEnv/data)      | Csv based data for skus, including sku_list, info and other dynamic data|
| [ReplenishmentEnv/env](ReplenishmentEnv/env)      | Kernel simulator for env                                    |
| [ReplenishmentEnv/utility](ReplenishmentEnv/utility)      | Utility simulator for env and wrapper               |
| [ReplenishmentEnv/wrapper](ReplenishmentEnv/wrapper)      | Wrapper for env                 |
| [Baseline](Baseline)                        | Show case of replenishment env.                                     |

## install MABIM 

* Create new virtual environment (Optional)
python -m venv myenv
In Windows 
```
myenv\Scripts\activate
```
In macOS and Linux:
```
source myenv/bin/activate
```

* Build and install MABIM
```
python setup.py install
```

## Create a environment
* Create a build-in environment
```
from ReplenishmentEnv import make_env
make_env("{task_name}")
```

* Create a consumed environment
    * Prepare the data including SKU data and warehouse information.
    * Write config as the format of demo [demo.yaml](ReplenishmentEnv/config/demo.yml)
    * Make the env by 
```
from ReplenishmentEnv import make_env
make_env("{config_name}", config_dir="{config_dir}")
```

## Run OR algorithm
* Install MABIM
* Install dependencies by ```pip install -r algo_requirements.txt```
* Run OR algorithm by
```
import os
from Baseline.OR_algorithm.base_stock import BS_static, BS_dynamic
from Baseline.OR_algorithm.search_sS import sS_static, sS_hindsight
env_name = "sku200.single_store.standard"

# Base stock static mode
vis_path = os.path.join("output", env_name, "BS_static")
BS_static_sum_balance = sum(BS_static(env_name, vis_path))
print(env_name, "BS_static", BS_static_sum_balance)

# Base stock dynamic mode
vis_path = os.path.join("output", env_name, "BS_dynamic")
BS_static_sum_balance = sum(BS_dynamic(env_name, vis_path))
print(env_name, "BS_dynamic", BS_static_sum_balance)

# (s, S) static mode
vis_path = os.path.join("output", env_name, "sS_static")
sS_static_sum_balance = sum(sS_static(env_name, vis_path))
print(env_name, "BS_static", sS_static_sum_balance)

# (s, S) hindsight mode
vis_path = os.path.join("output", env_name, "sS_hindsight")
sS_hindsight_sum_balance = sum(sS_hindsight(env_name, vis_path))
print(env_name, "sS_hindsight", sS_hindsight_sum_balance)
```
Visualization policy will be in output folder.

## Run MARL algorithm
The MARL training only tested in Linux. The training curve are available in wandb.
* Install MABIM
* Install dependencies by ```pip install -r algo_requirements.txt```
* Specify the environment by modify the task_type field in [replenishment.yaml](Baseline/MARL_algorithm/config/envs/replenishment.yaml)
* IPPO training 
    * Specify hyper parameter if needed in algorithm file, such as [ippo.yaml](Baseline/MARL_algorithm/config/algo/ippo.yaml)
    * Run ```python main.py --config=ippo --env-config=replenishment```    
* QTRAN training
    * Specify hyper parameter if needed in algorithm file, such as [qtran.yaml](Baseline/MARL_algorithm/config/algo/qtran.yaml)
    * Run ```python main.py --config=qtran --env-config=replenishment```
* Get training curve in wandb
* If need visualization policy, set the ```visualize:True``` on algorithm file
