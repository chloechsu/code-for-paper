import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "policy_net_type": ["CtsPolicy"],
    "game": ["Walker2d-v2", "Hopper-v2", "HalfCheetah-v2", "Humanoid-v2"],
    "mode": ["ppo"],
    "clip_eps": [0.2, 1e8],
    "kl_penalty_coeff": [0., 3.],
    "ppo_lr_adam": [1e-3, 3e-4, 1e-5],
    #"kl_penalty_direction": ["new_to_old", "old_to_new"],
    "kl_penalty_direction": ["new_to_old", "old_to_new"],
    "out_dir": ["ratio_logging/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "clip_advantages": [1e8],
    # "ppo_lr_adam": iwt(1e-5, 2.9e-4, 7e-5, 5),
    "val_lr": [2e-5],
    "cpu": [True],
    "advanced_logging": [True],
    "sign_advantages": [False, True],
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
