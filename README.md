# Code for "Revisiting Design Choices in Proximal Policy Optimization"

This repository is is forked from the open-source code for ICLR 2020 paper
"Implementation Matters in Deep RL: A Case Study on PPO and TRPO":
<https://github.com/implementation-matters/code-for-paper>.

We checked the existing open-source code and fixed two bugs in the initial
open source version after communicating with the authors, added customized 
code for experimenting with KL directions and Beta policy.

All our plots are produced via Jupyter notebooks in the ``analysis`` folder.

We assume that the user has a machine with MuJoCo and mujoco\_py properly set up and installed, i.e.
you should be able to run the following command on your system without errors:

```python
import gym
gym.make_env("Humanoid-v2")
```

To reproduce our MuJoCo figures: run the following commands:
1. ``cd src/gaussian_vs_beta/``
2. ``python setup_agents.py``: the setup\_agents.py script contains detailed
experiments settings.
3. ``cd ../``
4. Edit the ``NUM_THREADS`` variables in the ``run_agents.py`` file according to your local machine.
5. Train the agents: ``python run_agents.py reward_scaling/agent_configs``
6. Repeat the above with ``src/kl_direction`` and ``src/penalty_outside`` in step 1.
7. Plot results in the corresponding jupyter notebooks in the analysis folder.


For more details about the code, see the README file in the original github repo:
<https://github.com/implementation-matters/code-for-paper>.
