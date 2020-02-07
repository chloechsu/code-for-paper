# Code for "Questioning Proximal Policy Optimization with Multiplicative Weights"

This repository is is forked from the open-source code for ICLR 2020 paper
"Implementation Matters in Deep RL: A Case Study on PPO and TRPO":
<https://github.com/implementation-matters/code-for-paper>.

We thoroughly checked the open-source code and fixed two bugs in the initial
open source version after communicating with the authors, add customized the
code for experimenting with KL directions.

All our plots are produced via Jupyter notebooks in the ``analysis`` folder.

We assume that the user has a machine with MuJoCo and mujoco\_py properly set up and installed, i.e.
you should be able to run the following command on your system without errors:

```python
import gym
gym.make_env("Humanoid-v2")
```

To reproduce our results in Figure 1, Table 3, Figure 5, and Figure 6, one can run the
following commands:
1. ``cd src/reward_scaling/``
2. ``python setup_agents.py``: the setup\_agents.py script contains detailed
experiments settings.
3. ``cd ../``
4. Edit the ``NUM_THREADS`` variables in the ``run_agents.py`` file according to your local machine.
5. Train the agents: ``python run_agents.py reward_scaling/agent_configs``
6. Plot results in the ``analysis/figure1_table3_figure5_figure6.ipynb`` notebook.


To reproduce our results in Figure 7,
following commands:
1. ``cd src/kl_direction_experiment/``
2. ``python setup_agents.py``: the setup\_agents.py script contains detailed
experiments settings.
3. ``cd ../``
4. Edit the ``NUM_THREADS`` variables in the ``run_agents.py`` file according to your local machine.
5. Train the agents: ``python run_agents.py kl_direction_experiment/agent_configs``
6. Plot results in the ``analysis/appendix_figure7.ipynb`` notebook.


To reproduce Figure 2, see the ``analysis/figure2.ipynb`` notebook.

To reproduce Figure 3, see the ``analysis/figure3.ipynb`` notebook.


For more details about the code, see the README file in the original github repo:
<https://github.com/implementation-matters/code-for-paper>.
