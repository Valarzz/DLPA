# Model-based-Reinforcement-Learning-for-Parameterized-Action-Spaces

source code for ![]()

> We propose a novel model-based reinforcement learning algorithm---Dynamics Learning and predictive control with Parameterized Actions (DLPA)---for Parameterized Action Markov Decision Processes (PAMDPs). The agent learns a parameterized-action-conditioned dynamics model and plans with a modified Model Predictive Path Integral control. We theoretically quantify the difference between the generated trajectory and the optimal trajectory during planning in terms of the value they achieved through the lens of Lipschitz Continuity. Our empirical results on several standard benchmarks show that our algorithm achieves superior sample efficiency and asymptotic performance than state-of-the-art PAMDP methods.

### Requirements:

#### Dependencies:
- Pytorch == 2.0.1 
- gym == 0.10.5

#### Domains:
All the 8 domains with hybird action spaces this code tested on follows the installation instruction of ![HyAR](https://github.com/TJU-DRL-LAB/self-supervised-rl/tree/ece95621b8c49f154f96cf7d395b95362a3b3d4e/RL_with_Action_Representation/HyAR)

### Usage:

This code is tested on 8 benchmarks, for example, if testing on `platform` domain with a horizon of 10, a `masking` model structure, and recording data with `wandb`, run
```
python main.py --env 'Platform-v0' --mpc_horizon 10 --model_type "multi" --save_points 1
```
if testing on `goal` domain with a horizon of 8, a `sequential` model structure, and recording data with `wandb`, run
```
python main.py --env 'Goal-v0' --mpc_horizon 8 --model_type "overlay" --save_points 1
```
if testing on `hard_move` domain with a horizon of 5, a discrete action dimension of $2^8$, a `parallel` model structure, and not recording data, run
```
python main.py --env 'simple_move_4_direction_v1-v0' --mpc_horizon 5 --action_n_dim 8 --save_points 0 --model_type "concat"
```

