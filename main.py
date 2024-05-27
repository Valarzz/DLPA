import numpy as np
import torch
import gym
import argparse
import time

from DLPA import Trainer
from utils import Episode


def run(args):
    trainer = Trainer(args)

    total_timesteps = 0
    total_episodes = 0
    max_per_epi_steps = args.episode_length

    # trainer.evaluate(0)

    train_time = []
    infer_time = []

    while total_timesteps < args.max_timesteps:
        state = trainer.reset()
        episode = Episode(trainer.args, state)

        episode_reward = 0.

        for j in range(max_per_epi_steps):
            with torch.no_grad():
                start_infer = time.time()
                act, act_param = trainer.plan(state, step=total_episodes, t0=(j==0), local_step=j)
                infer_time.append(time.time()-start_infer)

                action = trainer.pad_action(act, act_param)
                state, reward, terminal = trainer.act(action, j, pre_state=state)

            episode += (state, act, act_param, reward, terminal)

            total_timesteps += 1
            episode_reward += reward

            train_metrics = {}
            if total_episodes >= args.seed_steps:
                for i in range(args.num_updates):
                    start_train = time.time()
                    train_log = trainer.train_sperate(total_episodes+i)
                    train_time.append(time.time()-start_train)

                    train_metrics.update(train_log)
                    trainer.upload_log(train_log)

            # if total_timesteps % args.eval_freq == 0:
            #     trainer.evaluate(total_timesteps)
            #     trainer.save_local()
            #     break

            if terminal:
                break
            
        trainer.buffer += episode
        total_episodes += 1

        if (len(train_time) > 1e3) and (len(infer_time) > 1e3):
            train_time = np.array(train_time)
            infer_time = np.array(infer_time)
            print(f"train aver: {train_time.mean()} || train len: {train_time.shape} || var: {train_time.var()}")
            print(f"infer aver: {infer_time.mean()} || infer len: {infer_time.shape} || var: {infer_time.var()}")
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 'Platform-v0', 'Goal-v0', "hard_goal-v0", 'simple_catch-v0', 'simple_move_4_direction_v1-v0'
    parser.add_argument("--env", default='Platform-v0')  
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=500_000, type=int)  # Max time steps to run environment for
    parser.add_argument("--eval_freq", default=50, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--eval_eposides", default=50, type=int)
    parser.add_argument("--num_updates", default=25, type=int)
    parser.add_argument('--action_n_dim', default=4, help='action_n_dim.', type=int)
    parser.add_argument("--seed_steps", default=50, type=int)
    
    parser.add_argument("--layers", default=64, type=int)

    parser.add_argument("--mpc_horizon", default=8, type=int)
    parser.add_argument("--mpc_gamma", default=0.99, type=float)
    parser.add_argument("--mpc_popsize", default=1000, type=int)
    parser.add_argument("--mpc_num_elites", default=100, type=int)
    parser.add_argument("--mpc_patrical", default=1, type=int)
    parser.add_argument("--mpc_init_mean", default=0., type=float)
    parser.add_argument("--mpc_init_var", default=1., type=float)
    parser.add_argument("--mpc_epsilon", default=0.001, type=float)
    parser.add_argument("--mpc_alpha", default=0.1, type=float)
    parser.add_argument("--mpc_max_iters", default=1e3, type=int)

    parser.add_argument("--max_buffer_size", default=1e6, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--mixture_coef", default=0.05, type=float)
    
    parser.add_argument("--min_std", default=0.05, type=float)
    parser.add_argument("--cem_iter", default=6, type=int)
    parser.add_argument("--mpc_temperature", default=0.5, type=float)
    parser.add_argument("--td_lr", default=3e-4, type=float)
    parser.add_argument("--rho", default=0.5, type=float)
    parser.add_argument("--grad_clip_norm", default=10, type=int)
    parser.add_argument("--consistency_coef", default=2, type=float)
    parser.add_argument("--reward_coef", default=0.5, type=float)
    parser.add_argument("--contin_coef", default=0.5, type=float)
    parser.add_argument("--value_coef", default=0.1, type=float)
    parser.add_argument("--per_alpha", default=0.6, type=float)
    parser.add_argument("--per_beta", default=0.4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    
    parser.add_argument("--model_type", default="multi", type=str)  # concat, multi, overlay
    parser.add_argument('--save_dir', default="070901", type=str)
    parser.add_argument('--visualise', default=0, type=int)
    parser.add_argument("--save_points", default=0, type=int)

    args = parser.parse_args()
    run(args)

    # for i in range(0, 3):
    #     args.seed = i
    #     run(args)
