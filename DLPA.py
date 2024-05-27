import numpy as np
import torch

from utils import device
import utils as u
from models.world_model import world_model
from copy import deepcopy
from torch.autograd import Variable
import os
import math
import copy

import wandb


class Trainer:
    def __init__(self, args):
        args.device = device

        u.set_seed(args.seed)
        self.env, self.args = u.make_env(args)

        if args.save_points:
            self.save_points()

        self.model = world_model(args).to(device)
        self.model_target = deepcopy(self.model).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.td_lr)

        self.buffer = u.ReplayBuffer(args)

        dir = f"result/DLPA/{args.env.split('-')[0]}"
        data = args.save_dir
        redir = os.path.join(dir, data)
        if not os.path.exists(redir):
            os.makedirs(redir)
        self.redir = redir
        self.Test_Reward_100 = []
        self.Test_epioside_step_100 = []

    def save_points(self):
        run = wandb.init(
            project="dlpa",
            config=self.args,
            dir="../scratch/wandb"
        )

    def upload_log(self, mylog):
        if self.args.save_points:
            wandb.log(mylog)

    def save_local(self):
        title3 = "Test_Reward_"
        title4 = "Test_epioside_step_"
        np.savetxt(os.path.join(self.redir, title3 + "{}".format(str(self.args.seed) + ".csv")), self.Test_Reward_100, delimiter=',')
        np.savetxt(os.path.join(self.redir, title4 + "{}".format(str(self.args.seed) + ".csv")), self.Test_epioside_step_100,
                delimiter=',')
        
        model_file = "world_model_"
        torch.save(self.model.state_dict(), os.path.join(self.redir, model_file + "{}".format(str(self.args.seed) + ".pth")))

    def act(self, action, timestep, pre_state=None):
        ret = self.env.step(action)

        if self.args.env == "simple_catch-v0":
            next_state, reward, terminal_n, _ = ret

            if pre_state[-2]<= 2/3 and action[0][-1] and np.sum(np.square(next_state[0][2:4]))>0.04:
                valid_time = pre_state[-2] + 1/6
            else:
                valid_time = pre_state[-2]

            next_state = next_state[0].tolist() + [valid_time, pre_state[-1]+1/12]

            reward = reward[0]

            terminal = all(terminal_n)
            if reward > 4 or reward == 0 or timestep >= self.args.episode_length:
                terminal = True

        elif self.args.env == "simple_move_4_direction_v1-v0":
            next_state, reward, done_n, _ = ret
            next_state = next_state[0].tolist()
            terminal = all(done_n)
            reward = reward[0]
            if reward > 4 or timestep >= self.args.episode_length:
                terminal = True

        else:
            (next_state, steps), reward, terminal, _ = ret

        next_state = np.array(next_state, dtype=np.float32, copy=False)
        return next_state, reward, terminal
    
    def reset(self):
        if self.args.env == "simple_catch-v0":
            state = self.env.reset()
            valid_time, timestep = -1., -1.
            state = state[0].tolist() + [valid_time, timestep]  # agent_vol, direction2target

        elif self.args.env == "simple_move_4_direction_v1-v0":
            state = self.env.reset()
            state = state[0].tolist()

        else:
            state, _ = self.env.reset()
        return np.array(state, dtype=np.float32, copy=False)

    def evaluate(self, total_timesteps):
        returns = []
        epioside_steps = []
        vis = self.args.visualise

        for epi in range(self.args.eval_eposides):
            state = self.reset()
            t = 0
            
            with torch.no_grad():
                act, act_param = self.plan(state, eval_mode=True, t0=True, step=0, local_step=t)
                action = self.pad_action(act, act_param)

            if vis:
                self.env.render()

            terminal = False
            total_reward = 0.
            
            while not terminal:
                t += 1
                state, reward, terminal = self.act(action, t, pre_state=state)

                with torch.no_grad():
                    act, act_param = self.plan(state, eval_mode=True, t0=False, step=0, local_step=t)
                    action = self.pad_action(act, act_param)

                if vis:
                    self.env.render()

                total_reward += reward
            epioside_steps.append(t)
            returns.append(total_reward)
        print("---------------------------------------")
        print(
            f"Timestep {total_timesteps} || Evaluation over {self.args.eval_eposides} episodes_rewards: {np.array(returns).mean():.3f} epioside_steps: {np.array(epioside_steps).mean():.3f}")
        print("---------------------------------------")
        Test_Reward = np.array(returns).mean()
        Test_epioside_step = np.array(epioside_steps).mean()

        self.Test_Reward_100.append(Test_Reward)
        self.Test_epioside_step_100.append(Test_epioside_step)

        self.upload_log({"Test_Reward": Test_Reward, "Test_epioside_step": Test_epioside_step})

    def rand_action(self):
        k = torch.randint(low=0, high=self.args.k_dim, size=[1])
        z = torch.rand([self.args.par_size[k]]) * self.args.scale + self.args.offsets
        
        return k.item(), z
    
    def dealRaw(self, k, z):
        size = torch.from_numpy(self.args.par_size).to(device)[k.argmax(-1)].unsqueeze(-1).repeat(1, self.args.z_dim)
        mask = torch.arange(self.args.z_dim).to(device).repeat(len(size), 1)
        mask = torch.where(mask<size, 1., 0.)
        z = z * mask
        return torch.cat([k, z], dim=-1)
    
    @torch.no_grad()
    def sample_from_N(self, mean, std):
        kmean = mean['k']
        zmean, zstd = mean['z'], std

        k_int = torch.multinomial(kmean, self.args.mpc_popsize, replacement=True)
        k_onehot = torch.nn.functional.one_hot(k_int, num_classes=self.args.k_dim).to(device)
        
        z_all = torch.clamp(zmean.unsqueeze(1) + zstd.unsqueeze(1) * \
                torch.randn(self.args.mpc_horizon, self.args.mpc_popsize, self.args.all_z_dim, device=zstd.device), self.args.lb, self.args.ub)
        
        offsets = torch.tensor(self.args.offset).to(device)[k_int.flatten()].unsqueeze(-1).repeat(1, self.args.z_dim) + torch.arange(self.args.z_dim, device=device)

        z_one = torch.zeros([self.args.mpc_horizon*self.args.mpc_popsize, self.args.all_z_dim+self.args.z_dim], device=device)
        z_one[:, :self.args.all_z_dim] = z_all.reshape([-1, self.args.all_z_dim])
        
        zs = torch.gather(z_one, 1, offsets)
        
        size = torch.from_numpy(self.args.par_size).to(device)[k_int.flatten()].unsqueeze(-1).repeat(1, self.args.z_dim)
        mask = torch.arange(self.args.z_dim).to(device).repeat(len(size), 1)
        mask = torch.where(mask<size, 1., 0.)
        zs = zs * mask
        
        zs = zs.reshape([self.args.mpc_horizon, self.args.mpc_popsize, self.args.z_dim])
        return torch.cat([k_onehot, zs], dim=-1)
    
    @torch.no_grad()
    def estimate_value(self, s, actions, horizon, local_step, eval_mode):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        num_traj = s.shape[0]
        c = torch.ones([num_traj, 1], device=device)

        for t in range(horizon):

            if eval_mode:
                s_pred, reward, ci, r1 = self.model.next(s, actions[t], reparameterize=False, return_log_prob=False, deterministic=True)
            else:
                s_pred, reward, ci, r1 = self.model.next(s, actions[t], reparameterize=True, return_log_prob=True, deterministic=False)

            ci = ci.argmax(-1).unsqueeze(-1)

            if r1 is not None:
                reward = torch.where(ci.bool(), reward, r1)

            G += discount * reward * c
            discount *= self.args.mpc_gamma
            c *= ci

            s = s_pred

        return G
    
    @torch.no_grad()
    def plan(self, state, eval_mode=False, step=None, t0=True, local_step=None):
        if step < self.args.seed_steps and not eval_mode:
            return self.rand_action()
        
        if not eval_mode:
            self.model.timestep += 1
        
        # Sample policy trajectories
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        horizon = self.args.mpc_horizon

        # if True:
        #     k_int = torch.randint(0, self.args.k_dim, size=[self.args.mpc_horizon, self.args.mpc_popsize])
        #     k_onehot = torch.nn.functional.one_hot(k_int, num_classes=self.args.k_dim).to(device)

        #     z_all = torch.randn(self.args.mpc_horizon, self.args.mpc_popsize, self.args.all_z_dim, device=device) * 2 - 1
            
        #     offsets = torch.tensor(self.args.offset).to(device)[k_int.flatten()].unsqueeze(-1).repeat(1, self.args.z_dim) + torch.arange(self.args.z_dim, device=device)

        #     z_one = torch.zeros([self.args.mpc_horizon*self.args.mpc_popsize, self.args.all_z_dim+self.args.z_dim], device=device)
        #     z_one[:, :self.args.all_z_dim] = z_all.reshape([-1, self.args.all_z_dim])
            
        #     zs = torch.gather(z_one, 1, offsets)
            
        #     size = torch.from_numpy(self.args.par_size).to(device)[k_int.flatten()].unsqueeze(-1).repeat(1, self.args.z_dim)
        #     mask = torch.arange(self.args.z_dim).to(device).repeat(len(size), 1)
        #     mask = torch.where(mask<size, 1., 0.)
        #     zs = zs * mask
            
        #     zs = zs.reshape([self.args.mpc_horizon, self.args.mpc_popsize, self.args.z_dim])
        #     actions = torch.cat([k_onehot, zs], dim=-1)

        #     s = state.repeat(self.args.mpc_popsize, 1)
        #     value = self.estimate_value(s, actions, self.args.mpc_horizon, local_step, eval_mode=eval_mode).nan_to_num_(0)
        #     ind = torch.argmax(value.squeeze(1))

        #     k = k_int[0, ind].item()
        #     z = zs[0, ind]

        #     if not eval_mode and (self.args.env == "simple_catch-v0" and k==0):
        #         z += torch.randn(self.args.par_size[k], device=device)
        #         z = z.clamp(-1., 1.)

        #     if self.args.env in [ "simple_catch-v0"] and k==1:
        #         z = torch.zeros(1)

        #     return k, z

        # Initialize state and parameters
        s = state.repeat(self.args.mpc_popsize, 1)

        kmean = torch.ones(horizon, self.args.k_dim, device=device)
        kmean /= kmean.sum(-1).unsqueeze(-1)
        
        zmean = torch.zeros(horizon, self.args.all_z_dim, device=device)
        std = 2*torch.ones(horizon, self.args.all_z_dim, device=device)
        mean = {'k': kmean, 'z': zmean}
        if not t0 and hasattr(self, '_prev_mean'):
            mean['k'][:-1] = self._prev_mean['k'][1:]
            mean['z'][:-1] = self._prev_mean['z'][1:]

        # Iterate CEM
        for i in range(self.args.cem_iter):
            actions = self.sample_from_N(mean, std)
                
            # Compute elite actions
            value = self.estimate_value(s, actions, horizon, local_step, eval_mode=eval_mode).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.args.mpc_num_elites, dim=0).indices
            elite_value = value[elite_idxs]  # [num_elite, 1]
            elite_actions = actions[:, elite_idxs]  # [horizon, num_elite, a_dim]

            max_value = elite_value.max(0)[0]

            # Update k parameters
            # k_score is k weights, softmax(elite_value-max)
            k_score = torch.exp(self.args.mpc_temperature*(elite_value - max_value))
            k_score /= k_score.sum(0)  # [num_elite, 1]
            kelites = elite_actions[:, :, :self.args.k_dim]
            _kmean = torch.sum(k_score.unsqueeze(0) * kelites, dim=1) / (k_score.sum(0) + 1e-9)

            # Update z parameters
            zelites = elite_actions[:, :, self.args.k_dim:]
            k_all = kelites.argmax(-1).unsqueeze(-1)  # [horizon, num_elite, 1]
            z_score = elite_value.unsqueeze(0).repeat([horizon, 1, 1])  # [horizon, num_elite, 1]
            _zmean, _std = torch.zeros_like(mean['z']), torch.zeros_like(std)

            for ki in range(self.args.k_dim):
                selected_ind = (k_all == ki)  # selected discrete type, [horizon, num_elite, 1]
                zis = zelites[:, :, :self.args.par_size[ki]]
                # zi: [horizon, num_elite, z_dim], = zi if selected else 0
                zi = torch.where(selected_ind, zis, torch.zeros_like(zis).to(device))

                # weight: [horizon, num_elite, z_dim], = softmax(selected(z))
                weight = torch.where(selected_ind, z_score, torch.tensor([float("-Inf")]).to(device))
                weight = torch.exp(self.args.mpc_temperature*(weight - max_value))
                weight_sum = weight.squeeze(-1).sum(1).reshape([-1, 1, 1]).repeat(1, self.args.mpc_num_elites, 1)
                weight /= (weight_sum + 1e-9)
                
                _zimean = torch.sum(weight * zi, dim=1) / (weight.sum(1) + 1e-9)
                _zistd = torch.sqrt(torch.sum(weight * (zi - _zimean.unsqueeze(1)) ** 2, dim=1) / (weight.sum(1) + 1e-9))

                ind_start = self.args.offset[ki]
                ind_end = ind_start + self.args.par_size[ki]

                if_non_select = selected_ind.squeeze(-1).sum(1).unsqueeze(-1)
                _zimean = torch.where(if_non_select==0, mean['z'][:, ind_start:ind_end], _zimean)
                _zistd = torch.where(if_non_select==0, std[:, ind_start:ind_end], _zistd)

                _zmean[:, ind_start:ind_end] = _zimean
                _std[:, ind_start:ind_end] = _zistd

            mean['k'] = self.args.mpc_alpha * mean['k'] + (1 - self.args.mpc_alpha) * _kmean
            mean['z'] = self.args.mpc_alpha * mean['z'] + (1 - self.args.mpc_alpha) * _zmean
            std = self.args.mpc_alpha * std + (1 - self.args.mpc_alpha) * _std

        # Outputs
        score = k_score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]

        k = mean[:self.args.k_dim].argmax()
        z = mean[self.args.k_dim:self.args.k_dim+self.args.par_size[k]]
        
        if not eval_mode:
            ind_start = self.args.offset[k]
            ind_end = ind_start + self.args.par_size[k]
            z += std[ind_start:ind_end] * torch.randn(self.args.par_size[k], device=device)

        k = k.item()
        if self.args.env in [ "simple_catch-v0"] and k==1:
            z = torch.zeros(1)
            
        return k, z

    def pad_action(self, act, act_param):
        act_param = act_param.cpu().numpy()

        if self.args.env == "simple_catch-v0":
            if act == 0:
                action = np.hstack(([1], act_param * math.pi, [1], [0]))
            else:  # catch
                action = np.hstack(([1], [0], [0], [1]))
            return [action]
        
        elif self.args.env == "hard_goal-v0":
            return self.pad_hardgoal(act, act_param)
        
        elif self.args.env == "simple_move_4_direction_v1-v0":
            action = np.hstack(([8], [act], [self.args.action_n_dim])).tolist()

            act_params = [0] * (2 ** self.args.action_n_dim)
            act_params[act] = act_param[0]
            action.append(act_params)
            return [action]
        
        else:
            params = [np.zeros((num,), dtype=np.float32) for num in self.args.par_size]
            params[act][:] = act_param
            return (act, params)
    
    def train_sperate(self, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        obs, next_obses, ks, zs, reward, idxs, weights, continuous, masks = self.buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        action = torch.cat([ks, zs], dim=-1)
        self.model.train()

        consistency_loss, reward_loss, continue_loss = 0, 0, 0

        for t in range(self.args.mpc_horizon):
            mask = masks[t].unsqueeze(1)

            if not mask.any():
                break
            
            rho =  self.args.rho ** t

            with torch.no_grad():
                next_obs = next_obses[t]

            obs, r0, c_pred, r1 = self.model.next(obs, action[t], reparameterize=True, return_log_prob=True, deterministic=False)

            if r1 is not None:
                reward_pred = torch.where(continuous[t].unsqueeze(-1).bool(), r0, r1)
            else:
                reward_pred = r0
            
            consistency_loss += rho * torch.mean(u.mse(obs, next_obs), dim=1, keepdim=True) * mask
            reward_loss += rho * u.mse(reward_pred, reward[t]) * mask
            continue_loss += rho * u.ce(c_pred, continuous[t]) * mask
        
        total_loss = self.args.consistency_coef * consistency_loss.clamp(max=1e4) + \
                        self.args.reward_coef * reward_loss.clamp(max=1e4) + \
                        self.args.contin_coef * continue_loss.clamp(max=1e4)
        
        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1/self.args.mpc_horizon))
        weighted_loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm, error_if_nonfinite=False)
        self.optim.step()

        self.model.eval()
        
        return {
                'consistency_loss': float(consistency_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'continuous_loss': float(continue_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item()),
                }
    
    def random_action(self):
        k = np.random.randint(0, self.args.k_dim)
        psize = self.args.par_size[k]
        z = np.random.random(psize) * 2 - 1
        return k, z
    
    