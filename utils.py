import gym
import os

from gym.wrappers import Monitor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import re
import torch.nn.functional as F

from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from common.soccer_domain import SoccerScaledParameterisedActionWrapper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

def ce(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.cross_entropy(pred.float(), target.long(), reduction=__REDUCE__(reduce))

def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(args):
    if args.env == "Platform-v0":
        import gym_platform

        env = gym.make("Platform-v0")
        env = ScaledStateWrapper(env)
        env = PlatformFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)

        state_dim = env.observation_space.spaces[0].shape[0]
        discrete_action_dim = env.action_space.spaces[0].n
        action_parameter_sizes = np.array(
			[env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
        parameter_action_dim = int(action_parameter_sizes.sum())

    elif args.env == "Goal-v0":
        env = gym.make('Goal-v0')
        env = GoalObservationWrapper(env)
        env = GoalFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)
        env = ScaledStateWrapper(env)
	
        state_dim = env.observation_space.spaces[0].shape[0]
        discrete_action_dim = env.action_space.spaces[0].n
        action_parameter_sizes = np.array(
			[env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
        parameter_action_dim = int(action_parameter_sizes.sum())

    elif args.env == "simple_catch-v0":
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios
        scenario = scenarios.load("simple_catch.py").Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        state_dim = obs_shape_n[0][0] + 1 + 1
        discrete_action_dim = 2
        action_parameter_sizes = np.array([1, 0])
        parameter_action_dim = int(action_parameter_sizes.sum())

    elif args.env == "hard_goal-v0":
        env = gym.make('Goal-v0')
        env = GoalObservationWrapper(env)
        env = GoalFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)
        env = ScaledStateWrapper(env)
	
        state_dim = env.observation_space.spaces[0].shape[0]
        discrete_action_dim = 11
        action_parameter_sizes = np.array(
			[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        parameter_action_dim = int(action_parameter_sizes.sum())

        args.discrete_emb_dim = 6
        args.parameter_emb_dim = 6

    elif args.env == 'simple_move_4_direction_v1-v0':
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios
        # load scenario from script
        scenario = scenarios.load("simple_move_4_direction_v1.py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]    
        state_dim = obs_shape_n[0][0]	
		
        action_n_dim = args.action_n_dim
        
        discrete_action_dim = 2 ** action_n_dim
        action_parameter_sizes = np.ones(discrete_action_dim, dtype=np.int64)
        parameter_action_dim = int(action_parameter_sizes.sum())

    elif args.env == 'SoccerScoreGoal-v0':
        import gym_soccer

        env = gym.make('SoccerScoreGoal-v0')
        print("Done making")
        env = SoccerScaledParameterisedActionWrapper(env)

        # save_dir = os.path.join(args.soccer_dir, "{}".format(str(args.seed)))
        # os.makedirs(save_dir, exist_ok=True)
        # env = Monitor(env, directory=save_dir, video_callable=False, write_upon_reset=False, force=True)

        state_dim = env.observation_space.spaces[0].shape[0]
        discrete_action_dim = env.action_space.spaces[0].n
        action_parameter_sizes = np.array(
			[env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
        parameter_action_dim = int(action_parameter_sizes.sum())

    else:
        raise f"Bad domain {args.env}, not implemented!"

    env.seed(args.seed)
    
    args.ub = 1.0
    args.lb = -1.0

    print("state_dim", state_dim)
    print("discrete_action_dim", discrete_action_dim)
    print("parameter_action_dim", parameter_action_dim)
    # exit()

    args.state_dim = state_dim
    args.k_dim = discrete_action_dim
    args.all_z_dim = parameter_action_dim
    args.par_size = action_parameter_sizes

    args.z_dim = action_parameter_sizes.max()
    args.action_dim = args.k_dim + args.z_dim
    args.max_action = args.ub
    
    args.scale = args.ub - args.lb
    args.offsets = args.lb
    args.offset = [args.par_size[:i].sum() for i in range(args.k_dim)]

    return env, args

def linear_schedule(schdl, step):
	match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
	init, final, duration = [float(g) for g in match.groups()]
	mix = np.clip(step / duration, 0.0, 1.0)
	return (1.0 - mix) * init + mix * final

class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_obs):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		dtype = torch.float32 
		self.z_dim = cfg.z_dim
		
		self.obs = torch.zeros((cfg.episode_length+1, *init_obs.shape), dtype=dtype, device=self.device)
		self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
		
		self.k = torch.zeros((cfg.episode_length), dtype=torch.int64, device=self.device)
		self.z = torch.zeros((cfg.episode_length, cfg.z_dim), dtype=torch.float32, device=self.device)
		
		self.reward = torch.zeros((cfg.episode_length,), dtype=torch.float32, device=self.device)
		self.continous = torch.zeros((cfg.episode_length,), dtype=torch.float32, device=self.device)
		
		self.cumulative_reward = 0
		self.done = False
		self._idx = 0
	
	def __len__(self):
		return self._idx

	@property
	def first(self):
		return len(self) == 0
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, k, z, reward, done):
		# print(k, z, reward, done)
		if len(z) < self.z_dim:
			new_z = torch.zeros([self.z_dim])
			new_z[:len(z)] = z
		else:
			new_z = z
		self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
		self.k[self._idx] = k
		self.z[self._idx] = new_z
		self.reward[self._idx] = reward
		self.continous[self._idx] = 1. - done

		self.cumulative_reward += reward
		self.done = done
		self._idx += 1
		
	def end(self):
		pass


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.num_classes = cfg.k_dim
		self.device = torch.device(cfg.device)
		self.capacity = min(cfg.max_timesteps, cfg.max_buffer_size)  # 1e5
		dtype = torch.float32
		# obs_shape = cfg.obs_shape if cfg.modality == 'state' else (3, *cfg.obs_shape[-2:])
		
		self._obs = torch.zeros((self.capacity+1, cfg.state_dim), dtype=dtype, device=self.device)
		self._last_obs = torch.zeros((self.capacity//cfg.episode_length, cfg.state_dim), dtype=dtype, device=self.device)
		
		self._k = torch.zeros((self.capacity, cfg.k_dim), dtype=torch.float32, device=self.device)
		self._z = torch.zeros((self.capacity, cfg.z_dim), dtype=torch.float32, device=self.device)

		self._reward = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
		self._continuous = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)

		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)

		self._eps = 1e-6
		self._full = False
		self.idx = 0

		# self.mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.mpc_horizon

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] 
		self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-1]
		
		self._k[self.idx:self.idx+self.cfg.episode_length] = torch.nn.functional.one_hot(episode.k, num_classes=self.num_classes)
		self._z[self.idx:self.idx+self.cfg.episode_length] = episode.z
		
		self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
		self._continuous[self.idx:self.idx+self.cfg.episode_length] = episode.continous

		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		
		mask = torch.arange(self.cfg.episode_length) >= min(self.cfg.episode_length-self.cfg.mpc_horizon, episode.continous.sum().item()+1)
		new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
		new_priorities[mask] = 0
		# print(new_priorities)
		# print(episode.k[0], episode.z[0])
		
		self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
		self.idx = (self.idx + self.cfg.episode_length) % self.capacity
		self._full = self._full or self.idx == 0

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.flatten().to(self.device) + self._eps

	def sample(self):
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()
		# print(idxs)
		# print("\n")
		horizon = self.cfg.mpc_horizon+1
		# horizon = self.cfg.mpc_horizon

		obs = self._obs[idxs]
		next_obs = torch.zeros((horizon, self.cfg.batch_size, *self._obs.shape[1:]), dtype=obs.dtype, device=obs.device)

		k = torch.zeros((horizon, self.cfg.batch_size, *self._k.shape[1:]), dtype=torch.float32, device=self.device)
		z = torch.zeros((horizon, self.cfg.batch_size, *self._z.shape[1:]), dtype=torch.float32, device=self.device)

		reward = torch.zeros((horizon, self.cfg.batch_size), dtype=torch.float32, device=self.device)
		continuous = torch.zeros((horizon, self.cfg.batch_size), dtype=torch.float32, device=self.device)
		trainmask = torch.ones((self.cfg.batch_size), dtype=torch.float32, device=self.device)
		trainmasks = torch.zeros((horizon, self.cfg.batch_size), dtype=torch.float32, device=self.device)

		for t in range(horizon):
			_idxs = idxs + t
			next_obs[t] = self._obs[_idxs+1]

			k[t] = self._k[_idxs]
			z[t] = self._z[_idxs]

			reward[t] = self._reward[_idxs]
			continuous[t] = self._continuous[_idxs]
			
			trainmasks[t] = trainmask
			trainmask = trainmask * continuous[t]

			# print(t, _idxs[0], k[t][0], z[t][0], continuous[t][0], trainmasks[t][0])

		# print(idxs[0], _idxs[0], horizon)
		mask = (_idxs+1) % self.cfg.episode_length == 0
		# print(mask[0])
		next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].cuda().float()

		return obs, next_obs, k, z, reward.unsqueeze(2), idxs, weights, continuous, trainmasks






