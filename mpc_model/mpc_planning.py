import numpy as np
import copy
import scipy.stats as stats
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import Dataset, DataLoader
from mpc_model.hard_code import worldmodel as platform_model
from mpc_model.hard_code_goal import worldmodel as goal_model
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_boundary(c_rate):
    median = (c_rate[0] - c_rate[1]) / 2
    offset = c_rate[0] - 1 * median
    return median, offset


def true_parameter_action(parameter_action, c_rate):
    parameter_action_ = copy.deepcopy(parameter_action)
    for i in range(len(c_rate)):
        median, offset = count_boundary(c_rate[i])
        parameter_action_[:, i] = parameter_action_[:, i] * median + offset
    return parameter_action_


def raw2embed(d_embed, p_embed, action_rep, k_dim, s):
    if action_rep.c_rate is not None:
        p_embed = true_parameter_action(p_embed, action_rep.c_rate)
    p_embed = p_embed.float().to(action_rep.device)

    d_embed = torch.clamp(d_embed, -1, 1)
    d = action_rep.select_discrete_action(d_embed.float())

    d_embed = action_rep.get_embedding(d).float().to(action_rep.device)
    if len(d_embed.shape) < 2:
        d_embed = d_embed.unsqueeze(0)

    with torch.no_grad():
        p, _ = action_rep.vae.decode(s.to(action_rep.device), p_embed, d_embed)

    d = d if isinstance(d, torch.Tensor) else torch.tensor(d)
    d = torch.nn.functional.one_hot(d, num_classes=k_dim)
    p = torch.clamp(p, -1, 1).cpu()
    return d, p


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class RandomOptimizer(Optimizer):
    def __init__(self, horizon, dis_dim, par_dim, par_size, popsize, upper_bound=None, lower_bound=None, max_iters=10, num_elites=100, epsilon=0.001, alpha=0.25):
        super().__init__()
        self.h, self.k_dim, self.z_dim, self.max_iters, self.popsize, self.num_elites = horizon, dis_dim, par_dim, max_iters, popsize, num_elites
        self.par_size = par_size
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

    def setup(self, cost_function):
        self.cost_function = cost_function

    def reset(self):
        pass

    def obtain_solution(self, kinit_mean, zinit_mean, init_var, onepa, s, embed=False, action_rep=None, debug=False):

        if embed:
            d_embed = torch.rand([self.popsize*self.h, action_rep.reduced_action_dim]) * (self.ub - self.lb) + self.lb
            p_embed = torch.rand([self.popsize*self.h, action_rep.reduce_parameter_action_dim]) * (self.ub - self.lb) + self.lb

            if action_rep.c_rate is not None:
                p_embed = true_parameter_action(p_embed, action_rep.c_rate)
            p_embed = p_embed.float().to(action_rep.device)

            d = action_rep.select_discrete_action(d_embed.float())

            d_embed = action_rep.get_embedding(d).float().to(action_rep.device)
            s = torch.from_numpy(np.tile(s, [self.popsize*self.h, 1])).float().to(action_rep.device)

            with torch.no_grad():
                p, _ = action_rep.vae.decode(s, p_embed, d_embed)

            d = torch.nn.functional.one_hot(torch.from_numpy(d), num_classes=self.k_dim)
            p = torch.clamp(p, self.lb[0], self.ub[0]).cpu()
            
            solutions = torch.cat([d, p], dim=-1).reshape([self.popsize, self.h, -1]).numpy()

        else:
            dd = torch.randint(0, self.k_dim, [self.popsize*self.h])
            d = torch.nn.functional.one_hot(dd, num_classes=self.k_dim)
            if onepa:
                mask = torch.zeros([self.popsize*self.h, self.par_size.max()])
                for i in range(self.popsize*self.h):
                    mask[i, :self.par_size[dd[i]]] = 1
                p = torch.rand([self.popsize*self.h, self.par_size.max()]) * (self.ub - self.lb) + self.lb
                p *= mask
            else:
                pp = torch.rand([self.popsize*self.h, self.z_dim]) * (self.ub - self.lb) + self.lb
                p = pp * d

            # print(self.ub, self.lb, self.z_dim)
            solutions = torch.cat([d, p], dim=-1).numpy()
            solutions = solutions.reshape([self.popsize, self.h, -1])
        
        costs, pred_s = self.cost_function(solutions)
        soln = solutions[np.argmin(costs)]  # [horizon, action_dim]
        
        ksoln = soln[0][:self.k_dim]
        zsoln = soln[0][self.k_dim:self.k_dim+self.z_dim]

        # print(ksoln, zsoln)
        # if onepa:
        #     zsoln = np.array([zsoln[0] for _ in range(self.z_dim)])
        
        return ksoln[np.newaxis, :], zsoln[np.newaxis, :], None, pred_s


def new_mean_var(kelites, zelites, h, k_dim, z_dim, pop, zmean, zvar, par_size):
    # print(kelites.shape, zelites.shape, pop, h)
    kelites = kelites.reshape([pop, -1])
    zelites = zelites.reshape([pop, -1])
    zmean = zmean.reshape([-1])
    zvar = zvar.reshape([-1])
    # print(kelites.shape, zelites.shape, zmean.shape, zvar.shape)

    samples = [[] for _ in range(h * z_dim)]
    for i in range(pop):
        for j in range(h * k_dim):
            if kelites[i][j] != 0:
                h_i = j // k_dim
                k_i = j % k_dim
                num = par_size[k_i]
                jz = h_i * z_dim + sum(par_size[:k_i])
                for m in range(num):
                    samples[jz+m].append(zelites[i][jz+m])

    mean = []
    var = []
    for i, sample in enumerate(samples):
        if sample != []:
            sample = np.array(sample)
            mean.append(sample.mean())
            var.append(sample.var())
        else:
            mean.append(zmean[i])
            var.append(zvar[i])

    return np.array(mean).reshape(h, z_dim), np.array(var).reshape(h, z_dim)


class CEMOptimizer(Optimizer):
    """A Pytorch-compatible CEM optimizer.
    """
    def __init__(self, horizon, dis_dim, par_dim, par_size, popsize, upper_bound=None, lower_bound=None, max_iters=10, num_elites=100, epsilon=0.001, alpha=0.25):
        super().__init__()
        self.h, self.k_dim, self.z_dim, self.max_iters, self.popsize, self.num_elites = horizon, dis_dim, par_dim, max_iters, popsize, num_elites
        self.par_size = par_size
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.all_z_dim = sum(par_size)
        self.offset = [par_size[:i].sum() for i in range(self.k_dim)]

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.mean, self.var = None, None
        self.cost_function = None

        # self.sol_dim = 

    def setup(self, cost_function):
        self.cost_function = cost_function

    def reset(self):
        pass

    # def obtain_solution(self, kinit_mean, zinit_mean, init_var, onepa, state, embed=False, action_rep=None, debug=False, dm=None, rm=None, cm=None, training=True):
    def obtain_solution(self, kinit_mean, zinit_mean, init_var, onepa, state, embed=False, action_rep=None, debug=False):
        """
        kinit_mean: [horizon, k_dim]
        zinit_mean, init_var: [horizon, z_dim]
        """
        kmean, zmean, zvar, t = kinit_mean, zinit_mean, init_var, 0
        
        if embed:
            Xz = stats.truncnorm(-2, 2, loc=np.zeros_like(zmean), scale=np.ones_like(zmean))
            Xk = stats.truncnorm(-2, 2, loc=np.zeros_like(kmean), scale=np.ones_like(kmean))
            s = torch.from_numpy(np.tile(state, [self.popsize*self.h, 1])).float()

            zvar = np.tile(np.square(self.lb[0] - self.ub[0]) / 16, [self.h, action_rep.reduce_parameter_action_dim])
            kvar = np.tile(np.square(self.lb[0] - self.ub[0]) / 16, [self.h, action_rep.reduced_action_dim])

            while (t < self.max_iters) and ((zvar.min() > self.epsilon) or (kvar.min() > self.epsilon)):
                zlb_dist, zub_dist = zmean - self.lb, self.ub - zmean
                constrained_zvar = np.minimum(np.minimum(np.square(zlb_dist / 2), np.square(zub_dist / 2)), zvar)
                raw_p_embed = Xz.rvs(size=[self.popsize, self.h, action_rep.reduce_parameter_action_dim]) * np.sqrt(constrained_zvar.astype(np.float32)) + zmean

                klb_dist, kub_dist = kmean - self.lb, self.ub - kmean
                constrained_kvar = np.minimum(np.minimum(np.square(klb_dist / 2), np.square(kub_dist / 2)), kvar)
                raw_d_embed = Xk.rvs(size=[self.popsize, self.h, action_rep.reduced_action_dim]) * np.sqrt(constrained_kvar.astype(np.float32)) + kmean

                d_embed = torch.from_numpy(raw_d_embed.reshape([-1, action_rep.reduced_action_dim]))
                p_embed = torch.from_numpy(raw_p_embed.reshape([-1, action_rep.reduce_parameter_action_dim]))

                d, p = raw2embed(d_embed, p_embed, action_rep, self.k_dim, s)

                solutions = torch.cat([d, p], dim=-1).reshape([self.popsize, self.h, -1]).numpy()

                costs, pred_s = self.cost_function(solutions)
                idx = np.argsort(costs)

                kelites = raw_d_embed[idx][:self.num_elites]
                zelites = raw_p_embed[idx][:self.num_elites]
                
                new_kmean, new_kvar = kelites.mean(0), kelites.var(0)
                new_zmean, new_zvar = zelites.mean(0), zelites.var(0)

                zmean = self.alpha * zmean + (1 - self.alpha) * new_zmean  # [h, d_dim]
                zvar = self.alpha * zvar + (1 - self.alpha) * new_zvar  # [h, d_dim]

                kmean = self.alpha * kmean + (1 - self.alpha) * new_kmean  # [h, d_dim]
                kvar = self.alpha * kvar + (1 - self.alpha) * new_kvar  # [h, d_dim]

            # k, z = raw2embed(torch.from_numpy(kmean[0]).unsqueeze(0), torch.from_numpy(zmean[0]).unsqueeze(0), action_rep, self.k_dim, torch.from_numpy(state).unsqueeze(0))
            return kmean, zmean, None, None

        else:
            # kmean, zmean, zvar, pred_s = self.from_tdmpc(kinit_mean, zinit_mean, init_var, onepa, state, embed, action_rep, debug, dm, rm, cm, training)
            # return kmean, zmean, zvar, pred_s

            kmean = torch.from_numpy(kmean)
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(zmean), scale=np.ones_like(zmean))
            while (t < self.max_iters) and (zvar.min(1).max() > self.epsilon):
                lb_dist, ub_dist = zmean - self.lb, self.ub - zmean
                constrained_zvar = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), zvar)
                
                ksamples = torch.nn.functional.one_hot(torch.multinomial(kmean, self.popsize, replacement=True).T, num_classes=self.k_dim).reshape([self.popsize, self.h, -1]).numpy()

                zsamples = X.rvs(size=[self.popsize, self.h, self.z_dim]) * np.sqrt(constrained_zvar.astype(np.float32)) + zmean
                zsamples = zsamples.astype(np.float32).reshape([self.popsize, self.h, -1])
                
                # zsamples *= ksamples

                if onepa:
                    mask = np.zeros([self.popsize, self.h, self.par_size.max()])
                    for i in range(self.popsize):
                        for j in range(self.h):
                            endd = self.par_size[ksamples[i, j].argmax()]
                            mask[i, j, :endd] = zsamples[i, j, :endd]
                    samples = np.concatenate([ksamples, mask], axis=-1)
                else:
                    samples = np.concatenate([ksamples, zsamples], axis=-1)
                
                costs, pred_s = self.cost_function(samples)  # samples: [pop_size, horizon, action_dim]
                idx = np.argsort(costs)

                kelites = ksamples[idx][:self.num_elites]  # [num_elites, h, k_dim]
                new_kmean = kelites.sum(0)  # [h, d_dim]
                new_kmean = torch.from_numpy(new_kmean / self.num_elites)  # [h, k_dim]

                kmean = self.alpha * kmean + (1 - self.alpha) * new_kmean  # [h, k_dim]
                kmean = kmean / kmean.sum(-1).unsqueeze(1)
                # print(kmean)

                zelites = zsamples[idx][:self.num_elites]  # [num_elites, h, z_dim]

                new_zmean, new_zvar = new_mean_var(kelites, zelites, self.h, self.k_dim, self.z_dim, self.num_elites, zmean, zvar, self.par_size)

                zmean = self.alpha * zmean + (1 - self.alpha) * new_zmean  # [h, d_dim]
                zvar = self.alpha * zvar + (1 - self.alpha) * new_zvar  # [h, d_dim]
                # print(zmean)

                t += 1
            
            return kmean.numpy(), zmean, zvar, pred_s

    def from_tdmpc(self, kinit_mean, zinit_mean, init_var, onepa, state, embed=False, action_rep=None, debug=False, dm=None, rm=None, cm=None, training=True):
        self.z_dim = max(self.par_size)
        self.lb, self.ub = -1., 1.
        mean = {'k': torch.from_numpy(kinit_mean).to(device), 'z': torch.from_numpy(zinit_mean).to(device)}
        horizon = self.h
        s = torch.from_numpy(state).repeat(self.popsize, 1).to(device)
        std = torch.from_numpy(init_var).to(device)

        for i in range(10):
            actions = self.sample_from_N(mean, std)
                
            # Compute elite actions
            value = self.estimate_value(s, actions, horizon, dm, rm, cm, training).nan_to_num_(0).to(device)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            elite_value = value[elite_idxs]  # [num_elite, 1]
            elite_actions = actions[:, elite_idxs]  # [horizon, num_elite, a_dim]

            max_value = elite_value.max(0)[0]

            # Update k parameters
            # k_score is k weights, softmax(elite_value-max)
            k_score = torch.exp(0.5*(elite_value - max_value))
            k_score /= k_score.sum(0)  # [num_elite, 1]
            kelites = elite_actions[:, :, :self.k_dim]
            _kmean = torch.sum(k_score.unsqueeze(0) * kelites, dim=1) / (k_score.sum(0) + 1e-9)

            # Update z parameters
            zelites = elite_actions[:, :, self.k_dim:]
            k_all = kelites.argmax(-1).unsqueeze(-1)  # [horizon, num_elite, 1]
            z_score = elite_value.unsqueeze(0).repeat([horizon, 1, 1])  # [horizon, num_elite, 1]
            _zmean, _std = torch.empty_like(mean['z']), torch.empty_like(std)

            for ki in range(self.k_dim):
                selected_ind = (k_all == ki)  # selected discrete type, [horizon, num_elite, 1]
                zis = zelites[:, :, :self.par_size[ki]]
                # zi: [horizon, num_elite, z_dim], = zi if selected else 0
                zi = torch.where(selected_ind, zis, torch.zeros_like(zis).to(device))

                # weight: [horizon, num_elite, z_dim], = softmax(selected(z))
                weight = torch.where(selected_ind, z_score, torch.tensor([float("-Inf")]).to(device))
                weight = torch.exp(0.5*(weight - max_value))
                weight_sum = weight.squeeze(-1).sum(1).reshape([-1, 1, 1]).repeat(1, self.num_elites, 1)
                weight /= (weight_sum + 1e-9)
                
                _zimean = torch.sum(weight * zi, dim=1) / (weight.sum(1) + 1e-9)
                _zistd = torch.sqrt(torch.sum(weight * (zi - _zimean.unsqueeze(1)) ** 2, dim=1) / (weight.sum(1) + 1e-9))

                ind_start = self.offset[ki]
                ind_end = ind_start + self.par_size[ki]

                if_non_select = selected_ind.squeeze(-1).sum(1).unsqueeze(-1)
                _zimean = torch.where(if_non_select==0, mean['z'][:, ind_start:ind_end], _zimean)
                _zistd = torch.where(if_non_select==0, std[:, ind_start:ind_end], _zistd)

                _zmean[:, ind_start:ind_end] = _zimean
                _std[:, ind_start:ind_end] = _zistd

            mean['k'] = self.alpha * mean['k'] + (1 - self.alpha) * _kmean
            mean['z'] = self.alpha * mean['z'] + (1 - self.alpha) * _zmean
            std = self.alpha * std + (1 - self.alpha) * _std

        return mean['k'].cpu().numpy(), mean['z'].cpu().numpy(), std.cpu().numpy(), None

    def sample_from_N(self, mean, std):
        kmean = mean['k']
        zmean, zstd = mean['z'], std

        k_int = torch.multinomial(kmean, self.popsize, replacement=True)
        k_onehot = torch.nn.functional.one_hot(k_int, num_classes=self.k_dim).to(device)
        
        z_all = torch.clamp(zmean.unsqueeze(1) + zstd.unsqueeze(1) * \
                torch.randn(self.h, self.popsize, self.all_z_dim, device=zstd.device), self.lb, self.ub)
        
        offsets = torch.tensor(self.offset).to(device)[k_int.flatten()].unsqueeze(-1).repeat(1, self.z_dim) + torch.arange(self.z_dim, device=device)
        z_one = torch.zeros([self.h*self.popsize, self.all_z_dim+self.z_dim-1], device=device)
        z_one[:, :self.all_z_dim] = z_all.reshape([-1, self.all_z_dim])
        
        zs = torch.gather(z_one, 1, offsets)
        
        size = torch.from_numpy(self.par_size).to(device)[k_int.flatten()].unsqueeze(-1).repeat(1, self.z_dim)
        mask = torch.arange(self.z_dim).to(device).repeat(len(size), 1)
        mask = torch.where(mask<size, 1., 0.)
        zs = zs * mask
        
        zs = zs.reshape([self.h, self.popsize, self.z_dim])
        return torch.cat([k_onehot, zs], dim=-1)
        
    def estimate_value(self, s, actions, horizon, dm, rm, cm, training=True):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        # actions = actions.cpu().numpy()
        allc = torch.ones([self.popsize, 1])
        for t in range(horizon):
            # print(s.shape, actions.shape)
            # s = s.cpu().numpy()
            s, reward, c = dm.predict(s, actions[t], training), rm.predict(s, actions[t], training), cm.predict(s, actions[t], training)
            print(reward.shape)
            exit()
            G += discount * reward * allc
        # print(G.shape)
        return torch.from_numpy(G)


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}
    def __init__(self, args, action_rep=None):
        if args.env == "Platform-v0":
            self.wm = platform_model("all")
        elif args.env == "Goal-v0":
            self.wm = goal_model("all")

        self.envname = args.env
        
        self.horizon = args.mpc_horizon
        self.gamma = args.mpc_gamma

        self.action_low = np.array([-args.max_action])  # array (dim,)
        self.action_high = np.array([args.max_action])  # array (dim,)
        
        self.dis_dim = args.discrete_action_dim
        self.par_dim = args.parameter_action_dim
        self.par_size = args.action_parameter_sizes

        self.popsize = args.mpc_popsize
        self.state_dim = args.state_dim
        
        self.mpc_type = args.mpc_type

        self.sparse = args.sparse
        
        self.use_terminal = args.use_terminal

        self.particle = args.mpc_patrical

        self.onepa = args.dm_onepa

        self.embed = args.embed
        self.action_rep = action_rep

        self.action_dim = self.dis_dim + self.par_size.max() if self.onepa else self.dis_dim + self.par_dim

        self.init_mean = np.array([args.mpc_init_mean] * self.horizon)  # 0
        self.init_var = np.array([args.mpc_init_var] * self.horizon)  # 1

        if len(self.action_low) == 1:  # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.par_dim])
            self.action_high = np.tile(self.action_high, [self.par_dim])

        self.optimizer = MPC.optimizers[self.mpc_type](horizon=self.horizon, 
                                                       dis_dim=self.dis_dim,
                                                       par_dim=self.par_dim,
                                                       par_size = self.par_size,
                                                       popsize=self.popsize, 
                                                       upper_bound=np.array([args.max_action]),
                                                       lower_bound=np.array([-args.max_action]),
                                                       max_iters=args.mpc_max_iters, 
                                                       num_elites=args.mpc_num_elites,
                                                       epsilon=args.mpc_epsilon, alpha=args.mpc_alpha)
        self.reset()

    def reset(self):
        if self.embed:
            self.zprob = np.zeros([self.action_rep.reduce_parameter_action_dim])
            self.prev_zsol = np.tile(self.zprob, [self.horizon, 1])

            self.kprob = np.zeros([self.action_rep.reduced_action_dim])
            self.prev_ksol = np.tile(self.kprob, [self.horizon, 1])

            self.init_zvar = np.tile(np.square(self.action_low[0] - self.action_high[0]) / 16, [self.horizon, 1])

        else:
            self.zprob = (self.action_low + self.action_high) / 2
            self.prev_zsol = np.tile(self.zprob, [self.horizon, 1])
            self.init_zvar = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon, 1])

            kprob = np.array([1 for _ in range(self.dis_dim)])
            self.kprob = kprob / kprob.sum()
            self.prev_ksol = np.tile(self.kprob, [self.horizon, 1])  # [horizon, k_dim]

    def act(self, dm, rm, cm, state, action_rep, training, debug=False):
        self.model = dm
        self.reward_model = rm
        self.state = state
        self.continuous = cm
        self.training = training

        self.optimizer.setup(self.cost_function)

        ksoln, zsoln, var, pred_s = self.optimizer.obtain_solution(self.prev_ksol, self.prev_zsol, self.init_zvar, self.onepa, state, self.embed, action_rep, debug=debug)

        # ksoln, zsoln, var, pred_s = self.optimizer.obtain_solution(self.prev_ksol, self.prev_zsol, self.init_zvar, self.onepa, state, self.embed, action_rep, debug=debug, dm=dm, rm=rm, cm=cm, training=training)
        # print(ksoln[0], zsoln[0])
        # print(pred_s)
        # exit()

        if self.mpc_type == "CEM":
            # print()
            self.prev_ksol = np.concatenate([np.copy(ksoln)[1:], self.kprob[np.newaxis, :]])
            self.prev_zsol = np.concatenate([np.copy(zsoln)[1:], self.zprob[np.newaxis, :]])
        else:
            pass
        
        if self.embed and self.mpc_type == "CEM":
            d, z = raw2embed(torch.from_numpy(ksoln[0]).unsqueeze(0), torch.from_numpy(zsoln[0]).unsqueeze(0), action_rep, self.dis_dim, torch.from_numpy(state).unsqueeze(0))
            p = np.zeros(self.par_dim)
            offset = self.par_size[:d.argmax()].sum()
            p[offset:offset+self.par_size[d.argmax()]] = z[0][:self.par_size[d.argmax()]]
            return d.numpy(), p, pred_s

        d = ksoln[0]
        p = zsoln[0]
        # p = np.zeros(self.par_dim)
        # # print(d, self.par_size[:d.argmax()].sum())
        # offset = self.par_size[:d.argmax()].sum()
        # p[offset:offset+self.par_size[d.argmax()]] = zsoln[0][offset:offset+self.par_size[d.argmax()]]
        # print(d, d.argmax(), self.par_size, offset, offset+self.par_size[d.argmax()])
        # print(d, p)
        
        return d, p, pred_s

    def cost_function(self, actions):  # action: [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize * self.particle) - 1_000.
        state = np.repeat(self.state.reshape(1, -1), self.popsize * self.particle, axis=0)
        c = np.ones(self.popsize * self.particle)  # 1: continuous, 0: terminal

        all_pred_state = []
        all_pred_reward = []
        flag = False

        for t in range(0, self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)

            if self.model.mode or self.continuous.mode or self.reward_model.mode:
                state_next = []
                ct = []
                cost = []
                for i in range(self.popsize * self.particle):
                    if c[i]:  # env continue
                        state_next_i, r_i, ct_i = self.wm.predict(state[i, :], action[i, :])
                        # if ct_i == 1 and r_i == 0:
                        #     print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        # elif ct_i == 0 and r_i != 0:
                        #     print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    else:
                        state_next_i = [0 for _ in range(self.state_dim)]
                        ct_i = True  # terminal
                        r_i = -1000
                    state_next.append(state_next_i)
                    ct.append(ct_i)
                    cost.append(r_i)

            if self.model.mode:
                state_next = np.array(state_next)
            else:
                state_next = self.model.predict(state, action, self.training)
                state_next += state

            if self.continuous.mode:
                ct = np.array(ct)  # 1: terminal
            else:
                ct = self.continuous.predict(state, action, self.training)
                ct = ct.argmax(-1)  # 1: terminal

            if self.reward_model.mode:
                cost = np.array(cost)
            else:
                cost = self.reward_model.predict(state, action, self.training)  # compute cost, sort is increasing, so -
                cost = cost.reshape(costs.shape)

            if self.sparse:
                if self.use_terminal:
                    c_prev = copy.deepcopy(c)
                    c *= (1-ct)  # 1: continuous
                    cost = cost * (c != c_prev).astype(float)
                    costs -= cost * (self.gamma ** t)  # self.gamma == 1
                    if np.all(1-c):  # 1: terminal, if all terminated, break
                        # print(c)
                        flag = True
                else:
                    costs -= cost * (self.gamma ** t)  # self.gamma == 1

            else:
                if self.use_terminal:
                    cost *= c
                    
                costs -= cost * (self.gamma ** t)  # self.gamma == 1

                if self.use_terminal:
                    c *= (1-ct)  # 1: continuous
                    if np.all(1-c):  # 1: terminal, if all terminated, break
                        flag = True

            state = copy.deepcopy(state_next)

            all_pred_state.append(state_next)
            # print(t, state_next.shape, np.all(1-c), self.horizon)
            all_pred_reward.append(cost)

            if flag:
                break

        # average between particles
        costs = np.sum(costs.reshape((self.particle, -1)), axis=0)

        # print(costs)
        best_ind = np.argmin(costs)
        all_pred_state = np.array(all_pred_state)
        pred_s = all_pred_state[0][best_ind]
        # print(best_ind, costs[best_ind], all_pred_state.shape, all_pred_reward.shape)
        # print("pred")
        # print(np.array(all_pred_reward)[:, best_ind])
        # print(pred_s)

        return costs, pred_s
        # return

