import torch
import torch.nn as nn
import numpy as np
import models.model_utils as u
import models.meta as meta
import torch.nn.functional as F
import models.model_utils as ptu


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))


def q(cfg, act_fn=nn.ELU()):
	"""
    Returns a Q-function that uses Layer Normalization.
    """
	return nn.Sequential(nn.Linear(cfg.state_dim+cfg.action_dim, cfg.q_dim), 
                         nn.LayerNorm(cfg.q_dim), 
                         nn.Tanh(),
						 nn.Linear(cfg.q_dim, cfg.q_dim), 
                         nn.ELU(),
						 nn.Linear(cfg.q_dim, 1))


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


def identity(x):
    return x


class Mlp(meta.PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-1,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
        

class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)
    

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, meta.ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            inp_dim,
            oup_dim,
            config,
            latent_dim=0,
            std=None,
            init_w=1e-3,
            tanh=True,
            model_type="concat",
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=inp_dim,
            output_size=oup_dim,
            init_w=init_w,
            **kwargs
        )
        self.tanh = tanh
        self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        self.model_type = model_type

        self.s_dim = config.state_dim
        self.k_dim = config.k_dim
        self.z_dim = config.z_dim
        self.multi_oup_dim = int(oup_dim / self.k_dim)

        if std is None:
            last_hidden_size = inp_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, oup_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        if self.model_type == "overlay":
            hidden_init=ptu.fanin_init
            b_init_value = 0.1

            self.dnn1 = nn.Linear(inp_dim, hidden_sizes[0])
            hidden_init(self.dnn1.weight)
            self.dnn1.bias.data.fill_(b_init_value)

            self.dnn2 = nn.Linear(hidden_sizes[0]+self.z_dim, hidden_sizes[1])
            hidden_init(self.dnn2.weight)
            self.dnn2.bias.data.fill_(b_init_value)

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return meta.np_ify(outputs)

    def forward(
            self,
            s, k, z,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        if self.model_type == "concat":
            h = torch.cat([s, k, z], dim=-1)

            for i, fc in enumerate(self.fcs):
                h = self.hidden_activation(fc(h))
            mean = self.last_fc(h)
            if self.std is None:
                log_std = self.last_fc_log_std(h)
                log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
                std = torch.exp(log_std)
            else:
                std = self.std
                log_std = self.log_std

        elif self.model_type == "multi":
            h = torch.cat([s, z], dim=-1)

            for i, fc in enumerate(self.fcs):
                h = self.hidden_activation(fc(h))
            mean = self.last_fc(h)
            if self.std is None:
                log_std = self.last_fc_log_std(h)
                log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
                std = torch.exp(log_std)
            else:
                std = self.std
                log_std = self.log_std

            idx = k.argmax(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.multi_oup_dim)
            
            mean = mean.view([-1, self.k_dim, self.multi_oup_dim])
            mean = torch.gather(mean, 1, idx)
            mean = mean.view(-1, self.multi_oup_dim)

            std = std.view([-1, self.k_dim, self.multi_oup_dim])
            std = torch.gather(std, 1, idx)
            std = std.view(-1, self.multi_oup_dim)

        elif self.model_type == "overlay":
            h = torch.cat([s, k], dim=-1)
            
            h = self.hidden_activation(self.dnn1(h))
            h = torch.cat([h, z], dim=-1)
            h = self.hidden_activation(self.dnn2(h))

            mean = self.last_fc(h)
            if self.std is None:
                log_std = self.last_fc_log_std(h)
                log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
                std = torch.exp(log_std)
            else:
                std = self.std
                log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        
        if deterministic:  # evaluate use this one
            action = torch.tanh(mean) if self.tanh else mean
        else:
            tanh_normal = ptu.TanhNormal(mean, std, tanh=self.tanh)
            if return_log_prob:
                if reparameterize:  # train use this one, in [-1, 1], tanh
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:  # plan use this one, in [-1, 1], tanh
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                    
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
                action = action if self.tanh else pre_tanh_value

            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()  

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )



