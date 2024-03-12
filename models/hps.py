import torch
import torch.nn as nn
import numpy as np
import models.model_utils as u
import models.networks as nets


class HPS(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.args = args
        self.discrete_policy = nets.FlattenMlp(
            hidden_sizes=[args.pi_layers, args.pi_layers],
            input_size=args.state_dim,
            output_size=args.k_dim,
        ).to(u.device)
        self.continuous_policy = nets.TanhGaussianPolicy(
            hidden_sizes=[args.pi_layers, args.pi_layers],
            inp_dim=args.state_dim + args.k_dim,
            oup_dim=args.z_dim,
        ).to(u.device)

    def forward(self, obs, discrete_act=None, detach=False,
                reparameterize=False,
                deterministic=False,
                return_log_prob=False,
                temperature=1.0): 
        # used during training given batch of experience (obs) output actions?
        ''' given context, get statistics under the current policy of a set of observations '''
        b, _ = obs.size()

        # run policy, get log probs and new actions
        if discrete_act is None:
            discrete_action = self.discrete_policy(obs)
            discrete_act = u.gumbel_softmax(discrete_action, temperature=temperature, hard=True, deterministic=deterministic)
            #discrete_act = softmax(discrete_action, hard=True, eps=0.1)
        if detach:
            in_ = torch.cat([obs, discrete_act.detach()], dim=1)
        else:
            in_ = torch.cat([obs, discrete_act], dim=1)
            
        cpolicy_outputs = self.continuous_policy(in_, reparameterize=reparameterize, return_log_prob=return_log_prob, deterministic=deterministic)

        return discrete_act, cpolicy_outputs











