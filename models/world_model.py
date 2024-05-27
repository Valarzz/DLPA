import numpy as np
import torch
import torch.nn as nn

import models.model_utils as u
from models.model_utils import device
import models.networks as nets


class world_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.args = args
        self.timestep = 0
        self.max_timestep = args.max_timesteps

        self.env = args.env
        self.s_dim = args.state_dim
        self.k_dim = args.k_dim
        self.z_dim = args.z_dim

        self.inp_dim = args.state_dim + args.k_dim + args.z_dim

        if args.model_type == "concat":
            model_inp_dim = self.inp_dim
            s_out_dim, r_out_dim, c_out_dim = self.s_dim, 1, 2
            model_layers = [args.layers for _ in range(2)]
        elif args.model_type == "multi":
            model_inp_dim = (self.s_dim+self.z_dim)
            s_out_dim, r_out_dim, c_out_dim = self.s_dim * self.k_dim, 1 * self.k_dim, 2 * self.k_dim
            model_layers = [args.layers, args.layers * self.k_dim]
        elif args.model_type == "overlay":
            model_inp_dim = (self.s_dim+self.k_dim)
            s_out_dim, r_out_dim, c_out_dim = self.s_dim, 1, 2
            model_layers = [args.layers for _ in range(2)]
            
        if args.env in ['Platform-v0', 'Goal-v0', 'hard_goal-v0']:
            s_tanh = True
        elif args.env in ['simple_catch-v0', 'simple_move_4_direction_v1-v0']:
            s_tanh = False
        else:
            raise f"ENV {args.env} not implemented yet"
        
        self._dyanmics = nets.TanhGaussianPolicy(hidden_sizes=model_layers, 
                                                oup_dim=s_out_dim, 
                                                inp_dim=model_inp_dim,
                                                tanh=s_tanh, model_type=args.model_type, config=args).to(device)
        
        if args.env == 'Platform-v0':
            r_tanh = True
        elif args.env in ['Goal-v0', 'simple_catch-v0', 'hard_goal-v0', 'simple_move_4_direction_v1-v0']:
            r_tanh = False
        else:
            raise f"ENV {args.env} not implemented yet"
        self._reward = nets.TanhGaussianPolicy(hidden_sizes=model_layers, 
                                                oup_dim=r_out_dim, 
                                                inp_dim=model_inp_dim,
                                                tanh=r_tanh, model_type=args.model_type, config=args).to(device)
        self._reward1 = nets.TanhGaussianPolicy(hidden_sizes=model_layers, 
                                                oup_dim=r_out_dim, 
                                                inp_dim=model_inp_dim,
                                                tanh=r_tanh, model_type=args.model_type, config=args).to(device)
        

        self._continue = nets.TanhGaussianPolicy(hidden_sizes=model_layers, 
                                                oup_dim=c_out_dim, 
                                                inp_dim=model_inp_dim,
                                                tanh=False, model_type=args.model_type, config=args).to(device)

    def linear_temp(self):
        return max((1 - (self.timestep / 20_000)) * 4.5 + 0.5, 0.5)  # (0, 1) -> (5, 0.5)
    
    def pi(self, s, std, reparameterize, return_log_prob, deterministic):
        '''
        inp:
        s: [N_policy_traj, s_dim]
        oup:
        dpolicy_outputs: [N_policy_traj, k_dim]
        cpolicy_outputs: [N_policy_traj, z_dim]

                        reparameterize  return_log_prob deterministic
        train           True            True            False
        train_plan      False           True            False    
        evaluate_plan   -               -               True
        TD_target       -               -               True
        estimate_value  -               -               True
        '''
        dpolicy_outputs, cpolicy_outputs = self._agent(s, reparameterize=reparameterize, return_log_prob=return_log_prob, deterministic=deterministic, temperature=self.linear_temp())

        return dpolicy_outputs, cpolicy_outputs[0]
    
    
    def next(self, s, a, reparameterize, return_log_prob, deterministic):
        '''
                        reparameterize  return_log_prob deterministic
            train           True            True            False
            train_plan      False           True            False    
            evaluate_plan   -               -               True
            estimate_value  -               -               True
        '''
        k = a[:, :self.k_dim]
        z = a[:, self.k_dim:]
        
        s = self._dyanmics(s, k, z, reparameterize=reparameterize, return_log_prob=return_log_prob, deterministic=deterministic)[0]

        r0 = self._reward(s, k, z, reparameterize=reparameterize, return_log_prob=return_log_prob, deterministic=deterministic)[0]
        r1 = self._reward1(s, k, z, reparameterize=reparameterize, return_log_prob=return_log_prob, deterministic=deterministic)[0] if self.env in ['simple_catch-v0', 'simple_move_4_direction_v1-v0'] else None

        if self.env == 'Platform-v0':
            r0 = (r0 + 1) / 2

        c = self._continue(s, k, z, reparameterize=reparameterize, return_log_prob=return_log_prob, deterministic=deterministic)[0]

        return s, r0, c, r1

