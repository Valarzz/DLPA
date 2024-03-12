import numpy as np
import torch
import torch.nn as nn
from mpc_model.utils import Model, MLPRegression, MLP
from torch.utils.data import Dataset, DataLoader
import wandb
import math


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def CPU(var):
    return var.cpu().detach()


class nDynamicModel(Model):
    def __init__(self, args, label):
        super().__init__()
        self.args = args
        self.state_dim = args.state_dim
        self.action_dim = args.discrete_action_dim + args.parameter_action_dim
        self.label = label

        self.onepa = args.dm_onepa
        if self.onepa:
            self.input_dim = self.state_dim + args.discrete_action_dim + args.action_parameter_sizes.max()
        else:
            self.input_dim = self.state_dim + self.action_dim  # state + action_type + all_action_para

        self.n_epochs = args.n_epochs

        if label == 'state':
            self.mode = args.dm_hard
            # self.n_epochs = args.dm_epoch  # supervised training epochs
            self.lr = args.dm_lr
            # self.batch_size = args.dm_batchsize

            self.save_model_flag = args.dm_saveflag
            self.save_model_path = args.dm_savepath

            self.validation_flag = args.dm_valflag
            self.validate_freq = args.dm_valfreq
            self.validation_ratio = args.dm_valrati
            # self.dm_continue = args.dm_continue

            self.load = args.dm_loadmodel
            self.path = args.dm_loadpath
            # self.model = CUDA(MLPRegression(self.input_dim, self.state_dim, args.dm_layers))
            self.model = CUDA(MLP(self.input_dim, self.state_dim, args.dm_layers, self.label, args.env))

            # self.criterion = nn.MSELoss(reduction='mean')
            self.criterion = nn.HuberLoss(reduction='mean')

        elif label == 'reward':
            self.mode = args.r_hard
            # self.n_epochs = args.r_epoch  # supervised training epochs
            self.lr = args.r_lr
            # self.batch_size = args.r_batchsize

            self.save_model_flag = args.r_saveflag
            self.save_model_path = args.r_savepath

            self.validation_flag = args.r_valflag
            self.validate_freq = args.r_valfreq
            self.validation_ratio = args.r_valrati

            self.load = args.r_loadmodel
            self.path = args.r_loadpath
            # self.model = CUDA(MLPRegression(self.input_dim, 1, args.r_layers))
            self.model = CUDA(MLP(self.input_dim, 1, args.r_layers, self.label, args.env))

            # self.criterion = nn.MSELoss(reduction='mean')
            self.criterion = nn.HuberLoss(reduction='mean')

        elif label == 'continuous':
            self.mode = args.c_hard
            # self.n_epochs = args.c_epoch  # supervised training epochs
            self.lr = args.c_lr
            # self.batch_size = args.c_batchsize

            self.save_model_flag = args.c_saveflag
            self.save_model_path = args.c_savepath

            self.validation_flag = args.c_valflag
            self.validate_freq = args.c_valfreq
            self.validation_ratio = args.c_valrati

            self.load = args.c_loadmodel
            self.path = args.c_loadpath
            # self.model = CUDA(MLPRegression(self.input_dim, 2, args.c_layers))
            self.model = CUDA(MLP(self.input_dim, 2, args.c_layers, self.label, args.env))

            self.criterion = nn.CrossEntropyLoss()

        if self.load:
            self.load_model(self.path)

        self.reset_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # self.mode = "dl"

    def predict(self, s, a, training):
        # convert to torch format
        # if torch.is_tensor(s):
        #     s = s.float()
        # else:
        #     s = CUDA(torch.tensor(s).float())

        # if torch.is_tensor(a):
        #     a = s.float()
        # else:
        #     a = CUDA(torch.tensor(a).float())

        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())

        inputs = torch.cat((s, a), axis=1)

        with torch.no_grad():
            self.model.eval()
            # oup = self.model(inputs)
            mean, std = self.model(inputs)
            oup = mean + torch.rand(std.shape).to(std.device) * std if training else mean

            if self.label == 'state':
                oup = torch.clamp(oup, -1., 1.)
            elif self.label == 'reward' and self.args.env == 'Platform-v0':
                oup = torch.clamp(oup, 0., 1.)

        return CPU(oup).numpy()

    def fit(self, dataset=None, logger=False, label='', action_rep=None):
        if self.mode:  # hard code
            return np.zeros(self.n_epochs), np.ones(self.n_epochs)
        else:
            # predataset = dm_dataset(dataset, self.label, self.args, action_rep)
            # loader = DataLoader(predataset, batch_size=self.batch_size, shuffle=True)

            self.model.train()

            losses = []
            accuracy = []
            ifr = True if ((self.label=='reward')and(self.args.sparse)) else False

            for epoch in range(self.n_epochs):
                # loss = []
                # acc = []
                # for x, y in loader:
                x, y = self.loader(dataset.sample(reward=ifr))

                self.optimizer.zero_grad()
                p, std = self.model(x)

                if self.args.oup_param == 'Gaussian':
                    scaler = 1. / (math.sqrt(2.) * std)
                    l = self.criterion(p*scaler, y*scaler) + torch.mean(torch.log(std))
                else:
                    l = self.criterion(p, y)

                # loss.append(l.item())

                l.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                self.optimizer.step()

                if self.args.oup_param == 'Gaussian':
                    a = (p.argmax(-1) == y.argmax(-1)).sum().item() if self.label == 'continuous' else 0
                else:
                    a = (p.argmax(-1) == y).sum().item() if self.label == 'continuous' else 0
                
                # acc.append(a / dataset.bs)
                
                # losses.append(np.array(loss).mean())
                # accuracy.append(np.array(acc))

                losses.append(l.item())
                accuracy.append(a / dataset.bs)

            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)

            return np.array(losses), np.array(accuracy)

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def loader(self, batch):
        s, a_type, a_para, ns, r, t = batch
        # if self.args.dm_onepa:
        #     # print(a_para[:5, :])
        #     mask = torch.zeros([len(a_para), self.args.action_parameter_sizes.max()])
        #     print(s.shape, a_type.shape, a_para.shape, ns.shape, r.shape, t.shape, mask.shape)
        #     for i in range(len(a_para)):
        #         offset = self.args.action_parameter_sizes[:a_type[i].argmax()].sum()
        #         mask[i] = a_para[i][offset:offset+self.args.action_parameter_sizes.max()]
        #     a_para = mask
        x = CUDA(torch.cat((s, a_type, a_para), axis=1))

        if self.label == 'state':
            y = ns
        elif self.label == 'reward':
            y = r
        elif self.label == 'continuous':
            if self.args.oup_param == 'Gaussian':
                y = torch.cat(((1-t), t), axis=1)
            else:
                y = t.squeeze(1).type(torch.LongTensor) # 0: continuous, 1: ternimal

        return CUDA(x), CUDA(y)


class dm_dataset(Dataset):
    def __init__(self, data, label, args, action_rep=None):
        s, a_type, a_para, ns, r, t = data
        if args.dm_onepa:
            # print(a_para[:5, :])
            mask = torch.zeros([len(a_para), args.action_parameter_sizes.max()])
            for i in range(len(a_para)):
                offset = args.action_parameter_sizes[:a_type[i].argmax()].sum()
                mask[i] = a_para[i][offset:offset+args.action_parameter_sizes.max()]
            a_para = mask
            # print(a_para[:5, :])
            # exit()
            # a_para = torch.gather(a_para, 1, a_type.argmax(-1).unsqueeze(-1))
        self.x = CUDA(torch.cat((s, a_type, a_para), axis=1))

        if label == 'state':
            self.y = ns
        elif label == 'reward':
            self.y = r
        elif label == 'continuous':
            if args.oup_param == 'Gaussian':
                self.y = torch.cat(((1-t), t), axis=1)
            else:
                self.y = t.squeeze(1).type(torch.LongTensor) # 0: continuous, 1: ternimal

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return CUDA(self.x[idx]), CUDA(self.y[idx])



