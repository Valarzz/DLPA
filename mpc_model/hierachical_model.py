import numpy as np
import torch
import torch.nn as nn
from mpc_model.utils import Model, MLPRegression, MLP
from torch.utils.data import Dataset, DataLoader
import wandb


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def CPU(var):
    return var.cpu().detach()


class hmodel(Model):
    def __init__(self, args, label):
        super().__init__()

        self.state_dim = args.state_dim
        # self.action_dim = args.discrete_action_dim + args.parameter_action_dim
        self.k_dim = args.discrete_action_dim
        self.z_dim = args.parameter_action_dim
        self.label = label

        self.onepa = args.dm_onepa
        # if self.onepa:
        #     self.input_dim = self.state_dim + args.discrete_action_dim + 1
        # else:
        #     self.input_dim = self.state_dim + self.action_dim  # state + action_type + all_action_para
        self.input_dim = self.state_dim + 1

        if label == 'state':
            self.mode = args.dm_hard
            self.n_epochs = args.dm_epoch  # supervised training epochs
            self.lr = args.dm_lr
            self.batch_size = args.dm_batchsize

            self.save_model_flag = args.dm_saveflag
            self.save_model_path = args.dm_savepath

            self.validation_flag = args.dm_valflag
            self.validate_freq = args.dm_valfreq
            self.validation_ratio = args.dm_valrati
            # self.dm_continue = args.dm_continue

            self.load = args.dm_loadmodel
            self.path = args.dm_loadpath
            # self.model = CUDA(MLPRegression(self.input_dim, self.state_dim, args.dm_layers))
            self.model = [CUDA(MLP(self.input_dim, self.state_dim, args.dm_layers, self.label)) for _ in range(args.discrete_action_dim)]

            # self.criterion = nn.MSELoss(reduction='mean')
            self.criterion = nn.HuberLoss(reduction='mean')

        elif label == 'reward':
            self.mode = args.r_hard
            self.n_epochs = args.r_epoch  # supervised training epochs
            self.lr = args.r_lr
            self.batch_size = args.r_batchsize

            self.save_model_flag = args.r_saveflag
            self.save_model_path = args.r_savepath

            self.validation_flag = args.r_valflag
            self.validate_freq = args.r_valfreq
            self.validation_ratio = args.r_valrati

            self.load = args.r_loadmodel
            self.path = args.r_loadpath
            # self.model = CUDA(MLPRegression(self.input_dim, 1, args.r_layers))
            self.model = [CUDA(MLP(self.input_dim, 1, args.r_layers, self.label)) for _ in range(args.discrete_action_dim)]

            # self.criterion = nn.MSELoss(reduction='mean')
            self.criterion = nn.HuberLoss(reduction='mean')

        elif label == 'continuous':
            self.mode = args.c_hard
            self.n_epochs = args.c_epoch  # supervised training epochs
            self.lr = args.c_lr
            self.batch_size = args.c_batchsize

            self.save_model_flag = args.c_saveflag
            self.save_model_path = args.c_savepath

            self.validation_flag = args.c_valflag
            self.validate_freq = args.c_valfreq
            self.validation_ratio = args.c_valrati

            self.load = args.c_loadmodel
            self.path = args.c_loadpath
            # self.model = CUDA(MLPRegression(self.input_dim, 2, args.c_layers))
            self.model = [CUDA(MLP(self.input_dim, 2, args.c_layers, self.label)) for _ in range(args.discrete_action_dim)]

            self.criterion = nn.CrossEntropyLoss()

        if self.load:
            self.load_model(self.path)

        self.reset_model()
        self.optimizer = [torch.optim.Adam(self.model[i].parameters(), lr=self.lr) for i in range(args.discrete_action_dim)]

        # self.mode = "dl"

    def predict(self, s, a):
        # convert to torch format
        k = a[:, :self.k_dim].argmax(-1)
        z = a[:, -1]

        with torch.no_grad():
            s = CUDA(torch.tensor(s).float())
            z = CUDA(torch.tensor(z).unsqueeze(1).float())
            inputs = torch.cat((s, z), axis=1)

            oups = []
            for i in range(self.k_dim):
                self.model[i].eval()
                oup = self.model[i](inputs)
                oups.append(CPU(oup).numpy())

            oup = np.zeros(oup.shape)
            for i in range(self.k_dim):
                ind = np.where(k==i)[0]
                oup[ind] = oups[i][ind]
            
            return oup

        # with torch.no_grad():
        #     self.model.eval()
        #     oup = self.model(inputs)
        #     oup = CPU(oup).numpy()

        # return oup

    def fit(self, dataset=None, logger=False, label=''):
        if self.mode:  # hard code
            return np.zeros(self.n_epochs)
        else:

            predataset = [dm_dataset(dataset[i], self.label, self.onepa) for i in range(self.k_dim)]
            
            loader = [DataLoader(predataset[i], batch_size=self.batch_size, shuffle=True) for i in range(self.k_dim)]

            all_l = {}
            all_a = {}
            for j in range(self.k_dim):
                self.model[j].train()
                losses = []
                accuracy = []
                for epoch in range(self.n_epochs):
                    loss = []
                    acc = []
                    for x, y in loader[j]:
                        self.optimizer[j].zero_grad()
                        p = self.model[j](x)
                        
                        l = self.criterion(p, y)
                        loss.append(l.item())

                        l.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                        self.optimizer[j].step()
                        a = (p.argmax(-1) == y).sum().item() if self.label == 'continuous' else 0
                        acc.append(a)
                losses.append(np.array(loss).mean())
                accuracy.append(np.array(acc).sum()/len(predataset[j]))
                
                all_l[str(j)] = np.array(losses)
                all_a[str(j)] = np.array(accuracy)

            # if logger:
            #     wandb.log({label: np.array(loss)})

            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)

            return all_l, all_a
            # return np.array(losses), np.array(accuracy)

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        for i in range(len(self.model)):
            self.model[i].apply(weight_reset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class dm_dataset(Dataset):
    def __init__(self, data, label, onepa):
        s, a_type, a_para, ns, r, t = data
        if onepa:
            a_para = torch.gather(a_para, 1, a_type.argmax(-1).unsqueeze(-1))
        self.x = torch.cat((s, a_para), axis=1)

        if label == 'state':
            self.y = ns
        elif label == 'reward':
            self.y = r
        elif label == 'continuous':
            # self.y = torch.cat((c, t), axis=1)
            self.y = CUDA(t.squeeze(1).type(torch.LongTensor)) # 0: continuous, 1: ternimal

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class hreplayBuffer(object):
    def __init__(self, state_dim,
                 discrete_action_dim, all_parameter_action_dim,
                 max_size=int(1e6)):
        self.replay_buffers = []
        self.k_dim = discrete_action_dim
        for _ in range(self.k_dim):
            self.replay_buffers.append(ReplayBuffer(state_dim=state_dim,
                                 discrete_action_dim=discrete_action_dim,
                                 all_parameter_action_dim=all_parameter_action_dim,
                                 max_size=int(1e5)))
            
    def add(self,
            state,
            discrete_action, all_parameter_action,
            next_state, reward, terminal):
        self.replay_buffers[discrete_action.argmax()].add(state,
                                                        discrete_action=discrete_action,
                                                        all_parameter_action=all_parameter_action,
                                                        next_state=next_state,
                                                        reward=reward, 
                                                        terminal=terminal)
        
    def sample(self, batch_size):
        samples = [self.replay_buffers[i].sample(batch_size) for i in range(self.k_dim)]
        return samples
            

class ReplayBuffer(object):
    def __init__(self, state_dim,
                 discrete_action_dim, all_parameter_action_dim,
                 max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim))

        self.next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.terminal = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self,
            state,
            discrete_action, all_parameter_action,
            next_state, reward, terminal):
        self.state[self.ptr] = state

        self.discrete_action[self.ptr] = discrete_action
        self.all_parameter_action[self.ptr] = np.array(all_parameter_action).reshape(3)

        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.terminal[self.ptr] = terminal  # 1:  terminal

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if batch_size < self.size:
            ind = np.random.choice(self.size, size=batch_size, replace=False)
        else:
            ind = np.random.choice(self.size, size=self.size, replace=False)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),

            torch.FloatTensor(self.discrete_action[ind]).to(self.device),
            torch.FloatTensor(self.all_parameter_action[ind]).to(self.device),

            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),

            torch.FloatTensor(self.terminal[ind]).to(self.device),   # terminal
        )

    def save(self, name):
        np.save(f"{name}_state.npy", self.state[:self.ptr])
        np.save(f"{name}_discrete_action.npy", self.discrete_action[:self.ptr])
        np.save(f"{name}_all_parameter_action.npy", self.all_parameter_action[:self.ptr])
        np.save(f"{name}_next_state.npy", self.next_state[:self.ptr])
        np.save(f"{name}_reward.npy", self.reward[:self.ptr])
        np.save(f"{name}_not_done.npy", self.not_done[:self.ptr])


