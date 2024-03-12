import torch
import torch.nn as nn
import wandb


def platform_reward(x):
    x = nn.Tanh()(x)
    return (x+1)/2  # fomr [-1, 1] to [0, 1]


class MLP(nn.Module):
    def __init__(self, inp_dim, oup_dim, layers, label, env, oup_param='deter'):
        super().__init__()
        self.label = label
        self.oup_param = oup_param
        self.env = env

        if self.oup_param == 'Gaussian':
            oup_dim *= 2
            self.mean = nn.Linear(oup_dim, oup_dim/2)
            self.std = nn.Sequential(nn.Linear(oup_dim, oup_dim/2),
                                     nn.ReLU())

        if label == 'state':
            self.last_act = nn.Tanh()

        elif label == 'continuous':
            self.last_act = nn.Identity()

        elif label == 'reward':
            if env == 'Platform-v0':
                self.last_act = platform_reward
            elif env == 'Goal-v0' or env == 'simple_catch':
                self.last_act = nn.Identity()

        # self.last_act = nn.Identity() if label=='continuous' else nn.Tanh()
        self.layers = nn.ModuleList([])
        for layer in layers:
            self.layers.append(nn.Sequential(nn.Linear(inp_dim, layer),
                                             nn.Dropout(p=0.1),
                                             nn.ReLU()))
            inp_dim = layer
        self.layers.append(nn.Sequential(nn.Linear(inp_dim, oup_dim),
                                         nn.Dropout(p=0.1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.oup_param == 'Gaussian':
            mean, std = self.last_act(self.mean(x)), self.std(x)
            return mean, std

        else:
            x = self.last_act(x)
            return x, torch.zeros(x.shape).to(x.device)


def mlp(sizes, activation, output_activation=nn.Tanh):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPRegression(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        """
            @param int - input_dim
            @param int - output_dim
            @param list - hidden_sizes : such as [32,32,32]
        """
        super().__init__()
        self.net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]

            @return tensor - out : shape [batch, output dim]
        """
        out = self.net(x)
        return out


class MLPCategorical(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        """
            @param int - input_dim
            @param int - output_dim
            @param list - hidden_sizes : such as [32,32,32]
        """
        super().__init__()
        self.logits_net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]

            @return tensor - out : shape [batch, 1]
        """
        logits = self.logits_net(x)
        out = Categorical(logits=logits)
        return torch.squeeze(out, -1)


class Model:

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, action):
        """
        Predict a batch of state and action pairs and return numpy array

        Parameters:
        ----------
            @param tensor or numpy - state : size should be (batch_size x state dim)
            @param tensor or numpy - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - state_next - size should be (batch_size x state dim)
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def fit(self, data, label):
        """
        Fit the model given data and label

        Parameters:
        ----------
            @param list of numpy array - data : each array size should be of (state dim + action dim)
            @param list of numpy array - label : each array size should be of (state dim)

        Return:
        ----------
            @param (int, int) - (training loss, test loss)
        """
        raise NotImplementedError("Must be implemented in subclass.")


def train(models, data, logger=False, model_type='normal', k_dim=3, action_rep=None):
    losses = {}
    acc = {}
    for key, model in models:
        loss, a = model.fit(data, action_rep=action_rep)
        losses[key] = loss
        if key == 'terminal':
            acc[key] = a

    if logger:
        if model_type == 'normal':
            keys = losses.keys()
            i = 0
            stop = 0
            while True:
                for key in keys:
                    if i < len(losses[key]):
                        wandb.log({key: losses[key][i]})
                        if key == 'terminal':
                            wandb.log({key + '_acc': acc[key][i]})
                    else:
                        stop += 1
                i += 1
                if stop == len(keys):
                    break

        elif model_type == 'h':
            keys = losses.keys()
            i = 0
            stop = 0
            while True:
                for key in keys:
                    if i < len(losses[key]['0']):
                        for j in range(k_dim):
                            wandb.log({key + f'_{j}': losses[key][str(j)][i]})
                        if key == 'terminal':
                            for j in range(k_dim):
                                wandb.log({f'accuracy_{j}': acc[key][str(j)][i]})
                    else:
                        stop += 1
                i += 1
                if stop == len(keys):
                    break


