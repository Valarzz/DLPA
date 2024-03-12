import numpy as np
import torch
import torch.nn as nn
from utils import Model, MLPRegression


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def CPU(var):
    return var.cpu().detach()


class DynamicModel(Model):
    def __init__(self, args):
        super().__init__()

        self.state_dim = args.state_dim
        self.action_dim = args.discrete_action_dim + args.parameter_action_dim

        self.onepa = args.dm_onepa
        if self.onepa:
            self.input_dim = self.state_dim + args.discrete_action_dim + 1
        else:
            self.input_dim = self.state_dim + self.action_dim  # state + action_type + all_action_para

        self.n_epochs = args.dm_epoch  # supervised training epochs
        self.lr = args.dm_lr
        self.batch_size = args.dm_batchsize

        self.save_model_flag = args.dm_saveflag
        self.save_model_path = args.dm_savepath

        self.validation_flag = args.dm_valflag
        self.validate_freq = args.dm_valfreq
        self.validation_ratio = args.dm_valrati
        self.dm_continue = args.dm_continue

        if args.dm_loadmodel:
            self.model = CUDA(torch.load(args.dm_savepath))
        else:
            if self.dm_continue:
                self.model = CUDA(MLPRegression(self.input_dim, self.state_dim + 2, args.dm_layers))  # next_s + continuous
            else:
                self.model = CUDA(MLPRegression(self.input_dim, self.state_dim, args.dm_layers))  # next_s + continuous


        self.reset_model()
        self.state_criterion = nn.MSELoss(reduction='mean')
        self.conti_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.mode = "dl"

    def process_dataset(self, dataset, train=True):
        ratio = int(len(dataset[0]) * (1 - self.validation_ratio))
        s, a_type, a_para, ns, r, c, t = (e[:ratio] for e in dataset) if train else (e[ratio:] for e in dataset)
        inds = list(range(len(s)))
        while len(inds) > 0:
            binds = inds[:self.batch_size]
            inds = inds[self.batch_size:]
            bs, ba_type, ba_para, ba_ns, bc, bt = s[binds], a_type[binds], a_para[binds], ns[binds], c[binds], t[binds]
            if self.onepa:
                ba_para = torch.gather(ba_para, 1, ba_type.argmax(-1).unsqueeze(-1))
            x = torch.cat((bs, ba_type, ba_para), axis=1)
            bct = torch.cat((bc, bt), axis=1)
            yield x, ba_ns, bct

    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        with torch.no_grad():
            next_state = self.model(inputs)
            next_state = CPU(next_state).numpy()
        return next_state

    def fit(self, dataset=None, logger=True):
        for epoch in range(self.n_epochs):
            train_loader = self.process_dataset(dataset, train=True)
            self.model.train()
            loss_this_epoch = []
            loss_state = []
            loss_ct = []
            for data, label_s, label_ct in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)

                if self.dm_continue:
                    s, ct = outputs[:, :self.state_dim], outputs[:, self.state_dim:]
                    ls = self.state_criterion(s, label_s)
                    lc = self.conti_criterion(ct, label_ct)
                    loss = ls + lc
                    loss_state.append(ls.item())
                    loss_ct.append(lc.item())
                else:
                    loss = self.state_criterion(outputs, label_s)

                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())

            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)

            if self.validation_flag and (epoch + 1) % self.validate_freq == 0:
                train_loader = self.process_dataset(dataset, train=True)
                test_loader = self.process_dataset(dataset, train=False)
                loss_test = 11111111
                if test_loader is not None:
                    loss_test = self.validate_model(test_loader)
                loss_train = self.validate_model(train_loader)
                if logger:
                    print(
                        f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}, loss test  {loss_test:.4f}")
        # print(outputs[0, :], '\n', label_s[0, :])
        if self.dm_continue:
            return np.mean(loss_this_epoch), np.mean(loss_state), np.mean(loss_ct)
        else:
            return np.mean(loss_this_epoch), None, None

    def validate_model(self, testloader):
        with torch.no_grad():
            self.model.eval()
            loss_list = []
            for data, label_s, label_ct in testloader:
                outputs = self.model(data)
                if self.dm_continue:
                    s, ct = outputs[:, :self.state_dim], outputs[:, self.state_dim:]
                    ls = self.state_criterion(s, label_s)
                    lc = self.conti_criterion(ct, label_ct)
                    loss = ls + lc
                else:
                    loss = self.state_criterion(outputs, label_s)
                loss_list.append(loss.item())
            return np.mean(loss_list)

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class RewardModel(Model):
    def __init__(self, args):
        super().__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.discrete_action_dim + args.parameter_action_dim

        self.onepa = args.dm_onepa
        if self.onepa:
            self.input_dim = self.state_dim + args.discrete_action_dim + 1
        else:
            self.input_dim = self.state_dim + self.action_dim  # state + action_type + all_action_para

        self.n_epochs = args.r_epoch  # supervised training epochs
        self.lr = args.r_lr
        self.batch_size = args.r_batchsize

        self.save_model_flag = args.r_saveflag
        self.save_model_path = args.r_savepath

        self.validation_flag = args.r_valflag
        self.validate_freq = args.r_valfreq
        self.validation_ratio = args.r_valrati

        if args.r_loadmodel:
            self.model = CUDA(torch.load(args.r_savepath))
        else:
            self.model = CUDA(MLPRegression(self.input_dim, 1, args.r_layers))  # next_s + continuous

        self.reset_model()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.mode = "dl"

    def process_dataset(self, dataset, train=True):
        ratio = int(len(dataset[0]) * (1 - self.validation_ratio))
        s, a_type, a_para, ns, r, c, t = (e[:ratio] for e in dataset) if train else (e[ratio:] for e in dataset)
        inds = list(range(len(s)))
        while len(inds) > 0:
            binds = inds[:self.batch_size]
            inds = inds[self.batch_size:]
            bs, ba_type, ba_para, br = s[binds], a_type[binds], a_para[binds], r[binds]
            if self.onepa:
                ba_para = torch.gather(ba_para, 1, ba_type.argmax(-1).unsqueeze(-1))
            x = torch.cat((bs, ba_type, ba_para), axis=1)
            y = br
            yield x, y

    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        with torch.no_grad():
            r = self.model(inputs)
            r = (r+1)/2
            r = CPU(r).numpy()
        return r

    def fit(self, dataset=None, logger=True):
        for epoch in range(self.n_epochs):
            train_loader = self.process_dataset(dataset, train=True)
            self.model.train()
            loss_this_epoch = []
            for data, label in train_loader:
                self.optimizer.zero_grad()
                r = self.model(data)
                r = (r + 1) / 2
                loss = self.criterion(r, label)
                # print(loss)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())

            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)

            if self.validation_flag and (epoch + 1) % self.validate_freq == 0:
                train_loader = self.process_dataset(dataset, train=True)
                test_loader = self.process_dataset(dataset, train=False)
                loss_test = 11111111
                if test_loader is not None:
                    loss_test = self.validate_model(test_loader)
                loss_train = self.validate_model(train_loader)
                if logger:
                    print(
                        f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}, loss test  {loss_test:.4f}")

        return np.mean(loss_this_epoch)

    def validate_model(self, testloader):
        with torch.no_grad():
            self.model.eval()
            loss_list = []
            for data, label in testloader:
                r = self.model(data)
                r = (r + 1) / 2
                loss = self.criterion(r, label)
                loss_list.append(loss.item())
            return np.mean(loss_list)

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)

        self.model.apply(weight_reset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))



