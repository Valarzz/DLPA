import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim,
                 discrete_action_dim, all_parameter_action_dim,
                 batch_size, max_size=int(1e6)):
        self.max_size = max_size
        self.bs = batch_size

        self.ptr = 0
        self.size = 0
        self.k_dim = discrete_action_dim
        self.z_dim = all_parameter_action_dim

        self.state = np.zeros((max_size, state_dim))
        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim))

        self.next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.terminal = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.create_end_buffer(max_size, state_dim, discrete_action_dim, all_parameter_action_dim)

    def add(self,
            state,
            discrete_action, all_parameter_action,
            next_state, reward, terminal):
        self.state[self.ptr] = state

        self.discrete_action[self.ptr] = discrete_action
        self.all_parameter_action[self.ptr] = np.array(all_parameter_action).reshape(self.z_dim)

        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.terminal[self.ptr] = terminal  # 1:  terminal

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if terminal:
            self.add_end_buffer(state, discrete_action, all_parameter_action, reward)

    def create_end_buffer(self, max_size, state_dim, discrete_action_dim, all_parameter_action_dim):
        self.end_state = np.zeros((max_size, state_dim))
        self.end_discrete_action = np.zeros((max_size, discrete_action_dim))
        self.end_all_parameter_action = np.zeros((max_size, all_parameter_action_dim))
        self.end_reward = np.zeros((max_size, 1))
        self.end_ptr = 0
        self.end_size = 0

    def add_end_buffer(self, state, discrete_action, all_parameter_action, reward):
        self.end_state[self.ptr] = state
        self.end_discrete_action[self.ptr] = discrete_action
        self.end_all_parameter_action[self.ptr] = np.array(all_parameter_action).reshape(self.z_dim)
        self.end_reward[self.ptr] = reward
        self.end_ptr = (self.end_ptr + 1) % self.max_size
        self.end_size = min(self.end_size + 1, self.max_size)

    def sample(self, batch_size=None, reward=False):
        if batch_size == None:
            batch_size = self.bs

        if reward:
            if batch_size < self.end_size:
                ind = np.random.choice(self.end_size, size=batch_size, replace=False)
            else:
                ind = np.random.choice(self.end_size, size=self.end_size, replace=False)

            return (
                torch.FloatTensor(self.end_state[ind]),

                torch.FloatTensor(self.end_discrete_action[ind]),
                torch.FloatTensor(self.end_all_parameter_action[ind]),
                None,
                torch.FloatTensor(self.end_reward[ind]),
                None,
            )

        else:
            if batch_size < self.size:
                ind = np.random.choice(self.size, size=batch_size, replace=False)
            else:
                ind = np.random.choice(self.size, size=self.size, replace=False)

            return (
                torch.FloatTensor(self.state[ind]),

                torch.FloatTensor(self.discrete_action[ind]),
                torch.FloatTensor(self.all_parameter_action[ind]),

                torch.FloatTensor(self.next_state[ind]),
                torch.FloatTensor(self.reward[ind]),

                torch.FloatTensor(self.terminal[ind]),   # terminal
            )

    def save(self, name):
        np.save(f"{name}_state.npy", self.state[:self.ptr])
        np.save(f"{name}_discrete_action.npy", self.discrete_action[:self.ptr])
        np.save(f"{name}_all_parameter_action.npy", self.all_parameter_action[:self.ptr])
        np.save(f"{name}_next_state.npy", self.next_state[:self.ptr])
        np.save(f"{name}_reward.npy", self.reward[:self.ptr])
        np.save(f"{name}_not_done.npy", self.terminal[:self.ptr])

