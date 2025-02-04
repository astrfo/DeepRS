import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

from memory.replay_buffer import ReplayBuffer

torch.set_default_dtype(torch.float64)


class BasePolicy(ABC):
    def __init__(self, model_class, **kwargs):
        self.gamma = kwargs['gamma']
        self.optimizer_name = kwargs['optimizer']
        self.adam_learning_rate = kwargs['adam_learning_rate']
        self.rmsprop_learning_rate = kwargs['rmsprop_learning_rate']
        self.rmsprop_alpha = kwargs['rmsprop_alpha']
        self.rmsprop_eps = kwargs['rmsprop_eps']
        self.criterion_name = kwargs['criterion']
        self.mseloss_reduction = kwargs['mseloss_reduction']
        self.replay_buffer_capacity = kwargs['replay_buffer_capacity']
        self.hidden_size = kwargs['hidden_size']
        self.sync_model_update = kwargs['sync_model_update']
        self.warmup = kwargs['warmup']
        self.tau = kwargs['tau']
        self.batch_size = kwargs['batch_size']
        self.target_update_freq = kwargs['target_update_freq']
        self.state_space = kwargs['state_space']
        self.action_space = kwargs['action_space']
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity, self.batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model_class
        self.model = None
        self.model_target = None
        self.total_steps = None
        self.loss = None
        self.pi = None

    def initialize(self):
        self.replay_buffer.initialize()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.total_steps = 0

        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.rmsprop_learning_rate, alpha=self.rmsprop_alpha, eps=self.rmsprop_eps)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer_name}')

        if self.criterion_name == 'mseloss':
            self.criterion = nn.MSELoss(reduction=self.mseloss_reduction)
        elif self.criterion_name == 'smoothl1loss':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f'Invalid criterion: {self.criterion_name}')

    def reset(self):
        pass

    def greedy_action(self, state):
        q_value = self.calc_q_value(state)
        return np.random.choice(np.where(q_value == max(q_value))[0])

    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass

    def calc_q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def calc_embed(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.embedding(s).squeeze().to('cpu').detach().numpy().copy()

    def sync_model_hard(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def sync_model_soft(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
