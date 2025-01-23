import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory.replay_buffer import ReplayBuffer
from network.rsrsdqnnet import RSRSDQNNet


class RSRSAlephQEpsRASChoiceCentroidDQN:
    def __init__(self, model=RSRSDQNNet, **kwargs):
        self.gamma = kwargs['gamma']
        self.epsilon_dash = kwargs['epsilon_dash']
        self.k = kwargs['k']
        self.zeta = kwargs['zeta']
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
        self.model_class = model
        self.model = None
        self.model_target = None
        self.centroids = None
        self.pseudo_counts = None
        self.weights = None
        self.ras = None
        self.total_steps = None
        self.loss = None

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.centroids = np.random.randn(self.action_space * self.k, self.hidden_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.pseudo_counts = np.zeros(self.action_space * self.k)
        self.weights = np.zeros(self.action_space * self.k)
        self.ras = np.zeros(self.action_space)
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

    def calc_q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def embed(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.embedding(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        self.total_steps += 1
        if self.total_steps < self.warmup:
            action = np.random.choice(self.action_space)
        else:
            q_value = self.calc_q_value(state)
            aleph = q_value.max() + np.float64(1e-10)
            diff = aleph - q_value
            z = 1.0 / np.sum(1.0 / diff)
            rho = z / diff
            b = self.ras / rho - 1.0 + np.float64(1e-10)
            rsrs = (1.0 + max(b)) * rho - self.ras

            if min(rsrs) < 0:
                rsrs -= min(rsrs)

            pi = rsrs / np.sum(rsrs)
            action = np.random.choice(self.action_space, p=pi)
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return
        
        controllable_state = self.embed(state)
        self.calculate_reliability(controllable_state, action)

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float64).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float64).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float64).to(self.device)
        d = torch.tensor(d, dtype=torch.float64).to(self.device)

        q = self.model(s)
        qa = q[np.arange(self.batch_size), a]
        with torch.no_grad():
            next_q_target = self.model_target(ns)
            next_qa_target = torch.amax(next_q_target, dim=1)
        
        target = r + self.gamma * next_qa_target * (1 - d)
        self.loss = self.criterion(qa, target)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        if self.sync_model_update == 'hard':
            if self.total_steps % self.target_update_freq == 0:
                self.sync_model_hard()
        elif self.sync_model_update == 'soft':
            self.sync_model_soft()
        else:
            raise ValueError(f'Invalid sync_model_update: {self.sync_model_update}')

    def calculate_reliability(self, controllable_state, action):
        self.pseudo_counts *= self.gamma
        self.weights *= self.gamma

        controllable_state_norm = controllable_state / (np.linalg.norm(controllable_state) + self.epsilon_dash)

        distances = np.linalg.norm(self.centroids - controllable_state_norm, axis=1)
        weight = 1 / (distances + self.epsilon_dash)

        denom = (self.weights[:, None] + weight[:, None] + self.epsilon_dash)
        self.centroids = (self.weights[:, None] * self.centroids + weight[:, None] * controllable_state_norm) / denom

        self.weights += weight
        self.pseudo_counts[action] += 1

        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)

        reliability_scores = self.weights / (self.pseudo_counts + self.epsilon_dash)
        action_reliability_scores = reliability_scores.reshape(self.action_space, self.k).mean(axis=1)
        action_reliability_scores_norm = action_reliability_scores / np.linalg.norm(action_reliability_scores)
        exp_scores = np.exp(action_reliability_scores_norm - np.max(action_reliability_scores_norm))
        action_reliability_softmax_scores = exp_scores / np.sum(exp_scores)
        self.ras = action_reliability_softmax_scores

    def sync_model_hard(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def sync_model_soft(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
