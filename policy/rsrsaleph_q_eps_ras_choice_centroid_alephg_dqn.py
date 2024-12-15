import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from memory.replay_buffer import ReplayBuffer
from network.rsrsnet import RSRSNet


class RSRSAlephQEpsRASChoiceCentroidAlephGDQN:
    def __init__(self, model=RSRSNet, **kwargs):
        self.gamma = kwargs['gamma']
        self.epsilon_dash = kwargs['epsilon_dash']
        self.k = kwargs['k']
        self.zeta = kwargs['zeta']
        self.global_aleph = kwargs['global_aleph']
        self.global_value_size = kwargs['global_value_size']
        self.global_value_buffer = deque(maxlen=self.global_value_size)
        self.aleph_beta = 1
        self.aleph_state = self.global_aleph
        self.global_value = 0
        self.adam_learning_rate = kwargs['adam_learning_rate']
        self.mseloss_reduction = kwargs['mseloss_reduction']
        self.replay_buffer_capacity = kwargs['replay_buffer_capacity']
        self.episodic_memory_capacity = kwargs['episodic_memory_capacity']
        self.hidden_size = kwargs['hidden_size']
        self.embedding_size = kwargs['embedding_size']
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
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        self.criterion = nn.MSELoss(reduction=self.mseloss_reduction)
        self.centroids = np.random.randn(self.action_space * self.k, self.embedding_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.pseudo_counts = np.zeros(self.action_space * self.k)
        self.weights = np.zeros(self.action_space * self.k)
        self.ras = np.zeros(self.action_space)
        self.total_steps = 0
        self.total_episodic_reward = 0
        self.loss = None

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        self.centroids = np.random.randn(self.action_space * self.k, self.embedding_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.pseudo_counts = np.zeros(self.action_space * self.k)
        self.weights = np.zeros(self.action_space * self.k)
        self.ras = np.zeros(self.action_space)

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def embed(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.embedding(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        self.total_steps += 1
        if self.total_steps < self.warmup:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_value(state)
            diff = self.aleph_state - q_values
            Z = 1.0 / np.sum(1.0 / diff)
            rho = Z / diff
            SRS = ((self.ras / rho).max() + self.epsilon_dash) * rho - self.ras
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)

            action = np.random.choice(self.action_space, p=pi)
        return action

    def update_global_value(self, reward):
        self.global_value_buffer.append(reward)
        self.global_value = np.mean(self.global_value_buffer)

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return
        
        controllable_state = self.embed(state)
        self.calculate_reliability(controllable_state, action)

        self.total_episodic_reward += reward
        self.calculate_aleph_state_beta(state)

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

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

    def calculate_aleph_state_beta(self, state):
        q_values = self.q_value(state)
        self.aleph_beta = (self.global_aleph - self.total_episodic_reward) / self.global_aleph
        self.aleph_beta = np.clip(self.aleph_beta, 0, 1)
        self.aleph_state = self.aleph_beta * self.global_aleph + (1-self.aleph_beta) * max(q_values)

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
                target_param.data.copy_(self.tau*local_param.data + (np.float32(1.0)-self.tau)*target_param.data)
