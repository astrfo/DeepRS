import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory.replay_buffer import ReplayBuffer
from memory.episodic_memory import EpisodicMemory
from network.conv_atari_rsrsnet import ConvRSRSAtariNet


class ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari:
    def __init__(self, model=ConvRSRSAtariNet, **kwargs):
        super().__init__()
        self.warmup = kwargs['warmup']
        self.k = kwargs['k']
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.epsilon_dash = kwargs['epsilon_dash']
        self.target_update_freq = kwargs['target_update_freq']
        self.hidden_size = kwargs['hidden_size']
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.frame_shape = kwargs['frame_shape']
        self.neighbor_frames = kwargs['neighbor_frames']
        self.replay_buffer_capacity = kwargs['replay_buffer_capacity']
        self.batch_size = kwargs['batch_size']
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity, self.batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames).float()
        self.model_target.to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.criterion = nn.SmoothL1Loss()
        self.n = np.zeros(self.action_space)
        self.total_steps = 0
        self.loss = None
        self.centroids = np.random.randn(self.action_space * self.k, self.hidden_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.pseudo_counts = np.zeros(self.action_space * self.k)
        self.weights = np.zeros(self.action_space * self.k)

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames).float()
        self.model_target.to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.n = np.zeros(self.action_space)

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
        if self.warmup > self.total_steps:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_value(state)
            aleph = max(q_values) + self.epsilon_dash
            diff = aleph - q_values
            Z = 1.0 / np.sum(1.0 / diff)
            rho = Z / diff
            SRS = ((self.n / rho).max() + self.epsilon_dash) * rho - self.n
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)

            action = np.random.choice(self.action_space, p=pi)
        return action

    def update(self, state, action, reward, next_state, done):
        reward = max(min(reward, 1), -1)
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return
        
        controllable_state = self.embed(state)
        self.calculate_reliability(controllable_state, action)

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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=40.0)
        self.optimizer.step()
        if self.total_steps % self.target_update_freq == 0:
            self.sync_model()

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
        self.n = action_reliability_softmax_scores

    def sync_model(self):
        self.model_target.load_state_dict(self.model.state_dict())
