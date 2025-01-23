import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss

from memory.replay_buffer import ReplayBuffer
from memory.episodic_memory import EpisodicMemory
from network.rsrsnet import RSRSNet

torch.set_default_dtype(torch.float64)


class RSRSAlephQEpsRASChoiceDQN:
    def __init__(self, model=RSRSNet, **kwargs):
        self.warmup = kwargs.get('warmup', 10)
        self.k = kwargs.get('k', 5)
        self.zeta = kwargs.get('zeta', 0.008)
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.tau = kwargs.get('tau', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        self.centroids = np.random.randn(self.action_space * self.k, self.hidden_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.pseudo_counts = np.zeros(self.action_space * self.k)
        self.weights = np.zeros(self.action_space * self.k)
        self.ras = np.zeros(self.action_space)
        self.total_steps = 0

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.centroids = np.random.randn(self.action_space * self.k, self.hidden_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.pseudo_counts = np.zeros(self.action_space * self.k)
        self.weights = np.zeros(self.action_space * self.k)
        self.ras = np.zeros(self.action_space)
        self.total_steps = 0

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def embed(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.embedding(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        self.total_steps += 1
        if len(self.replay_buffer.memory) < self.warmup:
            controllable_state = self.embed(state)
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_value(state)
            aleph = max(q_values) + 0.001
            controllable_state = self.embed(state)
            diff = aleph - q_values
            Z = np.float64(1.0) / np.sum(np.float64(1.0) / diff)
            rho = Z / diff
            b = self.ras / rho - np.float64(1.0) + 0.0001
            SRS = (np.float64(1.0) + max(b)) * rho - self.ras
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)

            action = np.random.choice(self.action_space, p=pi)
            self.calculate_reliability(controllable_state, action)
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float64).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float64).to(self.device)
        r = torch.tensor(r, dtype=torch.float64).to(self.device)
        d = torch.tensor(d, dtype=torch.float64).to(self.device)

        q = self.model(s)
        qa = q[np.arange(self.batch_size), a]
        next_q_target = self.model_target(ns)
        next_qa_target = torch.amax(next_q_target, dim=1)
        target = r + self.gamma * next_qa_target * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(qa, target)
        loss.backward()
        self.optimizer.step()
        if self.total_steps % 500 == 0:
            self.sync_model_hard()

    def calculate_reliability(self, controllable_state, action):
        self.pseudo_counts *= 0.99
        self.weights *= 0.99

        controllable_state_norm = controllable_state / (np.linalg.norm(controllable_state) + 0.0001)

        distances = np.linalg.norm(self.centroids - controllable_state_norm, axis=1)
        weight = 1 / (distances + 0.0001)

        denom = (self.weights[:, None] + weight[:, None] + 0.0001)
        self.centroids = (self.weights[:, None] * self.centroids + weight[:, None] * controllable_state_norm) / denom

        self.weights += weight
        self.pseudo_counts[action] += 1

        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)

        reliability_scores = self.weights / (self.pseudo_counts + 0.0001)
        action_reliability_scores = reliability_scores.reshape(self.action_space, self.k).mean(axis=1)
        action_reliability_scores_norm = action_reliability_scores / np.linalg.norm(action_reliability_scores)
        exp_scores = np.exp(action_reliability_scores_norm - np.max(action_reliability_scores_norm))
        action_reliability_softmax_scores = exp_scores / np.sum(exp_scores)
        self.ras = action_reliability_softmax_scores

    def sync_model(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)

    def sync_model_hard(self):
        self.model_target.load_state_dict(self.model.state_dict())
