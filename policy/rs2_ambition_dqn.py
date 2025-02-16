import sys
import numpy as np
import torch

from policy.base_policy import BasePolicy


class RS2AmbitionDQN(BasePolicy):
    def __init__(self, model_class, **kwargs):
        super().__init__(model_class, **kwargs)
        self.epsilon_dash = kwargs['epsilon_dash']
        self.k = kwargs['k']
        self.centroids_decay = kwargs['centroids_decay']
        self.centroids = None
        self.pseudo_counts = None
        self.weights = None
        self.ras = None
        self.pi = None

    def initialize(self):
        super().initialize()
        self.centroids = np.random.randn(self.action_space * self.k, self.hidden_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.pseudo_counts = np.zeros(self.action_space * self.k)
        self.weights = np.zeros(self.action_space * self.k)
        self.ras = np.zeros(self.action_space)

    def action(self, state):
        self.total_steps += 1
        if self.total_steps < self.warmup:
            action = np.random.choice(self.action_space)
        else:
            q_value = self.calc_q_value(state)
            aleph = q_value.max() + self.epsilon_dash
            diff = aleph - q_value
            z = 1.0 / np.sum(1.0 / diff)
            rho = z / diff
            b = self.ras / rho - 1.0 + sys.float_info.epsilon
            rsrs = (1.0 + max(b)) * rho - self.ras

            if min(rsrs) < 0:
                rsrs -= min(rsrs)

            self.pi = rsrs / np.sum(rsrs)
            action = np.random.choice(self.action_space, p=self.pi)
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return
        
        controllable_state = self.calc_embed(state)
        self.calc_reliability(controllable_state, action)

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

    def calc_reliability(self, controllable_state, action):
        self.pseudo_counts *= self.centroids_decay
        self.weights *= self.centroids_decay

        controllable_state_norm = controllable_state / (np.linalg.norm(controllable_state) + sys.float_info.epsilon)

        distances = np.linalg.norm(self.centroids - controllable_state_norm, axis=1)
        weight = 1 / (distances + sys.float_info.epsilon)

        denom = (self.weights[:, None] + weight[:, None] + sys.float_info.epsilon)
        self.centroids = (self.weights[:, None] * self.centroids + weight[:, None] * controllable_state_norm) / denom

        self.weights += weight
        self.pseudo_counts[action] += 1

        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)

        reliability_scores = self.weights / (self.pseudo_counts + sys.float_info.epsilon)
        action_reliability_scores = reliability_scores.reshape(self.action_space, self.k).mean(axis=1)
        action_reliability_scores_norm = action_reliability_scores / np.linalg.norm(action_reliability_scores)
        exp_scores = np.exp(action_reliability_scores_norm - np.max(action_reliability_scores_norm))
        action_reliability_softmax_scores = exp_scores / np.sum(exp_scores)
        self.ras = action_reliability_softmax_scores
