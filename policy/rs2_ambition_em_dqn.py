import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss

from policy.base_policy import BasePolicy
from memory.episodic_memory import EpisodicMemory


class RS2AmbitionEMDQN(BasePolicy):
    def __init__(self, model_class, **kwargs):
        super().__init__(model_class, **kwargs)
        self.epsilon_dash = kwargs['epsilon_dash']
        self.k = kwargs['k']
        self.episodic_memory_capacity = kwargs['episodic_memory_capacity']
        self.embedding_size = kwargs['embedding_size']
        self.episodic_memory = EpisodicMemory(self.episodic_memory_capacity, self.batch_size, self.action_space)

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space)
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

    def action(self, state):
        self.total_steps += 1
        if self.total_steps < self.warmup:
            controllable_state = self.calc_embed(state)
            action = np.random.choice(self.action_space)
            self.episodic_memory.add(controllable_state, action)
        else:
            q_value = self.calc_q_value(state)
            controllable_state = self.calc_embed(state)
            ras = self.calc_reliability(controllable_state)
            aleph = q_value.max() + np.float64(1e-10)
            diff = aleph - q_value
            z = 1.0 / np.sum(1.0 / diff)
            rho = z / diff
            b = ras / rho - 1.0 + np.float64(1e-10)
            rsrs = (1.0 + max(b)) * rho - ras

            if min(rsrs) < 0:
                rsrs -= min(rsrs)

            self.pi = rsrs / np.sum(rsrs)
            action = np.random.choice(self.action_space, p=self.pi)
            self.episodic_memory.add(controllable_state, action)
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return

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

    def calc_reliability(self, controllable_state):
        controllable_state_and_action = np.array(self.episodic_memory.memory, dtype=np.float64)
        controllable_state_vec = controllable_state_and_action[:, :len(controllable_state)]
        action_vec = controllable_state_and_action[:, len(controllable_state):]
        controllable_state = np.expand_dims(controllable_state, axis=0)
        
        index = faiss.IndexFlatL2(controllable_state_vec.shape[1])
        index.add(controllable_state_vec)
        D, I = index.search(controllable_state, self.k)

        squared_distance = D.flatten() ** 2
        average_squared_distance = np.average(squared_distance)
        regularization_squared_distance = squared_distance / average_squared_distance
        regularization_squared_distance = np.maximum(regularization_squared_distance, 0)

        inverse_kernel_function = self.epsilon_dash / (regularization_squared_distance + self.epsilon_dash)
        sum_kernel = np.sum(inverse_kernel_function)
        weight = inverse_kernel_function / sum_kernel
        ras = np.average(action_vec[I.flatten()], weights=weight, axis=0)
        return ras
