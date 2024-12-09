import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss

from memory.replay_buffer import ReplayBuffer
from memory.episodic_memory import EpisodicMemory
from network.rsrsnet import RSRSNet


class RSRSAlephQEpsRASChoiceDQN:
    def __init__(self, model=RSRSNet, **kwargs):
        self.gamma = kwargs['gamma']
        self.epsilon_dash = kwargs['epsilon_dash']
        self.k = kwargs['k']
        self.zeta = kwargs['zeta']
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
        self.episodic_memory = EpisodicMemory(self.episodic_memory_capacity, self.batch_size, self.action_space)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        self.criterion = nn.MSELoss(reduction=self.mseloss_reduction)
        self.total_steps = 0
        self.loss = None

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)

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
            controllable_state = self.embed(state)
            action = np.random.choice(self.action_space)
            self.episodic_memory.add(controllable_state, action)
        else:
            q_values = self.q_value(state)
            aleph = max(q_values) + self.epsilon_dash
            controllable_state = self.embed(state)
            ras = self.calculate_reliability(controllable_state)
            diff = aleph - q_values
            Z = 1.0 / np.sum(1.0 / diff)
            rho = Z / diff
            SRS = ((ras / rho).max() + self.epsilon_dash) * rho - ras
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)

            action = np.random.choice(self.action_space, p=pi)
            self.episodic_memory.add(controllable_state, action)
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return

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

    def calculate_reliability(self, controllable_state):
        controllable_state_and_action = np.array(self.episodic_memory.memory, dtype=np.float32)
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

    def sync_model_hard(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def sync_model_soft(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float32(1.0)-self.tau)*target_param.data)
