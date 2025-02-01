import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss

from memory.replay_buffer import ReplayBuffer
from memory.episodic_memory import EpisodicMemory
from network.conv_atari_rsrsnet import ConvRSRSAtariNet


class ConvRS2AmbitionEMDQN:
    def __init__(self, model=ConvRSRSAtariNet, **kwargs):
        super().__init__()
        self.warmup = kwargs['warmup']
        self.k = kwargs['k']
        self.zeta = kwargs['zeta']
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.epsilon_dash = kwargs['epsilon_dash']
        self.target_update_freq = kwargs['target_update_freq']
        self.tau = kwargs['tau']
        self.hidden_size = kwargs['hidden_size']
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.frame_shape = kwargs['frame_shape']
        self.neighbor_frames = kwargs['neighbor_frames']
        self.replay_buffer_capacity = kwargs['replay_buffer_capacity']
        self.episodic_memory_capacity = kwargs['episodic_memory_capacity']
        self.batch_size = kwargs['batch_size']
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity, self.batch_size)
        self.episodic_memory = EpisodicMemory(self.episodic_memory_capacity, self.batch_size, self.action_space)
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

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
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
        if len(self.episodic_memory.memory) < (self.episodic_memory_capacity // 10):
            controllable_state = self.embed(state)
            action = np.random.choice(self.action_space)
            self.episodic_memory.add(controllable_state, action)
        else:
            q_values = self.q_value(state)
            aleph = max(q_values) + 0.0001
            controllable_state = self.embed(state)
            self.calculate_reliability(controllable_state)
            diff = aleph - q_values
            Z = 1.0 / np.sum(0.0001 / diff)
            rho = Z / diff
            b = self.n / rho - 1.0 + 0.0001
            SRS = (1.0 + max(b)) * rho - self.n
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)

            action = np.random.choice(self.action_space, p=pi)
            self.episodic_memory.add(controllable_state, action)
        return action

    def update(self, state, action, reward, next_state, done):
        reward = max(min(reward, 1), -1)
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=40.0)
        self.optimizer.step()
        if self.total_steps % self.target_update_freq == 0:
            self.sync_model()

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
        self.n = np.average(action_vec[I.flatten()], weights=weight, axis=0)

    def sync_model(self):
        self.model_target.load_state_dict(self.model.state_dict())
