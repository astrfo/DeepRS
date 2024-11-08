import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss

from memory.replay_buffer import ReplayBuffer
from memory.episodic_memory import EpisodicMemory
from network.conv_rsrsalephnet import ConvRSRSAlephNet

torch.set_default_dtype(torch.float64)


class ConvRSRSAlephDQN(nn.Module):
    def __init__(self, model=ConvRSRSAlephNet, **kwargs):
        super().__init__()
        self.warmup = kwargs['warmup']
        self.k = kwargs['k']
        self.zeta = kwargs['zeta']
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.epsilon = kwargs['epsilon']
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
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='sum')
        self.n = np.zeros(self.action_space)

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.n = np.zeros(self.action_space)

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values, _ = self.model(s)
            return q_values.squeeze().to('cpu').detach().numpy().copy()

    def aleph_s(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            _, aleph_s = self.model(s)
            return aleph_s.squeeze().to('cpu').detach().numpy().copy()

    def embed(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.embedding(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        if len(self.episodic_memory.memory) < self.warmup:
            controllable_state = self.embed(state)
            action = np.random.choice(self.action_space)
            self.episodic_memory.add(controllable_state, action)
        else:
            q_values = self.q_value(state)
            aleph = self.aleph_s(state)
            controllable_state = self.embed(state)
            self.calculate_reliability(controllable_state)
            if max(q_values) >= aleph:
                fix_aleph = max(q_values) + np.float64(1e-10)
                diff = fix_aleph - q_values
                if min(diff) < 0: diff -= min(diff)
                Z = np.float64(1.0) / np.sum(np.float64(1.0) / diff)
                rho = Z / diff
            else:
                Z = 1 / np.sum(np.float64(1.0) / (aleph - q_values))
                rho = Z / (aleph - q_values)
            b = self.n / rho - np.float64(1.0) + np.float64(1e-10)
            SRS = (np.float64(1.0) + max(b)) * rho - self.n
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)

            prob = np.random.rand()
            top, bottom = self.action_space, -1
            while (top - bottom > 1):
                mid = int(bottom + (top - bottom)/2)
                if prob < np.sum(pi[0:mid]): top = mid
                else: bottom = mid
            if mid == bottom: action = mid
            else: action = mid-1
            self.episodic_memory.add(controllable_state, action)
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float64).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float64).to(self.device)
        r = torch.tensor(r, dtype=torch.float64).to(self.device)
        d = torch.tensor(d, dtype=torch.float64).to(self.device)

        q, _ = self.model(s)
        qa = q[np.arange(self.batch_size), a]
        next_q_target, _ = self.model_target(ns)
        next_qa_target = torch.amax(next_q_target, dim=1)
        target = r + self.gamma * next_qa_target * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(qa, target)
        loss.backward()
        self.optimizer.step()
        self.sync_model()

    def calculate_reliability(self, controllable_state):
        controllable_state_and_action = np.array([m for m in self.episodic_memory.memory])
        controllable_state_vec = controllable_state_and_action[:, :len(controllable_state)]
        action_vec = controllable_state_and_action[:, len(controllable_state):]
        controllable_state = np.expand_dims(controllable_state, axis=0)
        
        index = faiss.IndexFlatL2(controllable_state_vec.shape[1])
        index.add(controllable_state_vec.astype(np.float32))
        D, I = index.search(controllable_state.astype(np.float32), self.k)

        distance = D.flatten()
        action_vec = action_vec[I.flatten()]

        squared_distance = np.asarray(distance) ** 2
        average_squared_distance = np.average(squared_distance)
        regularization_squared_distance = np.divide(squared_distance, average_squared_distance, out=np.zeros_like(squared_distance), where=average_squared_distance!=0)
        regularization_squared_distance -= self.zeta
        np.putmask(regularization_squared_distance, regularization_squared_distance < 0, 0)
        inverse_kernel_function = [self.epsilon / (i + self.epsilon) for i in regularization_squared_distance]
        sum_kernel = np.sum(inverse_kernel_function)
        weight = [k_i/sum_kernel for k_i in inverse_kernel_function]
        self.n = np.average(action_vec, weights=weight, axis=0)

    def sync_model(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)
