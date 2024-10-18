import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim

from memory.replay_buffer import ReplayBuffer
from memory.episodic_memory import EpisodicMemory
from network.rsrs_duelingnet import RSRSDuelingNet

torch.set_default_dtype(torch.float64)


class RSRSDuelingDDQN:
    def __init__(self, model=RSRSDuelingNet, **kwargs):
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
        self.episodic_memory = EpisodicMemory(self.memory_capacity, self.batch_size, self.action_space)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        self.n = np.zeros(self.action_space)
        self.total_step = 0
        self.gamma_G = 0.9
        self.aleph_G = kwargs.get('aleph_G', 1.0)
        self.E_G = 0
        # self.zeta = 1
        self.q_list = [[] for _ in range(self.state_space)]
        self.E_G_list = []
        self.aleph_G_list = []

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.n = np.zeros(self.action_space)
        self.total_step = 0
        self.E_G = 0
        self.q_list = [[] for _ in range(self.state_space)]
        self.E_G_list = []
        self.aleph_G_list = []

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def embed(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.embedding(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state, discrete_state):
        q_values = self.q_value(state)
        self.q_list[discrete_state].append(q_values)
        if len(self.episodic_memory.memory) < self.warmup:
            controllable_state = self.embed(state)
            action = np.random.choice(self.action_space)
            self.episodic_memory.add(controllable_state, action)
        else:
            # q_values = self.q_value(state)
            controllable_state = self.embed(state)
            self.calculate_reliability(controllable_state)
            if (self.n == np.float64(1.0)).any(): self.n = (1 / self.total_step + self.n) / (self.action_space / self.total_step + np.sum(self.n))
            delta_G = min(self.E_G - self.aleph_G, 0)
            aleph = max(q_values) - delta_G
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

    def greedy_action(self, state, discrete_state):
        q_values = self.q_value(state)
        action = np.random.choice(np.where(q_values == max(q_values))[0])
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

        q = self.model(s)
        qa = q[np.arange(self.batch_size), a]

        next_q = self.model(ns)
        next_q_target = self.model_target(ns)
        next_qa_target = next_q_target[np.arange(self.batch_size), torch.argmax(next_q, axis=1)]
        target = r + self.gamma * next_qa_target * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(qa, target)
        loss.backward()
        self.optimizer.step()
        self.sync_model()

    def EG_update(self, total_reward, step):
        self.E_G = total_reward
        # self.E_G = 1/step * total_reward
        self.total_step += step
        self.E_G_list.append(self.E_G)
        self.aleph_G_list.append(self.aleph_G)

    def calculate_reliability(self, controllable_state):
        controllable_state_and_action = np.array([m for m in self.episodic_memory.memory])
        controllable_state_vec = controllable_state_and_action[:, :len(controllable_state)]
        action_vec = controllable_state_and_action[:, len(controllable_state):]
        controllable_state = np.expand_dims(controllable_state, axis=0)
        K_neighbor = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree', metric='euclidean').fit(controllable_state_vec)
        distance, indices = K_neighbor.kneighbors(controllable_state)

        distance = np.squeeze(distance)
        action_vec = action_vec[indices]
        action_vec = np.squeeze(action_vec)
        
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
