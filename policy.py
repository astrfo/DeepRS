import sys
from copy import deepcopy
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer, EpisodicMemory

torch.set_default_dtype(torch.float64)


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RSNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.embed(x))
        x = self.fc2(x)
        return x

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = self.embed(x)
        return x


class ConvQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, neighbor_frames):
        super().__init__()
        C, H, W = input_size
        self.kernel_sizes = [8, 4, 3]
        self.strides = [4, 2, 1]

        self.conv1 = nn.Conv2d(in_channels=C*neighbor_frames, out_channels=16, kernel_size=self.kernel_sizes[0], stride=self.strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_sizes[1], stride=self.strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_sizes[2], stride=self.strides[2])
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(W, n_layer=3)
        convh = self.conv2d_size_out(H, n_layer=3)
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.embed = nn.Linear(512, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.embed(x)
        x = self.head(x)
        return x

    def conv2d_size_out(self, size, n_layer):
        cnt = 0
        size_out = size
        while cnt < n_layer:
            size_out = (size_out - self.kernel_sizes[cnt]) // self.strides[cnt] + 1
            cnt += 1
        return size_out


class ConvRSNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, neighbor_frames):
        super().__init__()
        C, H, W = input_size
        self.kernel_sizes = [8, 4, 3]
        self.strides = [4, 2, 1]

        self.conv1 = nn.Conv2d(in_channels=C*neighbor_frames, out_channels=16, kernel_size=self.kernel_sizes[0], stride=self.strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_sizes[1], stride=self.strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_sizes[2], stride=self.strides[2])
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(W, n_layer=3)
        convh = self.conv2d_size_out(H, n_layer=3)
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.embed = nn.Linear(512, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.embed(x)
        x = self.head(x)
        return x

    def embedding(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.embed(x)
        return x

    def conv2d_size_out(self, size, n_layer):
        cnt = 0
        size_out = size
        while cnt < n_layer:
            size_out = (size_out - self.kernel_sizes[cnt]) // self.strides[cnt] + 1
            cnt += 1
        return size_out


class DQN:
    def __init__(self, model=QNet, **kwargs):
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.tau = kwargs.get('tau', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        self.q_list = [[] for _ in range(self.state_space)]
        self.batch_reward_list = []

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.q_list = [[] for _ in range(self.state_space)]
        self.batch_reward_list = []

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state, discrete_state):
        q_values = self.q_value(state)
        self.q_list[discrete_state].append(q_values)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.random.choice(np.where(q_values == max(q_values))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        self.batch_reward_list.append(np.count_nonzero(r == 1.0))

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
        self.sync_model()

    def EG_update(self, *args):
        pass

    def sync_model(self):
        # self.model_target.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)


class DDQN:
    def __init__(self, model=QNet, **kwargs):
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.tau = kwargs.get('tau', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        self.q_list = [[] for _ in range(self.state_space)]

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.q_list = [[] for _ in range(self.state_space)]

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state, discrete_state):
        q_values = self.q_value(state)
        self.q_list[discrete_state].append(q_values)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
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
        next_qa = torch.argmax(next_q, dim=1, keepdim=True)
        next_q_target = self.model_target(ns)
        next_qa_target = next_q_target.gather(1, next_qa).squeeze()
        target = r + self.gamma * next_qa_target * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(qa, target)
        loss.backward()
        self.optimizer.step()
        self.sync_model()

    def EG_update(self, *args):
        pass

    def sync_model(self):
        # self.model_target.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)


class RSRS:
    def __init__(self, model=RSNet, **kwargs):
        self.aleph = kwargs.get('aleph', 0.7)
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
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.episodic_memory = EpisodicMemory(self.memory_capacity, self.batch_size, self.action_space)
        self.device = torch.device('cpu')
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
        self.aleph_G = 0.9
        self.E_G = 0
        # self.zeta = 1
        self.q_list = [[] for _ in range(self.state_space)]
        self.pi_list = []
        self.batch_reward_list = []

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
        self.pi_list = []
        self.batch_reward_list = []

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
            controllable_state = self.embed(state)
            self.calculate_reliability(controllable_state)
            if (self.n == np.float64(1.0)).any(): self.n = (1 / self.total_step + self.n) / (self.action_space / self.total_step + np.sum(self.n))
            delta_G = min(self.E_G - self.aleph_G, 0)
            aleph = max(q_values) - delta_G
            if max(q_values) >= aleph:
                fix_aleph = max(q_values) + sys.float_info.epsilon
                diff = fix_aleph - q_values
                if min(diff) < 0: diff -= min(diff)
                Z = np.float64(1.0) / np.sum(np.float64(1.0) / diff)
                rho = Z / diff
            else:
                Z = 1 / np.sum(np.float64(1.0) / (aleph - q_values))
                rho = Z / (aleph - q_values)
            b = self.n / rho - np.float64(1.0) + sys.float_info.epsilon
            SRS = (np.float64(1.0) + max(b)) * rho - self.n
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)
            self.pi_list.append(pi)

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
        self.batch_reward_list.append(np.count_nonzero(r == 1.0))

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
        self.sync_model()

    def EG_update(self, total_reward, step):
        self.E_G = total_reward
        self.total_step += step

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
        # self.model_target.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)


class ConvDQN(nn.Module):
    def __init__(self, model=ConvQNet, **kwargs):
        super().__init__()
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.tau = kwargs.get('tau', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.frame_shape = kwargs['frame_shape']
        self.neighbor_frames = kwargs.get('neighbor_frames', 4)
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        self.q_list = [[] for _ in range(self.state_space)]
        self.batch_reward_list = []

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.q_list = [[] for _ in range(self.state_space)]
        self.batch_reward_list = []

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state, discrete_state):
        q_values = self.q_value(state)
        self.q_list[discrete_state].append(q_values)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.random.choice(np.where(q_values == max(q_values))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        self.batch_reward_list.append(np.count_nonzero(r == 1.0))

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
        self.sync_model()

    def EG_update(self, *args):
        pass

    def sync_model(self):
        # self.model_target.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)


class ConvDDQN(nn.Module):
    def __init__(self, model=ConvQNet, **kwargs):
        super().__init__()
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.tau = kwargs.get('tau', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.frame_shape = kwargs['frame_shape']
        self.neighbor_frames = kwargs.get('neighbor_frames', 4)
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        self.q_list = [[] for _ in range(self.state_space)]

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.q_list = [[] for _ in range(self.state_space)]

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state, discrete_state):
        q_values = self.q_value(state)
        self.q_list[discrete_state].append(q_values)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
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
        next_qa = torch.argmax(next_q, dim=1, keepdim=True)
        next_q_target = self.model_target(ns)
        next_qa_target = next_q_target.gather(1, next_qa).squeeze()
        target = r + self.gamma * next_qa_target * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(qa, target)
        loss.backward()
        self.optimizer.step()
        self.sync_model()

    def EG_update(self, *args):
        pass

    def sync_model(self):
        # self.model_target.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)


class ConvRSRS(nn.Module):
    def __init__(self, model=ConvRSNet, **kwargs):
        super().__init__()
        self.aleph = kwargs.get('aleph', 0.7)
        self.warmup = kwargs.get('warmup', 10)
        self.k = kwargs.get('k', 5)
        self.zeta = kwargs.get('zeta', 0.008)
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.tau = kwargs.get('tau', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.frame_shape = kwargs['frame_shape']
        self.neighbor_frames = kwargs.get('neighbor_frames', 4)
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.episodic_memory = EpisodicMemory(self.memory_capacity, self.batch_size, self.action_space)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        self.n = np.zeros(self.action_space)
        self.total_step = 0
        self.gamma_G = 0.9
        self.aleph_G = 0.9
        self.E_G = 0
        # self.zeta = 1
        self.q_list = [[] for _ in range(self.state_space)]
        self.pi_list = []
        self.batch_reward_list = []

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.n = np.zeros(self.action_space)
        self.total_step = 0
        self.E_G = 0
        self.q_list = [[] for _ in range(self.state_space)]
        self.pi_list = []
        self.batch_reward_list = []

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
            controllable_state = self.embed(state)
            self.calculate_reliability(controllable_state)
            if (self.n == np.float64(1.0)).any(): self.n = (1 / self.total_step + self.n) / (self.action_space / self.total_step + np.sum(self.n))
            delta_G = min(self.E_G - self.aleph_G, 0)
            aleph = max(q_values) - delta_G
            if max(q_values) >= aleph:
                fix_aleph = max(q_values) + sys.float_info.epsilon
                diff = fix_aleph - q_values
                if min(diff) < 0: diff -= min(diff)
                Z = np.float64(1.0) / np.sum(np.float64(1.0) / diff)
                rho = Z / diff
            else:
                Z = 1 / np.sum(np.float64(1.0) / (aleph - q_values))
                rho = Z / (aleph - q_values)
            b = self.n / rho - np.float64(1.0) + sys.float_info.epsilon
            SRS = (np.float64(1.0) + max(b)) * rho - self.n
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)
            self.pi_list.append(pi)

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
        self.batch_reward_list.append(np.count_nonzero(r == 1.0))

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
        self.sync_model()

    def EG_update(self, total_reward, step):
        self.E_G = total_reward
        self.total_step += step

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
        # self.model_target.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)

    def extract(self, target, inputs):
        self.feature = None
        def forward_hook(model, inputs, outputs):
            self.feature = outputs.detach().clone()
        handle = target.register_forward_hook(forward_hook)
        self.model(inputs)
        handle.remove()


if __name__ == '__main__':
    print('started policy')

    qnet = QNet(input_size=7, hidden_size=14, output_size=2)
    device = torch.device('cpu')
    optimizer = optim.Adam(qnet.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    state = [0, 1, 4, 6, 8, 1, 0]
    action = 0
    s = torch.tensor(state, dtype=torch.float64).to(device)
    # a = torch.tensor(action, dtype=torch.float64).to(device)
    print(f's: {s}')
    q = qnet(s)
    print(f'q: {q}')
    print(f'q_a: {q[action]}')
    q = q[action]

    next_state = [0, 3, 7, 6, 1, 7, 3]
    ns = torch.tensor(next_state, dtype=torch.float64).to(device)
    print(f'ns: {ns}')
    next_q = qnet(ns)
    print(f'next_q: {next_q}')
    print(f'next_q_a: {max(next_q)}')

    reward = 1
    gamma = 0.99
    r = torch.tensor(reward, dtype=torch.float64).to(device)
    target = r + gamma * max(next_q)
    print(f'r: {r}')
    print(f'target: {target}')

    optimizer.zero_grad()
    loss = criterion(target, q)
    loss.backward()
    optimizer.step()

    print('finished policy')