import numpy as np
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer, EpisodicMemory


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


class ConvQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, neighbor_frames):
        super().__init__()
        C, H, W = input_size

        self.conv1 = nn.Conv2d(in_channels=C*neighbor_frames, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(W)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(H)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x

    def conv2d_size_out(self, size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1


class ConvRSNet(nn.Module):
    def __init__(self, input_size, embed_size, output_size, neighbor_frames):
        super().__init__()
        C, H, W = input_size

        self.conv1 = nn.Conv2d(in_channels=C*neighbor_frames, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(W)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(H)))
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.embed = nn.Linear(512, embed_size)
        self.head = nn.Linear(embed_size, output_size)

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

    def conv2d_size_out(self, size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1


class DQN:
    def __init__(self, model=QNet, **kwargs):
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.action_space = kwargs['action_space']
        self.state_shape = kwargs['state_shape']
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_shape, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_shape, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_shape, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_shape, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            s = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.model(s)
            action = np.random.choice(np.where(q_values == max(q_values))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q = self.model(s)
        q = q[np.arange(self.batch_size), a]
        next_q = self.model_target(ns)
        next_q = torch.amax(next_q, dim=1)
        target = r + self.gamma * next_q * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(target, q)
        loss.backward()
        self.optimizer.step()

    def sync_model(self):
        self.model_target.load_state_dict(self.model.state_dict())


class ConvDQN(nn.Module):
    def __init__(self, model=ConvQNet, **kwargs):
        super().__init__()
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.embed_size = kwargs.get('embed_size', 64)
        self.action_space = kwargs['action_space']
        self.state_shape = kwargs['state_shape']
        self.frame_shape = kwargs['frame_shape']
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.neighbor_frames = kwargs.get('neighbor_frames', 4)
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_value(state)
            action = np.random.choice(np.where(q_values == max(q_values))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q = self.model(s)
        q = q[np.arange(self.batch_size), a]
        next_q = self.model_target(ns)
        next_q = torch.amax(next_q, dim=1)
        target = r + self.gamma * next_q * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(target, q)
        loss.backward()
        self.optimizer.step()

    def sync_model(self):
        self.model_target.load_state_dict(self.model.state_dict())


class RSRS(nn.Module):
    def __init__(self, model=ConvRSNet, **kwargs):
        super().__init__()
        self.aleph = kwargs.get('aleph', 0.7)
        self.warmup = kwargs.get('warmup', 10)
        self.k = kwargs.get('k', 5)
        self.zeta = kwargs.get('zeta', 0.008)
        self.alpha = kwargs.get('alpha', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.embed_size = kwargs.get('embed_size', 64)
        self.action_space = kwargs['action_space']
        self.frame_shape = kwargs['frame_shape']
        self.sync_interval = kwargs.get('sync_interval', 20)
        self.neighbor_frames = kwargs.get('neighbor_frames', 4)
        self.memory_capacity = kwargs.get('memory_capacity', 10**4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.episodic_memory = EpisodicMemory(self.memory_capacity, self.batch_size, self.action_space)
        self.device = torch.device('cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.frame_shape, embed_size=self.embed_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, embed_size=self.embed_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
        self.model = self.model_class(input_size=self.frame_shape, embed_size=self.embed_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, embed_size=self.embed_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
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
        if len(self.episodic_memory.memory) < self.warmup:
            controllable_state = self.embed(state)
            action = np.random.choice(self.action_space)
            self.episodic_memory.add(controllable_state, action)
        else:
            #SRS
            controllable_state = self.embed(state)
            self.calculate_reliability(controllable_state)
            self.N = np.sum(self.n)
            self.q = self.q_value(state)
            adjusted_q = deepcopy(self.q)
            if max(self.q) > self.aleph:
                adjusted_q = self.q - (max(self.q) - self.aleph) - self.epsilon
            self.Z = 1 / (np.sum(1 / (self.aleph - adjusted_q)))
            self.rho = self.Z / (self.aleph - adjusted_q)
            self.b = self.n / self.rho - self.N + self.epsilon
            self.SRS = (self.N + max(self.b)) * self.rho - self.n
            self.pi = self.SRS / np.sum(self.SRS)
            action = np.random.choice(len(self.pi), p=self.pi)
            self.episodic_memory.add(controllable_state, action)
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q = self.model(s)
        q = q[np.arange(self.batch_size), a]
        next_q = self.model_target(ns)
        next_q = torch.amax(next_q, dim=1)
        target = r + self.gamma * next_q * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(target, q)
        loss.backward()
        self.optimizer.step()

    def calculate_reliability(self, controllable_state):
        controllable_state_and_action = np.array([e for e in self.episodic_memory.memory])
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
        regularization_squared_distance = squared_distance / average_squared_distance
        regularization_squared_distance -= self.zeta
        np.putmask(regularization_squared_distance, regularization_squared_distance < 0, 0)
        inverse_kernel_function = [self.epsilon / (i + self.epsilon) for i in regularization_squared_distance]
        sum_kernel = np.sum(inverse_kernel_function)
        weight = [k_i/sum_kernel for k_i in inverse_kernel_function]
        self.n = np.average(action_vec, weights=weight, axis=0)

    def sync_model(self):
        self.model_target.load_state_dict(self.model.state_dict())

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
    s = torch.tensor(state, dtype=torch.float32).to(device)
    # a = torch.tensor(action, dtype=torch.float32).to(device)
    print(f's: {s}')
    q = qnet(s)
    print(f'q: {q}')
    print(f'q_a: {q[action]}')
    q = q[action]

    next_state = [0, 3, 7, 6, 1, 7, 3]
    ns = torch.tensor(next_state, dtype=torch.float32).to(device)
    print(f'ns: {ns}')
    next_q = qnet(ns)
    print(f'next_q: {next_q}')
    print(f'next_q_a: {max(next_q)}')

    reward = 1
    gamma = 0.99
    r = torch.tensor(reward, dtype=torch.float32).to(device)
    target = r + gamma * max(next_q)
    print(f'r: {r}')
    print(f'target: {target}')

    optimizer.zero_grad()
    loss = criterion(target, q)
    loss.backward()
    optimizer.step()

    print('finished policy')