import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer

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


class DQN:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 32
        self.hidden_size = 128
        self.action_space = 2
        self.state_shape = 4
        self.sync_interval = 20
        self.memory_capacity = 10**4
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cpu')
        self.model = QNet(input_size=self.state_shape, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = QNet(input_size=self.state_shape, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def reset(self):
        self.replay_buffer.reset()

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

        q = self.model(s)
        q = q[np.arange(self.batch_size), a]
        next_q = self.model_target(ns)
        next_q = torch.amax(next_q, dim=1)
        target = r + self.gamma * next_q

        self.optimizer.zero_grad()
        loss = self.criterion(target, q)
        loss.backward()
        self.optimizer.step()

    def sync_model(self):
        self.model_target.load_state_dict(self.model.state_dict())


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