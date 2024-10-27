import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory.replay_buffer import ReplayBuffer
from network.qnet import QNet
from network.rndnet import RNDNet

torch.set_default_dtype(torch.float64)


class DQN_RND:
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
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.rnd_model_pred = RNDNet(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.hidden_size)
        self.rnd_model_pred.to(self.device)
        self.rnd_model_target = RNDNet(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.hidden_size)
        self.rnd_model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.rnd_optimizer = optim.Adam(self.rnd_model_pred.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss(reduction='sum')
        for param in self.rnd_model_target.parameters():
            param.requires_grad = False

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.action_space)
        self.model_target.to(self.device)
        self.rnd_model_pred = RNDNet(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.hidden_size)
        self.rnd_model_pred.to(self.device)
        self.rnd_model_target = RNDNet(input_size=self.state_space, hidden_size=self.hidden_size, output_size=self.hidden_size)
        self.rnd_model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.rnd_optimizer = optim.Adam(self.rnd_model_pred.parameters(), lr=self.alpha)

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
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
        intrinsic_reward = self.get_intrinsic_reward(state)
        reward += intrinsic_reward
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
        next_q_target = self.model_target(ns)
        next_qa_target = torch.amax(next_q_target, dim=1)
        target = r + self.gamma * next_qa_target * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(qa, target)
        loss.backward()
        self.optimizer.step()

        target_feature = self.rnd_model_target(s).detach()
        predictor_feature = self.rnd_model_pred(s)
        
        self.rnd_optimizer.zero_grad()
        rnd_loss = self.criterion(predictor_feature, target_feature)
        rnd_loss.backward()
        self.rnd_optimizer.step()

        self.sync_model()

    def sync_model(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)

    def get_intrinsic_reward(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            target_feature = self.rnd_model_target(s)
            predictor_feature = self.rnd_model_pred(s)
            intrinsic_reward = torch.mean((target_feature - predictor_feature) ** 2, dim=-1)
        return intrinsic_reward
