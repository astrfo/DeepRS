import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory.replay_buffer import ReplayBuffer
from network.conv_atari_qnet import ConvQAtariNet

torch.set_default_dtype(torch.float64)


class ConvDQNAtari(nn.Module):
    def __init__(self, model=ConvQAtariNet, **kwargs):
        super().__init__()
        self.alpha = kwargs['alpha']
        self.gamma = kwargs['gamma']
        self.epsilon = kwargs['epsilon']
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 1e6
        self.total_steps = 0
        self.warmup_steps = 1e2
        self.tau = kwargs['tau']
        self.hidden_size = kwargs['hidden_size']
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.frame_shape = kwargs['frame_shape']
        self.neighbor_frames = kwargs['neighbor_frames']
        self.memory_capacity = kwargs['memory_capacity']
        self.batch_size = kwargs['batch_size']
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.criterion = nn.SmoothL1Loss()

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

    def q_value(self, state):
        s = torch.tensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.total_steps / self.epsilon_decay)
        self.total_steps += 1
        q_values = self.q_value(state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(q_values)
        return action

    def update(self, state, action, reward, next_state, done):
        reward = max(min(reward, 1), -1)
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup_steps:
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.sync_model()

    def sync_model(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
