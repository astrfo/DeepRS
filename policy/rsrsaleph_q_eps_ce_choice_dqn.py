import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory.replay_buffer import ReplayBuffer
from network.rsrscenet import RSRSCENet


class RSRSAlephQEpsCEChoiceDQN:
    def __init__(self, model=RSRSCENet, **kwargs):
        self.gamma = kwargs['gamma']
        self.epsilon_dash = kwargs['epsilon_dash']
        self.k = kwargs['k']
        self.adam_learning_rate = kwargs['adam_learning_rate']
        self.replay_buffer_capacity = kwargs['replay_buffer_capacity']
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.total_steps = 0
        self.loss = None

    def reset(self):
        self.replay_buffer.reset()
        self.model = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.state_space, hidden_size=self.hidden_size, embedding_size=self.embedding_size, output_size=self.action_space).float()
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        self.total_steps = 0

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values, _ = self.model(s)
            return q_values.squeeze().to('cpu').detach().numpy().copy()

    def embed(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model.embedding(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        self.total_steps += 1
        if self.total_steps < self.warmup:
            controllable_state = self.embed(state)
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_value(state)
            aleph = q_values.max() + self.epsilon_dash
            ras = self.calculate_reliability(state)
            diff = aleph - q_values
            z = 1.0 / np.sum(1.0 / diff)
            rho = z / diff
            rsrs = ((ras / rho).max() + sys.float_info.epsilon) * rho - ras

            if np.any(rsrs <= 0):
                rsrs -= np.min(rsrs)
                rsrs += sys.float_info.epsilon

            log_rsrs = np.log(rsrs)
            exp_rsrs = np.exp(log_rsrs - log_rsrs.max())
            pi = exp_rsrs / np.sum(exp_rsrs)

            action = np.random.choice(self.action_space, p=pi)
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

        q, _ = self.model(s)
        qa = q[np.arange(self.batch_size), a]
        with torch.no_grad():
            next_q_target, _ = self.model_target(ns)
            next_qa_target = torch.amax(next_q_target, dim=1)
        
        target = r + self.gamma * next_qa_target * (1 - d)
        q_loss = self.criterion(qa, target)

        one_hot_action = np.zeros(self.action_space)
        one_hot_action[action] = 1
        one_hot_action = torch.tensor(one_hot_action, dtype=torch.float32).to(self.device)
        confidence = self.calculate_reliability(state)
        confidence = torch.tensor(confidence, dtype=torch.float32).to(self.device)
        confidence_loss = self.ce_criterion(confidence, one_hot_action)

        self.loss = q_loss + confidence_loss

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.sync_model_update == 'hard':
            if self.total_steps % self.target_update_freq == 0:
                self.sync_model_hard()
        elif self.sync_model_update == 'soft':
            self.sync_model_soft()

    def calculate_reliability(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            _, confidence = self.model(s)
            confidence = confidence.squeeze().to('cpu').detach().numpy().copy()
        
        if np.any(confidence <= 0):
            confidence -= np.min(confidence)
            confidence += sys.float_info.epsilon
        
        log_ras = np.log(confidence)
        exp_ras = np.exp(log_ras - log_ras.max())
        ras = exp_ras / np.sum(exp_ras)
        return ras

    def sync_model_hard(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def sync_model_soft(self):
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float32(1.0)-self.tau)*target_param.data)
