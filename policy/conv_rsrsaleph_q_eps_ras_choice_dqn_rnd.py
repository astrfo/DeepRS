import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss

from memory.replay_buffer import ReplayBuffer
from memory.episodic_memory import EpisodicMemory
from network.conv_rsrsnet import ConvRSRSNet
from network.conv_rndnet import ConvRNDNet

torch.set_default_dtype(torch.float64)


class ConvRSRSAlephQEpsRASChoiceDQN_RND:
    def __init__(self, model=ConvRSRSNet, **kwargs):
        super().__init__()
        self.warmup = kwargs['warmup']
        self.k = kwargs['k']
        self.zeta = kwargs['zeta']
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.epsilon_dash = kwargs['epsilon_dash']
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
        self.rnd_model_pred = ConvRNDNet(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.rnd_model_pred.to(self.device)
        self.rnd_model_target = ConvRNDNet(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.rnd_model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.rnd_optimizer = optim.Adam(self.rnd_model_pred.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='sum')
        self.n = np.zeros(self.action_space)
        for param in self.rnd_model_target.parameters():
            param.requires_grad = False

    def reset(self):
        self.replay_buffer.reset()
        self.episodic_memory.reset()
        self.model = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model.to(self.device)
        self.model_target = self.model_class(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.model_target.to(self.device)
        self.rnd_model_pred = ConvRNDNet(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.rnd_model_pred.to(self.device)
        self.rnd_model_target = ConvRNDNet(input_size=self.frame_shape, hidden_size=self.hidden_size, output_size=self.action_space, neighbor_frames=self.neighbor_frames)
        self.rnd_model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.rnd_optimizer = optim.Adam(self.rnd_model_pred.parameters(), lr=self.learning_rate)
        self.n = np.zeros(self.action_space)
        for param in self.rnd_model_target.parameters():
            param.requires_grad = False

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

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
            aleph = max(q_values) + np.float64(1e-10)
            controllable_state = self.embed(state)
            self.calculate_reliability(controllable_state)
            diff = aleph - q_values
            Z = np.float64(1.0) / np.sum(np.float64(1.0) / diff)
            rho = Z / diff
            b = self.n / rho - np.float64(1.0) + np.float64(1e-10)
            SRS = (np.float64(1.0) + max(b)) * rho - self.n
            if min(SRS) < 0: SRS -= min(SRS)
            pi = SRS / np.sum(SRS)

            action = np.random.choice(self.action_space, p=pi)
            self.episodic_memory.add(controllable_state, action)
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
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)

    def get_intrinsic_reward(self, state):
        s = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            target_feature = self.rnd_model_target(s)
            predictor_feature = self.rnd_model_pred(s)
            intrinsic_reward = torch.mean((target_feature - predictor_feature) ** 2, dim=-1)
        return intrinsic_reward.squeeze().to('cpu').detach().numpy().copy()
