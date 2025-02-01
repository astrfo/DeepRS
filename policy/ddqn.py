import numpy as np
import torch

from policy.base_policy import BasePolicy


class DDQN(BasePolicy):
    def __init__(self, model_class, **kwargs):
        super().__init__(model_class, **kwargs)
        self.epsilon_fixed = kwargs['epsilon_fixed']

    def action(self, state):
        self.total_steps += 1
        if np.random.rand() < self.epsilon_fixed:
            action = np.random.choice(self.action_space)
        else:
            q_value = self.calc_q_value(state)
            action = np.random.choice(np.where(q_value == max(q_value))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float64).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float64).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float64).to(self.device)
        d = torch.tensor(d, dtype=torch.float64).to(self.device)

        q = self.model(s)
        qa = q[np.arange(self.batch_size), a]
        with torch.no_grad():
            next_q = self.model(ns)
            next_qa = torch.argmax(next_q, dim=1, keepdim=True)
            next_q_target = self.model_target(ns)
            next_qa_target = next_q_target.gather(1, next_qa).squeeze()
        
        target = r + self.gamma * next_qa_target * (1 - d)
        self.loss = self.criterion(qa, target)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.sync_model_update == 'hard':
            if self.total_steps % self.target_update_freq == 0:
                self.sync_model_hard()
        elif self.sync_model_update == 'soft':
            self.sync_model_soft()
        else:
            raise ValueError(f'Invalid sync_model_update: {self.sync_model_update}')
