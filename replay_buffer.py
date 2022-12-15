from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.memory_capacity)

    def reset(self):
        self.memory = deque(maxlen=self.memory_capacity)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.memory.append(data)

    def encode(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices = np.random.randint(0, len(self.memory), self.batch_size)
        for index in indices:
            s, a, r, ns, d = self.memory[index]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


if __name__ == '__main__':
    print('started replay_buffer')
    memory = ReplayBuffer(memory_capacity=10, batch_size=2)
    print('finished replay_buffer')