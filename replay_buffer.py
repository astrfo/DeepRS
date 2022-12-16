import numpy as np
from collections import deque

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


class EpisodicMemory:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.memory_capacity)

    def reset(self):
        self.memory = deque(maxlen=self.memory_capacity)

    def add(self, controllable_state, action):
        data = (controllable_state, action)
        self.memory.append(data)


if __name__ == '__main__':
    print('started replay_buffer')
    rb = ReplayBuffer(memory_capacity=3, batch_size=2)
    print(f'memory0: {rb.memory}')
    rb.add(state=[1, 2], action=0, reward=1, next_state=[2, 2], done=False)
    print(f'memory1: {rb.memory}')
    rb.add(state=[2, 2], action=1, reward=1, next_state=[2, 3], done=False)
    print(f'memory2: {rb.memory}')
    rb.add(state=[2, 3], action=0, reward=0, next_state=[1, 3], done=False)
    print(f'memory3: {rb.memory}')
    rb.add(state=[1, 3], action=1, reward=0, next_state=[1, 2], done=True)
    print(f'memory4: {rb.memory}')
    print('finished replay_buffer')