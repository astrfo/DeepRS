import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.index = 0

    def reset(self):
        self.memory = []
        self.index = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if self.memory_capacity > len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.index] = data
        self.index = (self.index + 1) % self.memory_capacity

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
    def __init__(self, memory_capacity, batch_size, action_space):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.action_space = action_space
        self.memory = []
        self.index = 0

    def reset(self):
        self.memory = []
        self.index = 0

    def add(self, controllable_state, action):
        action_one_hot = np.zeros((self.action_space))
        action_one_hot[action] = 1
        data = np.append(controllable_state, action_one_hot, axis=0)
        if self.memory_capacity > len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.index] = data
        self.index = (self.index + 1) % self.memory_capacity


if __name__ == '__main__':
    print('started replay_buffer')
    em = EpisodicMemory(memory_capacity=3, batch_size=2)
    em.add([1, 1], 1)
    print(em.memory)
    em.add([1, 2], 0)
    print(em.memory)
    em.add([2, 2], 0)
    print(em.memory)
    a = np.array([r for r in em.memory])
    b = a[:, :2]
    c = a[:, 2:]
    print(f'state: {b}')
    print(f'action: {c}')
    em.memory.pop(0)
    print(em.memory)
    print('finished replay_buffer')