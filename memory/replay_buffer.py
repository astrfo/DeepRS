import numpy as np


class ReplayBuffer:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.index = 0

    def initialize(self):
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
