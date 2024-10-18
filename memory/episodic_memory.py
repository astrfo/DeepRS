import numpy as np


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
