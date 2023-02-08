import numpy as np

class GridWorld:
    def __init__(self):
        self.map = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.actions = np.array([-1, 1])
        self.width = self.map.shape[0]
        self.discrete_start_state = 0
        self.start_state = self.one_hot(0)
        self.discrete_current_state = self.discrete_start_state
        self.current_state = self.start_state

    def one_hot(self, state):
        one_hot_array = np.zeros(self.map.shape)
        one_hot_array[state] = 1
        return one_hot_array

    def reset(self):
        self.discrete_current_state = self.discrete_start_state
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        if action == 0 and self.discrete_current_state > 0:
            self.discrete_current_state += self.actions[action]
        elif action == 1 and self.discrete_current_state < (self.width - 1):
            self.discrete_current_state += self.actions[action]
        self.current_state = self.one_hot(self.discrete_current_state)
        reward = self.map[self.discrete_current_state]
        if reward > 0:
            terminated, truncated, info = True, False, False
        else:
            terminated, truncated, info = False, False, False
        return self.current_state, reward, terminated, truncated, info

    def close(self):
        pass