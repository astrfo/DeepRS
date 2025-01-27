import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper


class OneHotObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(env.observation_space.n,), dtype=np.float32
        )

    def observation(self, observation):
        one_hot = np.zeros(self.observation_space.shape, dtype=np.float32)
        one_hot[observation] = 1.0
        return one_hot
