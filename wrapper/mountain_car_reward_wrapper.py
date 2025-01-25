import numpy as np
from gymnasium import RewardWrapper

class MountainCarRewardWrapper(RewardWrapper):
    """
    位置エネルギーと運動エネルギーに基づく報酬を計算
    位置エネルギー: mgh, 運動エネルギー: 0.5mv^2
    m: 質量, g: 重力加速度, h: 高さ, v: 速度
    """
    def __init__(self, env):
        super().__init__(env)
        # 初期状態を取得して保存
        initial_state, _ = env.reset()
        self.initial_position = initial_state[0]
        self.initial_velocity = initial_state[1]
        self.m = 1.0
        self.g = 0.0025

    def energy(self, position, velocity):
        h = np.sin((position - (0.5 * np.pi)) + 0.5) + 1.0
        u = self.m * self.g * h  # 位置エネルギー
        k = 0.5 * velocity**2  # 運動エネルギー
        xmax = 0.6  # 最大位置
        vmax = 0.07  # 最大速度
        n = 1 / ((self.m * self.g * (np.sin((xmax - (0.5 * np.pi)) + 0.5) + 1.0)) + (0.5 * vmax**2))
        e = n * (u + k)
        return e

    def reward(self, reward):
        # 環境から現在の状態 (position, velocity) を取得
        current_position, current_velocity = self.env.unwrapped.state

        modified_reward = (reward * 0.01) + (self.energy(current_position, current_velocity) - self.energy(self.initial_position, self.initial_velocity))
        if current_position >= 0.5:  # ゴール位置 (0.5)
            modified_reward += 1.0

        return modified_reward
