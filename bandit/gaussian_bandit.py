from typing import Tuple

import numpy as np

from bandit.base_bandit import BaseBandit


class GaussianBandit(BaseBandit):
    """誤差項が正規分布を取るバンディット環境クラス

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): ユーザーの特徴数
        scale (float): 各アームの持つ真の線形パラメータθの分散
        noise (float): 報酬に関する誤差項の分散
    """
    def __init__(self, n_sims: int, n_arms: int, n_features: int, scale: float=0.01, noise: float=0.1) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features)
        self.scale = scale
        self.noise = noise

        self.params = np.stack(
            [np.random.multivariate_normal(
                np.zeros(self.n_features),
                self.scale * np.identity(self.n_features), size=self.n_arms).T
             for _ in range(n_sims)], axis=0)  # sim*f*a

    def initialize(self) -> None:
        """パラメータの初期化"""
        pass

    def pull(self, chosen_arm: int, x: np.matrix, n_sim: int) -> Tuple[int, float, int]:
        """選ばれた腕を引き、(報酬, リグレット, 最適解だったかどうか)を返す"""
        e = np.random.normal(loc=0, scale=self.noise)
        mu = self.params[n_sim].T @ x
        reward = np.random.normal(loc=mu[chosen_arm] + e, scale=self.scale)
        regret = np.max(mu) - mu[chosen_arm]
        best_arm = np.argmax(mu)
        success = 1 if chosen_arm == best_arm else 0

        return reward, regret, success
