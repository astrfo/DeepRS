import math
from typing import Union

import numpy as np

from policy.base_policy import BaseContextualPolicy


class LinTS(BaseContextualPolicy):
    """Linear Thompson Sampling

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): ユーザーの特徴数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (list[int]): 各腕が選択された回数
        sigma (float): 誤差項(事前ガウシアン分布)の分散
    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1, sigma: float=1.0) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)
        self.sigma = sigma
        self.name = 'LinTS σ^2={}'.format(self.sigma)

        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat=np.zeros((self.n_arms, self.n_features))

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat=np.zeros((self.n_arms, self.n_features))

    def choose_arm(self, x: np.ndarray) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す"""
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f
            scale = self.sigma * self._A_inv
            theta_tilde = [np.random.multivariate_normal(mean=self.theta_hat[i], cov=scale[i]) for i in range(self.n_arms)]  # a*f
            result = np.argmax(theta_tilde @ x)

        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータの更新"""
        super().update(x, chosen_arm, reward)

        x = np.expand_dims(x, axis=1)
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
        self.b[chosen_arm] += np.ravel(x) * reward

        if self.steps % self.batch_size == 0:
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)
