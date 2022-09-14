import math
from typing import Union

import numpy as np

from policy.base_policy import BaseContextualPolicy


class LinUCB(BaseContextualPolicy):
    """Linear Upper Confidence Bound

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (int): 各腕が選択された回数
        alpha (float): 学習率
    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1, alpha: float=0.1) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)
        self.alpha = alpha
        #self.name = 'LinUCB α={}'.format(self.alpha)
        self.name = 'LinUCB'

        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat=np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat=np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)

    def choose_arm(self, x: np.ndarray) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す"""
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f
            self.theta_hat_x = self.theta_hat @ x
            #sigma_hat : rootの中の計算
            sigma_hat = np.array([np.sqrt(x.T @ self._A_inv[i] @ x) for i in range(self.n_arms)])  # a * (1*f @ f*f @ f*1) -> a*1
            mu_hat = self.theta_hat @ x + self.alpha*math.sqrt(math.log(self.steps+1)) * sigma_hat  # a*f @ f*1 + a*1
            result = np.argmax(mu_hat)

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

    def get_theta_x(self) -> np.ndarray:
        """推定量を返す"""
        return self.theta_hat_x
            

