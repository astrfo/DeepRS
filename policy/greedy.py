"""greedy policy"""
import math
from typing import Union
import numpy as np

from policy.base_policy import BaseContextualPolicy


#@は行列用の掛け算
class Greedy(BaseContextualPolicy):
    """Greedy policy

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (int): 各腕が選択された回数

    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1,n_steps:int=1) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)

        self.name = 'Greedy'

        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f
        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)
        self.n_steps = n_steps



    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()
        #a:行動数,f:特徴量の次元数
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)

    def choose_arm(self, x: np.ndarray) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す.

        Args:
            x(int, float):特徴量
            step(int):現在のstep数
        Retuens:
            result(int):選んだ行動
        """

        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f,[2,117]
            self.theta_hat_x = self.theta_hat @ x

            result = np.argmax(self.theta_hat_x)

        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """

        super().update(x, chosen_arm, reward)
        x = np.expand_dims(x, axis=1)
        """パラメータの更新"""
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)

        self.b[chosen_arm] += np.ravel(x) * reward

        #更新
        if self.steps % self.batch_size == 0:

            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)


    def get_theta(self) -> np.ndarray:
        """推定量を返す"""
        return self.theta_hat

    def get_theta_x(self) -> np.ndarray:
        """推定量_特徴量ありを返す"""
        return self.theta_hat_x