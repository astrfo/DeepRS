"""シンプルLinRS"""
import math
from typing import Union

import numpy as np

from policy.base_policy import BaseContextualPolicy


#@は行列用の掛け算
class LinRS(BaseContextualPolicy):
    """Linear Risk-sensitive Satisficing Value Function

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (int): 各腕が選択された回数
        aleph (float): 満足化基準値

    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1, aleph: float=1.0) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)
        self.aleph = aleph
        self.name = 'LinRS ℵ={}'.format(self.aleph)
        
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f
        self.m = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))
        self._m = np.zeros((self.n_arms, self.n_features))

        self.theta_hat=np.zeros((self.n_arms, self.n_features))
        self.phi_hat=np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x=np.zeros(self.n_arms)
        self.phi_hat_x=np.zeros(self.n_arms)
        

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()
        #a:行動数,f:特徴量の次元数
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f
        self.m = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))
        self._m = np.zeros((self.n_arms, self.n_features))

        self.theta_hat=np.zeros((self.n_arms, self.n_features))
        self.phi_hat=np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x=np.zeros(self.n_arms)
        self.phi_hat_x=np.zeros(self.n_arms)

    def choose_arm(self, x: np.ndarray,step) -> np.ndarray:
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
            self.phi_hat = np.array([self._A_inv[i] @ self._m[i] for i in range(self.n_arms)])  # a*f,[2,117]
            self.theta_hat_x = self.theta_hat @ x
            self.phi_hat_x = self.phi_hat @ x
            rs = self.phi_hat_x * (self.theta_hat_x - self.aleph)  # a*1,[2]
            result = np.argmax(rs)
        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータ更新
        
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        super().update(x, chosen_arm, reward)

        x = np.expand_dims(x, axis=1)#配列1の手前に次元を追加,[117]→[117,1]
        """パラメータの更新"""
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
        
        self.b[chosen_arm] += np.ravel(x) * reward#ravel,1次元のリストに変換
        self.m[chosen_arm] += np.ravel(x)

        if self.steps % self.batch_size == 0:
            self._A_inv, self._b, self._m = np.copy(self.A_inv), np.copy(self.b), np.copy(self.m)

    def get_theta_x(self) -> np.ndarray:
        """推定量_特徴量ありを返す"""    
        return self.theta_hat_x

    def get_phi_x(self) -> np.ndarray:
        """疑似試行回数_特徴量ありを返す"""   
        return self.phi_hat_x
        
           
                     
