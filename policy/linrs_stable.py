"""StableLinRS"""
import math
from typing import Union
import numpy as np
from collections import deque
from scipy.stats import entropy

from policy.base_policy import BaseContextualPolicy


#@は行列用の掛け算
class StableLinRS(BaseContextualPolicy):
    """Linear Risk-sensitive Satisficing Value Function

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (int): 各腕が選択された回数
        aleph (float): 満足化基準値
        eta (float): 学習率
        t (float): softmax用温度パラメータ

    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1,n_steps:int=1, aleph: float=1.0, eta: float=1.0) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)
        self.aleph = aleph
        self.eta = eta
        
        #self.name = 'Stable LinRS ℵ={}'.format(self.aleph)
        self.name = 'Stable LinRS'
        
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f
        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)
        self.phi_hat = np.zeros((self.n_arms, self.n_features))
        self.n_steps = n_steps
        self.m = deque(maxlen=100)
        self.n = deque(maxlen=100)
        self._x = deque(maxlen=100)
        self.rho = np.zeros((self.n_arms,1))
        self.rs = np.zeros(self.n_arms)
        self.n_x = np.zeros(self.n_arms)

        

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
        self.phi_hat = np.zeros((self.n_arms, self.n_features))
        
        self.m = deque(maxlen=100)
        self.n = deque(maxlen=100)
        self._x = deque(maxlen=100)
        self.rho = np.zeros((self.n_arms,1))
        self.rs = np.zeros(self.n_arms)
        self.n_x = np.zeros(self.n_arms)

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
            self.n_x = self.softmax(self.phi_hat@x)
            self.rs = self.n_x *(self.theta_hat_x - self.aleph)  # a*1,[2]

            result = np.argmax(self.rs)

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


        #first in first out で target を格納
        self.T = np.zeros((self.n_arms,1))
        self.T[chosen_arm] = 1#選択した腕に対してのみ1を加える
        if self.steps != 1:
            self.rho = self.counts/(self.steps-1)
            self.rho[chosen_arm] = (self.counts[chosen_arm]-1)/(self.steps-1)
            self.rho = np.expand_dims(self.rho, axis=1)
            self.m.append(((self.eta * self.T + self.rho)/(1 + self.eta)).tolist())#target
        else:
            self.m.append((self.T + self.rho).tolist())

        self.n.append((self.softmax(self.phi_hat@x)).tolist())#推論結果
        self._x.append(np.ravel(x).tolist())#特徴量
        
        #更新
        if self.steps % self.batch_size == 0:
            
            #計算しやすい型に変換
            m_l = list(self.m)
            m_l = np.squeeze(np.array(m_l))
            n_l = list(self.n)
            n_l = np.squeeze(np.array(n_l))
            x_l = list(self._x)
            x_l = np.array(x_l)

            grad = ((m_l - n_l).T @x_l)/x_l.shape[0]# 1/n Σ(M-N)x

            self.phi_hat += self.eta * grad
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)

    def softmax(self,a):
        """softmax値を返す"""
        a_max = max(a)
        x = np.exp(a-a_max)#オーバーフロー対策しつつeの計算
        u = np.sum(x)
        return x/u

    def cross_entropy_error(self,t,y):
        """クロスエントロピー誤差を返す"""
        delta = 1e-7
        return -(t * np.log(y + delta)).sum()

    def get_theta(self) -> np.ndarray:
        """推定量を返す"""
        return self.theta_hat

    def get_theta_x(self) -> np.ndarray:
        """推定量_特徴量ありを返す"""    
        return self.theta_hat_x

    
    def get_entropy_arm(self) -> np.ndarray:
        if np.sum(self.n_x)==0:
            return 1
        return entropy(np.array(self.n_x), base=self.n_arms)

    def get_phi_x(self) -> np.ndarray:
        """疑似試行回数_特徴量ありを返す""" 
        return np.array(self.n[-1])    
    
    def get_target(self) -> np.ndarray:
        return np.array(self.m[-1])
        
           
                     
