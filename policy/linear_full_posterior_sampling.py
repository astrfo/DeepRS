import math
from typing import Union, Tuple

import numpy as np
from scipy.stats import invgamma

from policy.base_policy import BaseContextualPolicy


class LinearTS(BaseContextualPolicy):
    """Linear Thompson Sampling with inverse gamma

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): ユーザーの特徴数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (int): 各腕が選択された回数
        lambda_prior (float): 正規行列のスケーリングパラメータ
        a0 (float): σ^2生成時のInverseGamma第1引数
        b0 (float): σ^2生成時のInverseGamma第2引数
    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1,
                 batch_size: int=1, lambda_prior: float=0.25,
                 a0: float=6, b0: float=6) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)

        self._lambda_proir = lambda_prior

        self.mu = [np.zeros(self.n_features + 1) for _ in range(self.n_arms)]
        self.cov = [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
                    for _ in range(self.n_arms)]
        self.precision = [self._lambda_proir * np.identity(self.n_features + 1)
                          for _ in range(self.n_arms)]

        self._a0 = a0
        self._b0 = b0

        self.a = [self._a0 for _ in range(self.n_arms)]
        self.b = [self._b0 for _ in range(self.n_arms)]

        self._mu = [np.zeros(self.n_features + 1) for _ in range(self.n_arms)]
        self._cov = [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
                    for _ in range(self.n_arms)]
        self._precision = [self._lambda_proir * np.identity(self.n_features + 1)
                          for _ in range(self.n_arms)]

        self._a = [self._a0 for _ in range(self.n_arms)]
        self._b = [self._b0 for _ in range(self.n_arms)]

        self.contexts = None
        self.rewards = None
        self.actions = []

        self.name = 'LinTS λ={}'.format(self._lambda_proir)

        self.theta_hat=np.zeros((self.n_arms,self.n_features))
        self.theta_hat_x=np.zeros(self.n_arms)

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()
        self.mu = [np.zeros(self.n_features + 1) for _ in range(self.n_arms)]

        self.cov = [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
                    for _ in range(self.n_arms)]

        self.precision = [self._lambda_proir * np.identity(self.n_features + 1)
                          for _ in range(self.n_arms)]

        self.a = [self._a0 for _ in range(self.n_arms)]
        self.b = [self._b0 for _ in range(self.n_arms)]

        self._mu = [np.zeros(self.n_features + 1) for _ in range(self.n_arms)]
        self._cov = [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
                    for _ in range(self.n_arms)]
        self._precision = [self._lambda_proir * np.identity(self.n_features + 1)
                          for _ in range(self.n_arms)]

        self._a = [self._a0 for _ in range(self.n_arms)]
        self._b = [self._b0 for _ in range(self.n_arms)]

        self.contexts = None
        self.rewards = None
        self.actions = []

        self.theta_hat=np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x=np.zeros(self.n_arms)

    def choose_arm(self, x: np.ndarray) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す"""
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            sigma2 = [self._b[i] * invgamma.rvs(self._a[i])
                      for i in range(self.n_arms)]
            try:
                beta = [np.random.multivariate_normal(self._mu[i], sigma2[i] * self._cov[i])
                        for i in range(self.n_arms)]
            except np.linalg.LinAlgError as e:
                print('except')
                d = self.n_features + 1
                beta = [np.random.multivariate_normal(np.zeros(d), np.identity(d))
                        for i in range(self.n_arms)]
            vals = [beta[i][: -1] @ x.T + beta[i][-1]
                    for i in range(self.n_arms)]#サンプリング
            self.theta_hat_x=[beta[i][:-1]@x.T for i in range(self.n_arms)]
            self.theta_hat=[beta[i][:-1] for i in range(self.n_arms)]
            
            result = np.argmax(vals)

        return result

    def _add(self, context, action, reward, intercept: int=1) -> None:
        """入力データをリストに追加"""
        if intercept:
            c = np.array(context[:])
            c = np.append(c, 1.0).reshape((1, self.n_features + 1))
        else:
            c = np.array(context[:]).reshape((1, self.n_features))

        if self.contexts is None:
            self.contexts = c
        else:
            self.contexts = np.vstack((self.contexts, c))

        r = np.zeros((1, self.n_arms))
        r[0, action] = reward
        if self.rewards is None:
            self.rewards = r
        else:
            self.rewards = np.vstack((self.rewards, r))

        self.actions.append(action)

    def _get_data(self, action) -> Tuple[np.matrix, np.matrix]:
        """現時刻で行った行動時のデータと報酬リストを取得"""
        ind = np.array([i for i in range(self.steps) if self.actions[i] == action])
        return self.contexts[ind, :], self.rewards[ind, action]

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータの更新"""
        super().update(x, chosen_arm, reward)

        self._add(x, chosen_arm, reward)

        x, y = self._get_data(chosen_arm)

        s = x.T @ x

        precision_a = s + self._lambda_proir * np.identity(self.n_features + 1)#Σaの元
        cov_a = np.linalg.inv(precision_a)#Σaを逆行列へ
        mu_a = cov_a @ x.T @ y#μ_a

        a_post = self._a0 + x.shape[0] / 2.0#a_0
        b_upd = 0.5 * (y.T @ y - mu_a.T @ precision_a @ mu_a)#b_0
        b_post = self._b0 + b_upd#b_0

        self.mu[chosen_arm] = mu_a
        self.cov[chosen_arm] = cov_a
        self.precision[chosen_arm] = precision_a
        self.a[chosen_arm] = a_post
        self.b[chosen_arm] = b_post

        if self.steps % self.batch_size == 0:
            self._mu, self._cov, self._precision = np.copy(self.mu), np.copy(self.cov), np.copy(self.precision)
            self._a, self._b = np.copy(self.a), np.copy(self.b)

    def get_theta(self, x: np.ndarray) -> np.ndarray:
        return (self.theta_hat)
    def get_theta_x(self, x: np.ndarray) -> np.ndarray:
        return (self.theta_hat_x)

