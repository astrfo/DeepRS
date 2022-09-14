import math
from typing import Union

import numpy as np
from scipy.stats import invgamma
from scipy.linalg import lu

from policy.base_policy import BaseContextualPolicy


class LinearTSfixed(BaseContextualPolicy):
    """Linear Thompson Sampling with inverse gamma

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): バッチサイズ
        counts (list[int]): 各腕が選択された回数
        lambda_prior (float): 正規行列のスケーリングパラメータ
        a0 (float): σ^2生成時のInverseGamma第1引数
        b0 (float): σ^2生成時のInverseGamma第2引数
    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1,
                 batch_size: int=1, lambda_prior: float=0.25,
                 a0: float=6.0, b0: float=6.0) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)

        self._lambda_proir = lambda_prior

        self.mu = np.array(
            [np.zeros(self.n_features + 1) for _ in range(self.n_arms)])
        self.cov = np.array(
            [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self.precision = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])

        self._a0 = a0
        self._b0 = b0

        self.ig_a = np.array([self._a0 for _ in range(self.n_arms)])
        self.ig_b = np.array([self._b0 for _ in range(self.n_arms)])

        self.A = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.n_features + 1))
        self.c = np.zeros(self.n_arms)

        self._mu = np.array(
            [np.zeros(self.n_features + 1) for _ in range(self.n_arms)])
        self._cov = np.array(
            [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self._precision = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])

        self._ig_a = np.array([self._a0 for _ in range(self.n_arms)])
        self._ig_b = np.array([self._b0 for _ in range(self.n_arms)])

        self._A = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features + 1))
        self._c = np.zeros(self.n_arms)

        #self.name = 'LinTS λ={}'.format(self._lambda_proir)
        self.name = 'LinTS'
        self.theta_hat_x=np.zeros(self.n_arms)

        self.flag = True

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()

        self.mu = np.array(
            [np.zeros(self.n_features + 1) for _ in range(self.n_arms)])
        self.cov = np.array(
            [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self.precision = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])

        self.ig_a = np.array([self._a0 for _ in range(self.n_arms)])
        self.ig_b = np.array([self._b0 for _ in range(self.n_arms)])

        self.A = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.n_features + 1))
        self.c = np.zeros(self.n_arms)

        self._mu = np.array(
            [np.zeros(self.n_features + 1) for _ in range(self.n_arms)])
        self._cov = np.array(
            [(1.0 / self._lambda_proir) * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self._precision = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])

        self._ig_a = np.array([self._a0 for _ in range(self.n_arms)])
        self._ig_b = np.array([self._b0 for _ in range(self.n_arms)])

        self._A = np.array(
            [self._lambda_proir * np.identity(self.n_features + 1)
             for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features + 1))
        self._c = np.zeros(self.n_arms)
        self.theta_hat_x=np.zeros(self.n_arms)

        self.flag = True

    def choose_arm(self, x: np.ndarray) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す"""
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            sigma2 = [self._ig_b[i] * invgamma.rvs(self._ig_a[i])
                      for i in range(self.n_arms)]
            try:
                #beta = [np.random.multivariate_normal(self._mu[i], sigma2[i] * self._cov[i])
                        #for i in range(self.n_arms)]
                #L = [np.linalg.cholesky(sigma2[i] * self._cov[i]) for i in range(self.n_arms)] #コレスキー分解
                #z = [np.random.standard_normal(len(sigma2[i] * self._cov[i])) for i in range(self.n_arms)] #標準正規乱数ベクトル
                #beta = [L[i] @ z[i] + self._mu[i] for i in range(self.n_arms)]
                L = [np.linalg.cholesky(self._cov[i]) for i in range(self.n_arms)] # コレスキー分解 この時点でsigma2でスケーリングするとエラーが増える印象?なので_covだけに
                z = [np.random.standard_normal(len(self._cov[i])) for i in range(self.n_arms)] # 標準正規乱数ベクトル 上に合わせて_covだけに
                beta = [sigma2[i] * L[i] @ z[i] + self._mu[i] for i in range(self.n_arms)]  # 最後にスケーリング
            except np.linalg.LinAlgError as e:
                """
                コレスキー分解等出来ない場合は従来通り多重正規分布をそのまま生成
                その分時間がかかる
                ※RuntimeError で multivariate_normal の共分散に関するエラーがでたりするが実行には一応問題なさそう
                """
                if self.gen_f():
                    print('except')
                #d = self.n_features + 1
                #beta = [np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(self.n_arms)]
                try:
                    beta = [np.random.multivariate_normal(self._mu[i], sigma2[i] * self._cov[i]) for i in range(self.n_arms)]
                except np.linalg.LinAlgError as e:
                    d = self.n_features + 1
                    beta = [np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(self.n_arms)]
            vals = [beta[i][: -1] @ x.T + beta[i][-1]
                    for i in range(self.n_arms)]
            self.theta_hat_x=[beta[i][: -1] @ x.T for i in range(self.n_arms)]

            result = np.argmax(vals)

        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータの更新"""
        super().update(x, chosen_arm, reward)

        x = np.append(x, 1.0).reshape((1, self.n_features + 1))
        #A,bの更新
        precision_a = self.A[chosen_arm] + x.T @ x  # f*f
        cov_a = np.linalg.inv(precision_a)
        precision_b = self.b[chosen_arm] + x * reward  # 1*f
        mu_a = precision_b @ cov_a  # 1*f
        #alpha,betaの更新
        a_post = self._a0 + self.counts[chosen_arm] / 2.0
        precision_c = self.c[chosen_arm] + reward * reward
        b_upd = 0.5 * (precision_c - mu_a @ precision_a @ mu_a.T)
        b_post = self._b0 + b_upd

        self.mu[chosen_arm] = mu_a[0]
        self.cov[chosen_arm] = cov_a
        self.A[chosen_arm] = precision_a
        self.b[chosen_arm] = precision_b
        self.c[chosen_arm] = precision_c
        self.ig_a[chosen_arm] = a_post
        self.ig_b[chosen_arm] = b_post[0]

        if self.steps % self.batch_size == 0:
            self._mu, self._cov = np.copy(self.mu), np.copy(self.cov)
            self._A, self._b, self._c = np.copy(self.A), np.copy(self.b), np.copy(self.c)
            self._ig_a, self._ig_b = np.copy(self.ig_a), np.copy(self.ig_b)

    def gen_f(self):
        """行列分解に関してのエラーが1度でも起きたかどうかのフラグを返す"""
        if self.flag:
            self.flag = False
            return True
        else:
            return False

    def f_theta_hat(self, x: np.ndarray) -> np.ndarray:
        return (self.theta_hat @ x)[1]
    def get_theta_x(self) -> np.ndarray:
        return self.theta_hat_x
