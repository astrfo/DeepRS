from typing import Union

import numpy as np

from policy.base_policy import BaseContextualPolicy


class Uniform(BaseContextualPolicy):
    """全ての選択肢から一様ランダムに選択するアルゴリズム

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): ユーザーの特徴数
    """
    def __init__(self, n_arms: int, n_features: int) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features)
        self.name = 'Uniform'

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()

    def choose_arm(self, x: np.ndarray,step) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す"""
        return np.random.randint(self.n_arms)

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータの更新"""
        super().update(x, chosen_arm, reward)
