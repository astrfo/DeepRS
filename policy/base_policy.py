from typing import Union

import numpy as np


class BaseContextualPolicy(object):
    """文脈付きアルゴリズムのベースとなるクラス

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): バッチサイズ、パラメータ更新の間隔
        steps (int): 現在のstep数としてupdate関数によって更新される値
        counts (list[int]): 各腕が選択された回数
    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1) -> None:
        """クラスの初期化"""
        self.n_arms = n_arms
        self.n_features = n_features
        self.warmup = warmup
        self.batch_size = batch_size
        self.steps = 0
        self.counts = np.zeros(self.n_arms)
        self.name = None

    def initialize(self) -> None:
        """パラメータの初期化"""
        self.steps = 0
        self.counts = np.zeros(self.n_arms)

    def choose_arm(self, x: np.matrix) -> int:
        """選択肢にある腕を選択する"""
        pass

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータの更新"""
        self.steps += 1
        self.counts[chosen_arm] += 1
