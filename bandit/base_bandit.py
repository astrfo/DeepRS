from typing import Tuple

import numpy as np


class BaseBandit(object):
    """ベースとなるバンディットクラス

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): ユーザーの特徴数
    """
    def __init__(self, n_arms: int, n_features: int) -> None:
        """クラスの初期化"""
        self.n_arms = n_arms
        self.n_features = n_features

    def initialize(self) -> None:
        """パラメータの初期化"""
        pass

    def pull(self, chosen_arm: int, x: np.matrix, n_sim: int) -> Tuple[int, float, int]:
        """選ばれた腕を引く"""
        pass
