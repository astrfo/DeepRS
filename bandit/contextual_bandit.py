# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class ContextualBandit(object):
  """d次元の特徴量とk本の腕を持つ文脈付きバンディットを実装"""

  def __init__(self, n_features, n_arms, n_contexts, data_type):
    """クラスの初期化

    Args:
      context_dim(int): 特徴量の次元数
      num_actions(int): 行動数
      num_contexts(int): データ数
      datatype(str): データセット名
    """

    self._context_dim = n_features
    self._num_actions = n_arms
    self._num_contexts = n_contexts
    self._datatype = data_type

  def feed_data(self, data):
    """1sim毎の初期化

    Args:
      data: 特徴量ベクトル + 報酬が 1sim分入ったデータ 
    Raises:
      ValueError: データの次元数がdata.shape[1]と一致していない場合
    """

    if data.shape[1] != self.context_dim + self.num_actions:
      raise ValueError('Data dimensions do not match.')

    self._number_contexts = data.shape[0]
    self.data = data
    self.order = range(self.number_contexts)

  def reset(self):
    """シャッフルする"""
    self.order = np.random.permutation(self.number_contexts)

  def context(self, number):
    """ 指定されたstepの文脈を返す
    Args:
      number(int):指定のstep数
    Returns:
      self.data[self.order[number]][:self.context_dim]: 指定されたstepの特徴量のみ返す
    """

    return self.data[self.order[number]][:self.context_dim]

  def reward(self, number, action):
    """指定されたstep数の特徴量・行動に対する報酬を返す
    Args:
      number(int):指定のstep数
      action(int):行動
    Returns:
      self.data[self.order[number]][self.context_dim + action]:行動に対する報酬を返す
    """
    return self.data[self.order[number]][self.context_dim + action]

  def optimal(self, number):
    """指定stepにおける最適な行動を返す
    Args:
      number(int):指定のstep数
    Returns:
      np.argmax(self.data[self.order[number]][self.context_dim:]):指定のstepにおける最適行動
    """
    return np.argmax(self.data[self.order[number]][self.context_dim:])

  @property
  def context_dim(self):
    """特徴量の次元数を返す"""
    return self._context_dim

  @property
  def num_actions(self):
    """行動数を返す"""
    return self._num_actions

  @property
  def number_contexts(self):
    """特徴の数"""
    return self._number_contexts
