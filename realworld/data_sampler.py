""" dataを加工するモジュール"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
#from absl import app
#from absl import flags
import numpy as np
import os
#import tensorflow as tf
import pandas as pd
import math

base_route = os.getcwd()
data_route = 'datasets'


def one_hot(df, cols):
  """特徴量をone_hotベクトルに直す
  Args:
    cols(pandas.core.indexes.base.Index):特徴(edible,cap-shape...)

  Returns:
    df(pandas.core.frame.DataFrame):特徴量をone-hotベクトルに変換したもの

  Example:
    >>> df = one_hot(df, df.columns)
    >>> print(df.iloc(0, ))

    edible_e               0
    edible_p               1
    cap-shape_b            0
    cap-shape_c            0
    cap-shape_f            0
    cap-shape_k            0
    cap-shape_s            0
    cap-shape_x            1
    .
    .
    .
  """
  for col in cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)
  return df
  

def sample_mushroom_data(num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):

    """mushroomデータセットの加工
    Args:
      num_contexts(int):データの数
      r_noeat(int):食べなかった時の報酬
      r_eat_safe(int):食用キノコを食べた時の報酬
      r_eat_poison_bad(int):毒キノコを食べた時の負の報酬
      r_eat_poison_good(int):毒キノコを食べた時の正の報酬
      prob_poison_bad(float):毒キノコを食べた時に負の報酬になる確率

    Returns:
      np.hstack((contexts, no_eat_reward, eat_reward)):加工後のデータセット[特徴ベクトル + 各行動の報酬]
      opt_vals(float):各stepで最適な行動を選択した時の報酬
      exp_rewards(float):各行動の報酬確率
    """

    path = os.path.join(base_route, data_route, 'mushroom.csv')

    df = pd.read_csv(path)
    df = one_hot(df, df.columns)#one-hotに変換

    #df=df.sample(n=100, replace=False)#特徴量を100種類に固定(重複なし)

    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)# num_contextsの数だけランダムでサンプルを抽出[num_contexts,119]
    

    contexts = df.iloc[ind, 2:]#特徴のみ持ってくる

    exp_rewards =[[r_noeat,r_noeat],[(r_eat_poison_bad + r_eat_poison_good)*prob_poison_bad,r_eat_safe]]
    # キノコの報酬を設定する
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    random_poison = np.random.choice(
          [r_eat_poison_bad, r_eat_poison_good],
          p=[prob_poison_bad, 1 - prob_poison_bad],
          size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    eat_reward = eat_reward.values.reshape((num_contexts, 1))

    # 最適な期待報酬と最適な行動を計算
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad) 
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
      r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

    if r_noeat > exp_eat_poison_reward:
        opt_actions = df.iloc[ind, 0]
    else:
        opt_actions = np.ones((num_contexts, 1))
    opt_vals = (opt_exp_reward.values, opt_actions.values)

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals, exp_rewards

def sample_stock_data(context_dim, num_actions, num_contexts,
                      sigma, shuffle_rows=True):
  """Samples linear bandit game from stock prices dataset.

  Args:
    file_name: Route of file containing the stock prices dataset.株価データセットを格納したファイルのルート。
    context_dim: Context dimension (i.e. vector with the price of each stock).特徴量の次元(つまり各株式の価格を持つベクトル)
    num_actions: Number of actions (different linear portfolio strategies).アクションの数（異なる線形のポートフォリオ戦略）
    num_contexts: Number of contexts to sample.サンプルとなる特徴の数
    sigma: Vector with additive noise levels for each action.各アクションの加算ノイズレベルを持つベクトル
    shuffle_rows: If True, rows from original dataset are shuffled.Trueの場合、元のデータセットの行がシャッフルされる

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).行をもつサンプル行(特徴、報酬1...報酬k)
    opt_vals: Vector of expected optimal (reward, action) for each context.各特徴に対する期待される最適（報酬、行動）のベクトル
  """

  file_name = os.path.join(base_route, data_route, 'raw_stock_contexts')

  #with tf.gfile.Open(file_name, 'r') as f:#file_nameをreadモードで開きfに代入、処理が終わったら閉じる
    #contexts = np.loadtxt(f, skiprows=1)#1行目以外を読み込んでcontextに代入
  contexts = np.loadtxt(file_name, skiprows=1)#1行目以外を読み込んでcontextに代入

  if shuffle_rows:
    np.random.shuffle(contexts)#シャッフル
  contexts = contexts[:num_contexts, :]#datasetを必要な分だけ抽出

  betas = np.random.uniform(-1, 1, (context_dim, num_actions))#-1から1の一様乱数、context_dim行/num_actions列分
  betas /= np.linalg.norm(betas, axis=0)#行動ごとにnorm計算(各要素の２乗和の平方根)

  mean_rewards = np.dot(contexts, betas)
  noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
  rewards = mean_rewards + noise

  opt_actions = np.argmax(mean_rewards, axis=1)
  opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
  return np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions)

def sample_jester_data(context_dim, num_actions, num_contexts,
                       shuffle_rows=True, shuffle_cols=False):
  """Jesterデータセットの加工

  Args:
    file_name: ファイルのルート
    context_dim(int): 特徴量の次元
    num_actions(int): 行動数
    num_contexts(int): データ数
    shuffle_rows: Trueの時は元データセットの行がシャッフルされる(行)
    shuffle_cols: Trueの時は特徴や行動がランダムにシャフルされる(列)

  Returns:
    dataset(float): 行を持つサンプル行列(特徴、評価値_1,...評価値_k)
    opt_vals(float, int): 各特徴に対する決定論的最適(報酬、行動)のベクトル
    exp_rewards(float):各データの報酬

  Raises:
    Wrong data dimensions.: 特徴量の次元数がdataset.shape[1]と一致しない場合
  """

  file_name = os.path.join(base_route, data_route, 'jester_data_40jokes_19181users.npy')

  #with tf.io.gfile.GFile(file_name, 'rb') as f:
    #dataset = np.load(f)
  dataset = np.load(file_name)
  if shuffle_cols:
    dataset = dataset[:, np.random.permutation(dataset.shape[1])]
  if shuffle_rows:
    np.random.shuffle(dataset)
  
  #データ種類100に固定しその中から必要分データを抽出
  """ind = np.random.choice(range(dataset.shape[0]), 100, replace=False)#ランダムに100抽出
  ind = np.random.choice(ind, num_contexts, replace=True)
  dataset = dataset[ind, :]"""
  dataset = dataset[:num_contexts, :]#num_contextsのぶんだけ抽出


  assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

  opt_actions = np.argmax(dataset[:, context_dim:], axis=1)#ユーザの評価値の中で1番大きい行動をopt_actionとする
  opt_rewards = np.array([dataset[i, context_dim + a]
                          for i, a in enumerate(opt_actions)])
  exp_rewards = dataset[:,context_dim:]#報酬箇所

  return dataset, (opt_rewards, opt_actions),exp_rewards

def sample_artificial_data(data_type, num_contexts,context_dim):
    path_1 = os.path.join(base_route, data_route, 'artificial_feature_data_' + data_type[11:] + '.csv')
    df_x = pd.read_csv(path_1)

    ind = np.random.choice(range(df_x.shape[0]), num_contexts, replace=True)# num_contextsの数だけランダムに抽出
    df = df_x.iloc[ind,:].values

    exp_rewards = df_x.iloc[ind, context_dim:].values#報酬箇所

    """最適期待報酬と最適行動を求める"""
    opt_actions = np.argmax(exp_rewards, axis=1)#ユーザの評価値の中で1番大きい行動をopt_actionとする
    opt_rewards = np.array([exp_rewards[i,a] for i, a in enumerate(opt_actions)])#iはindex,aは要素(行動)
    opt_values = (opt_rewards, opt_actions)

    return df, opt_values, exp_rewards

def sample_mixed_artificial_data(data_type, num_contexts,context_dim):
    path_1 = os.path.join(base_route, data_route, 'artificial_feature_data_mixed_' + data_type[17:] + '.csv')
    df_x = pd.read_csv(path_1)

    ind = np.random.choice(range(df_x.shape[0]), num_contexts, replace=True)# num_contextsの数だけランダムに抽出
    df = df_x.iloc[ind,:].values

    exp_rewards = df_x.iloc[ind, context_dim:].values#報酬箇所

    """最適期待報酬と最適行動を求める"""
    opt_actions = np.argmax(exp_rewards, axis=1)#ユーザの評価値の中で1番大きい行動をopt_actionとする
    opt_rewards = np.array([exp_rewards[i,a] for i, a in enumerate(opt_actions)])#iはindex,aは要素(行動)
    opt_values = (opt_rewards, opt_actions)

    return df, opt_values, exp_rewards




