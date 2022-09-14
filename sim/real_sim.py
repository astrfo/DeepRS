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

# coding: shift_jis

"""実世界データを用いたシミュレーションをおこなうモジュール"""
import os
import time
from typing import List

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from policy.base_policy import BaseContextualPolicy
from bandit.base_bandit import BaseBandit

from bandit.contextual_bandit import ContextualBandit
from realworld.setup_context import ContextData

#import codecs

class ContextualBanditSimulator(object):
  """指定されたアルゴリズムで文脈付きバンディットを実行する

  Args:
    context_dim(int): 特徴ベクトルの次元数
    num_actions(int): 選択肢の数
    dataset(int, float): 特徴量 + 各行動の報酬が入ったデータセット
    algos(list of str): 使用するアルゴリズムのリスト
  """
  def __init__(self, policy_list: List[BaseContextualPolicy],
                 bandit: BaseBandit, n_sims: int, n_steps: int, n_arms: int,
                 n_features: int, data_type) -> None:
    """クラスの初期化
    Args:
      policy_list(list of str) : 方策リスト
      bandit : bandit環境
      n_sims(int) : sim数
      n_steps(int) : step数
      n_arms(int) : アーム数(行動数)
      n_features(int) : 特徴量の次元数
      data_type(str) : dataset名
    """

    self.policy_list = policy_list
    self.bandit = bandit
    self.n_sims = n_sims
    self.n_steps = n_steps
    self.n_arms = n_arms
    self.n_features = n_features

    self.data_type = data_type

    self.policy_name = []
    self.result_list = []
    self.elapsed_time_tmp = np.zeros(len(policy_list))

  def generate_feature_data(self) -> None:
    """加工したdataset・最適行動を取った時の報酬・最適行動・報酬期待値を返す"""
    dataset, opt_rewards, opt_actions, exp_rewards = ContextData.sample_data(self.data_type, self.n_steps)
    return dataset, opt_rewards, opt_actions,exp_rewards

  def processing_data(self, rewards,regrets,accuracy,successes,errors,entropy_of_reliability,times):
    """データを平均,最小値,最大値に処理して返す"""
    result_data = np.concatenate(([rewards], [regrets], [accuracy],[successes],[errors],[entropy_of_reliability],[times]),axis=0)
    min_data = result_data.min(axis=1)
    max_data = result_data.max(axis=1)
    ave_data = np.sum(result_data, axis=1) / self.n_sims

    return ave_data[0], ave_data[1], ave_data[2],ave_data[3],ave_data[4],ave_data[5],ave_data[6],min_data, max_data

  def run_sim(self):
    """方策ごとにシミュレーションを回す"""
    cmab = self.bandit
    i = 0
    time_now = datetime.datetime.now()
    results_dir = 'csv/{0:%Y%m%d%H%M}/'.format(time_now)
    os.makedirs(results_dir, exist_ok=True)

    results_dir = 'csv_reward_count/{0:%Y%m%d%H%M}/'.format(time_now)
    os.makedirs(results_dir, exist_ok=True)

    results_dir = 'csv_mse/{0:%Y%m%d%H%M}/'.format(time_now)
    os.makedirs(results_dir, exist_ok=True)
    """方策ごとに実行"""
    for policy in self.policy_list:
        print(policy.name)
        self.policy_name.append(policy.name)
        """結果表示に用いる変数の初期化"""
        rewards = np.zeros((self.n_sims, self.n_steps), dtype=float)
        reward_csv = np.zeros((self.n_sims, self.n_steps), dtype=float)
        regrets = np.zeros((self.n_sims, self.n_steps), dtype=float)
        entropy_of_reliability = np.zeros((self.n_sims, self.n_steps), dtype=float)
        errors = np.zeros((self.n_sims, self.n_steps), dtype=float)
        mse_arm = np.zeros((self.n_sims, self.n_steps,self.n_arms), dtype=float)
        successes = np.zeros((self.n_sims, self.n_steps), dtype=int)
        accuracy = np.zeros((self.n_sims, self.n_steps), dtype=int)
        times = np.zeros((self.n_sims, self.n_steps), dtype=float)
        elapsed_time = 0.0

        start_tmp = time.time()
        """シミュレーション開始"""
        for sim in np.arange(self.n_sims):
            print('{} : '.format(sim), end='')
            dataset, opt_rewards, opt_actions,exp_rewards = self.generate_feature_data()#加工したデータセットを持ってくる
            cmab.feed_data(dataset)#1シミュレーションごとにデータの中から必要なぶんだけ取り出し(初期化も兼ねてる)
            policy.initialize()
            """初期化"""
            elapsed_time =0.0
            sum_reward, sum_regret = 0.0, 0.0
            mse_tmp = np.zeros((self.n_steps, self.n_arms), dtype=float)

            """step開始"""
            for step in np.arange(self.n_steps):
                start=time.time()
                x = cmab.context(step)#特徴量のみ持ってくる
                chosen_arm = policy.choose_arm(x)
                reward = cmab.reward(step, chosen_arm)

                regret = opt_rewards[step] - reward
                theta_hat=policy.get_theta_x()#推定量theta_hat@x持ってくる

                if "LinRS" in policy.name:
                  reliability = policy.get_entropy_arm() #底が腕の本数のエントロピーの計算
                  entropy_of_reliability[sim,step] += reliability

                success_acc = 1 if chosen_arm == opt_actions[step] else 0#真のgreedy(Accuracy)
                success_greedy = 1 if chosen_arm == np.argmax(theta_hat) else 0#主観greedy

                if self.data_type == 'mushroom':
                  theta_error = mean_squared_error([row[opt_actions[step]] for row in exp_rewards], theta_hat, squared=False) #True:MSE, False:RMSE
                  #theta_error = mean_absolute_error([row[opt_actions[step]] for row in exp_rewards], theta_hat)
                elif self.data_type == 'jester':
                  theta_error = mean_squared_error(exp_rewards[step], theta_hat, squared=False)
                  #theta_error = mean_absolute_error(exp_rewards[step], theta_hat)
                elif self.data_type.startswith('artificial'):
                  theta_error = mean_squared_error(exp_rewards[step], theta_hat, squared=False)
                  #theta_error = mean_absolute_error(exp_rewards[step], theta_hat)
                elif self.data_type.startswith('mixed_artificial'):
                  #theta_error = mean_squared_error(exp_rewards[step], theta_hat, squared=False)
                  #theta_error = mean_absolute_error(exp_rewards[step], theta_hat)
                  theta_error = np.average((exp_rewards[step]-theta_hat)/exp_rewards[step]) #MPE
                  old_sort_mse= mean_squared_error([exp_rewards[step]], [theta_hat], multioutput='raw_values') #各腕ごとの MSE
                  l = np.argsort(exp_rewards[step])[::-1]
                  m = 0
                  for k in l:
                    mse_tmp[step,k] += old_sort_mse[m]
                    m += 1
                else:
                  print("The error for data_type is not calculated!!")

                policy.update(x, chosen_arm, reward)#方策アップデート

                sum_reward += reward
                sum_regret += regret
                
                rewards[sim, step] += sum_reward
                reward_csv[sim,step] += reward
                regrets[sim, step] += sum_regret
                errors[sim,step] += theta_error
                accuracy[sim, step] += success_acc
                successes[sim, step] += success_greedy
                elapsed_time += time.time()-start
                times[sim,step] +=elapsed_time

            print('{}'.format(regrets[sim, -1]))
            mse_arm[sim,:,:]+= mse_tmp

        self.elapsed_time_tmp[i] = time.time() - start_tmp
        i += 1
        #print("経過時間 : {}".format(elapsed_time_tmp))
        mse = np.mean(mse_arm,axis=0)
        print("mse:",mse)

        ave_rewards, ave_regrets, accuracy,greedy_rate,errors,entropy_of_reliability,ave_times,min_data,max_data = \
           self.processing_data(rewards, regrets, accuracy,successes,errors,entropy_of_reliability,times)
        data = [ave_rewards, ave_regrets, accuracy,greedy_rate,errors,entropy_of_reliability,ave_times,min_data, max_data]

        data_dic = \
            {'rewards': data[0], 'regrets': data[1], 'accuracy': data[2],'greedy_rate': data[3],'errors':data[4],'entropy_of_reliability':data[5],'times':data[6],
             'min_rewards': data[7][0], 'min_regrets': data[7][1],'min_accuracy': data[7][2],'min_greedy_rate': data[7][3],'min_errors':data[7][4],'min_entropy_of_reliability':data[7][5],'min_times':data[7][6],
             'max_rewards': data[8][0],'max_regrets': data[8][1], 'max_accuracy': data[8][2],'max_greedy_rate': data[8][3],'max_errors':data[8][4],'max_entropy_of_reliability':data[8][5],'max_times':data[8][6]}
        data_dic_pd = pd.DataFrame(data_dic)
        data_dic_pd.to_csv('csv/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, policy.name.replace(' ', '_')))

        reward_csv_pd = pd.DataFrame(reward_csv)
        reward_csv_pd.to_csv('csv_reward_count/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, policy.name.replace(' ', '_')))
        print('rewards: {0}\nregrets: {1}\naccuracy: {2}\ngreedy_rate: {3}\nerrors:{4}\ntimes: {5}\n'
              'min_rewards: {6}\nmin_regrets: {7}\nmin_accuracy: {8}\nmin_greedy_rate: {9}\nmin_errors: {10}\nmin_times:{11}\n'
              'max_rewards: {12}\nmax_regrets: {13}\nmax_accuracy: {14}\nmax_greedy_rate: {15}\nmax_errors:{16}\nmax_times:{17}'
              .format(data_dic['rewards'], data_dic['regrets'],data_dic['accuracy'],data_dic['greedy_rate'],data_dic['errors'] ,data_dic['times'],
                      data_dic['min_rewards'],data_dic['min_regrets'], data_dic['min_accuracy'],data_dic['min_greedy_rate'],data_dic['min_errors'],data_dic['min_times'],
                      data_dic['max_rewards'], data_dic['max_regrets'],data_dic['max_accuracy'],data_dic['max_greedy_rate'],data_dic['max_errors'],data_dic['max_times']))
        mse_pd = pd.DataFrame(mse)
        mse_pd.to_csv('csv_mse/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, policy.name.replace(' ', '_')))


        self.result_list.append(data_dic)
        self.dy = np.gradient(data[1])
        print('傾き: ', self.dy)
    sim_time = self.elapsed_time_tmp / self.n_sims
    print("1 sim time:", sim_time)
    self.result_list = pd.DataFrame(self.result_list)
    

  def run(self) -> None:
        """一連のシミュレーションを実行"""
        self.run_sim() # 結果のlog採取
        #self.plots() # plot は別

  def plots(self) -> None:
        """結果データのプロット"""
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.ymargin'] = 0
        mpl.rcParams['font.family'] = 'Noto Sans CJK JP'

        time_now = datetime.datetime.now()
        results_dir = 'png/{0:%Y%m%d%H%M}/'.format(time_now)
        os.makedirs(results_dir, exist_ok=True)

        for i, data_name in enumerate(['rewards', 'regrets','accuracy', 'greedy_rate','errors', 'entropy_of_reliability']):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            for j, policy_name in enumerate(self.policy_name):
                cmap = plt.get_cmap("tab10")
                if data_name == 'greedy_rate' or data_name =='accuracy':
                    """通常ver"""
                    #ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name,linewidth=1.5, alpha=0.8)
                    """移動平均ver"""
                    b=np.ones(30)/30.0
                    y3=np.convolve(self.result_list.at[j, data_name], b, mode='same')
                    ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), y3, label=policy_name, color=cmap(j), linewidth=1.5, alpha=0.8)
                    ax.set_ylim([0.2, 1.1])
                elif data_name =='errors':
                    """通常ver"""
                    #ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name, linewidth=1.5, alpha=0.8)
                    #ax.fill_between(x=np.linspace(1, self.n_steps, num=self.n_steps), y1=self.result_list.at[j, 'min_'+data_name], y2=self.result_list.at[j, 'max_'+data_name], alpha=0.1)
                    """移動平均ver"""
                    b=np.ones(10)/10.0
                    y3=np.convolve(self.result_list.at[j, data_name], b, mode='same')#移動平均
                    ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), y3, label=policy_name+"moving_average", color=cmap(j), linewidth=1.5, alpha=0.8)
                    ax.set_ylim([0,1.0])
                elif data_name == 'entropy_of_reliability':
                    if 'LinRS' in policy_name:
                      ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name, linewidth=1.5, alpha=0.8)
                      ax.set_ylim([0,1.1])
                else:
                    ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name, linewidth=1.5, alpha=0.8)
                    #ax.fill_between(x=np.linspace(1, self.n_steps, num=self.n_steps), y1=self.result_list.at[j, 'min_'+data_name], y2=self.result_list.at[j, 'max_'+data_name], alpha=0.1)
            ax.set_xlabel('step',fontsize=23)
            if data_name == 'rewards':
              ax.set_ylabel('reward',fontsize=23)
            elif data_name == 'regrets':
              ax.set_ylabel('regret',fontsize=23)
            elif data_name == 'greedy_rate':
              ax.set_ylabel('greedy rate',fontsize=23)
            elif data_name == 'errors':
              ax.set_ylabel('error',fontsize=23)
            elif data_name == 'entropy_of_reliability':
              ax.set_ylabel('entropy of reliability',fontsize=23)
            else:
              ax.set_ylabel(data_name,fontsize=23)
            ax.spines["top"].set_linewidth(2)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["right"].set_linewidth(2)

            if data_name == 'greedy_rate' or data_name =='accuracy':
              leg=ax.legend(loc='lower right', fontsize=23)
            elif data_name == 'entropy_of_reliability':
              leg=ax.legend(loc='lower left', fontsize=23)
            else:
              leg=ax.legend(loc='upper left', fontsize=23)

            plt.tick_params(labelsize=10)
            ax.grid(axis='y')

            #plt.show()
            #path = os.getcwd()#現在地
            #results_dir = os.path.join(path, 'png/{0:%Y%m%d%H%M}/'.format(time_now))#保存場所
            fig.savefig(results_dir + data_name, bbox_inches='tight', pad_inches=0)



        #plt.clf()
