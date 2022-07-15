import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


"""定数系"""
n_steps = 100000

"""引数確認"""
args = sys.argv
if len(args) <= 1:
    print('Usage: python plot.py [path of directory where csv is stored]')
    sys.exit()

"""csvデータの取得"""
directory = os.listdir(args[1])
files = [f for f in directory if os.path.isfile(os.path.join(args[1], f))]
print(directory)
print(files)#時間が古い順にソートした方が良い
policy_names = [file_name[:-4].replace('_', ' ') for file_name in files]

result_list = []
for file_name in files:
    df = pd.read_csv(args[1] + '/' + file_name)
    dict_type = df.to_dict(orient='list')
    result_list.append(dict_type)
result_list = pd.DataFrame(result_list)

"""結果データのプロット"""
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

time_now = datetime.datetime.now()
results_dir = 'png/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)

for i, data_name in enumerate(
        ['rewards', 'regrets', 'accuracy', 'greedy_rate', 'errors']):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    for j, policy_name in enumerate(policy_names):
        cmap = plt.get_cmap("tab10")
        if data_name == 'greedy_rate' or data_name == 'accuracy':
            """通常ver"""
            # ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name,linewidth=1.5, alpha=0.8)
            """移動平均ver"""
            b = np.ones(10) / 10.0
            y3 = np.convolve(result_list.at[j, data_name], b,
                             mode='same')
            # ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), y3, label=policy_name+"moving_average", color=cmap(j), linewidth=1.5, alpha=0.8)
            ax.plot(np.linspace(1, n_steps, num=n_steps), y3,
                    label=policy_name, color=cmap(j), linewidth=1.5,
                    alpha=0.8)
            ax.set_ylim([0.2, 1.1])
        elif data_name == 'errors' or data_name == 'errors_greedy':
            """通常ver"""
            # ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name, linewidth=1.5, alpha=0.8)
            # ax.fill_between(x=np.linspace(1, self.n_steps, num=self.n_steps), y1=self.result_list.at[j, 'min_'+data_name], y2=self.result_list.at[j, 'max_'+data_name], alpha=0.1)
            """移動平均ver"""
            b = np.ones(10) / 10.0
            y3 = np.convolve(result_list.at[j, data_name], b,
                             mode='same')  # 移動平均
            ax.plot(np.linspace(1, n_steps, num=n_steps), y3,
                    label=policy_name + "moving_average",
                    color=cmap(j), linewidth=1.5, alpha=0.8)
            ax.set_ylim([0, 20.0])
        else:
            ax.plot(np.linspace(1, n_steps, num=n_steps),
                    result_list.at[j, data_name],
                    label=policy_name, linewidth=1.5, alpha=0.8)
            # ax.fill_between(x=np.linspace(1, self.n_steps, num=self.n_steps), y1=self.result_list.at[j, 'min_'+data_name], y2=self.result_list.at[j, 'max_'+data_name], alpha=0.1)
    ax.set_xlabel('steps', fontsize=14)
    ax.set_ylabel(data_name, fontsize=14)
    if data_name == 'greedy_rate' or data_name == 'accuracy':
        leg = ax.legend(loc='lower right', fontsize=23)
    else:
        leg = ax.legend(loc='upper left', fontsize=23)
    # leg.get_lines()[0].set_linewidth(3)
    # leg.get_lines()[1].set_linewidth(3)
    # leg.get_lines()[2].set_linewidth(3)
    # leg.get_lines()[3].set_linewidth(3)

    plt.tick_params(labelsize=10)
    ax.grid(axis='y')

    # plt.show() # サーバで実行する際にエラーになるのでコメントアウト
    # path = os.getcwd()#現在地
    # results_dir = os.path.join(path, 'png/{0:%Y%m%d%H%M}/'.format(time_now))#保存場所
    fig.savefig(results_dir + data_name, bbox_inches='tight',
                pad_inches=0)

plt.clf()
