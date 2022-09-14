import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

# python3 plot/plot.py csv/指定ディレクトリ

"""定数系"""
n_steps = 100000
name = ['Regional LinRS', 'LinTS', 'LinUCB', 'Greedy']

"""引数確認"""
args = sys.argv
if len(args) <= 1:
    print('Usage: python plot.py [path of directory where csv is stored]')
    sys.exit()

"""csvデータの取得"""
directory = os.listdir(args[1])
files = [f for f in directory if os.path.isfile(os.path.join(args[1], f))]
print(directory)
policy_names = [file_name[:-4].replace('_', ' ') for file_name in files]
f = [files[policy_names.index(i)] for i in name]
policy_names = [file_name[:-4].replace('_', ' ') for file_name in f]

result_list = []
for file_name in f:
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
        ['rewards', 'regrets', 'accuracy', 'greedy_rate', 'errors', 'entropy_of_reliability']):
    fig = plt.figure(figsize=(12, 8))
    #fig = plt.figure(figsize=(8, 8))
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
                    label=policy_name, color=cmap(j), linewidth=3,
                    alpha=0.8)
            ax.set_ylim([0.2, 1.1])
        elif data_name == 'errors':
            """通常ver"""
            # ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name, linewidth=1.5, alpha=0.8)
            # ax.fill_between(x=np.linspace(1, self.n_steps, num=self.n_steps), y1=self.result_list.at[j, 'min_'+data_name], y2=self.result_list.at[j, 'max_'+data_name], alpha=0.1)
            """移動平均ver"""
            b = np.ones(30) / 30.0
            y3 = np.convolve(result_list.at[j, data_name], b,
                             mode='same')  # 移動平均
            ax.plot(np.linspace(1, n_steps, num=n_steps), y3,
                    label=policy_name,
                    color=cmap(j), linewidth=3, alpha=0.8)
            ax.set_ylim([-5.0, 5.0])
        elif data_name == 'entropy_of_reliability':
            if 'LinRS' in policy_name:
                ax.plot(np.linspace(1, n_steps, num=n_steps), result_list.at[j, data_name], label=policy_name, linewidth=3, alpha=0.8)
                ax.set_ylim([0,1.1])

        elif data_name == 'regrets':
            """ 白黒plotの時
            if policy_name == 'Regional LinRS':
                    ax.plot(np.linspace(1, n_steps, num=n_steps),
                    result_list.at[j, data_name],
                    label=policy_name, color="k",linewidth=3, alpha=0.8)
            elif policy_name == 'LinTS':
                    ax.plot(np.linspace(1, n_steps, num=n_steps),
                    result_list.at[j, data_name],
                    label=policy_name, color="k",linestyle = "dashed",linewidth=3, alpha=0.8)  
            elif policy_name == 'LinUCB':
                    ax.plot(np.linspace(1, n_steps, num=n_steps),
                    result_list.at[j, data_name],
                    label=policy_name, color="k",linestyle = "dashdot",linewidth=3, alpha=0.8)  
            elif policy_name == 'Greedy':
                    ax.plot(np.linspace(1, n_steps, num=n_steps),
                    result_list.at[j, data_name],
                    label=policy_name, color="k",linestyle = "dotted",linewidth=3, alpha=0.8)"""
            ax.plot(np.linspace(1, n_steps, num=n_steps),
                    result_list.at[j, data_name],
                    label=policy_name, linewidth=3, alpha=0.8)
            ax.set_ylim([0,4000])
        elif data_name == 'rewards':
                ax.plot(np.linspace(1, n_steps, num=n_steps),
                result_list.at[j, data_name],
                label=policy_name, linewidth=3, alpha=0.8)
                ax.set_ylim([0,80000])
        else:
            ax.plot(np.linspace(1, n_steps, num=n_steps),
                    result_list.at[j, data_name],
                    label=policy_name, linewidth=3, alpha=0.8)
            # ax.fill_between(x=np.linspace(1, self.n_steps, num=self.n_steps), y1=self.result_list.at[j, 'min_'+data_name], y2=self.result_list.at[j, 'max_'+data_name], alpha=0.1)

    ax.set_xlabel('step', fontsize=23)
    ax.xaxis.offsetText.set_fontsize(23)
    #ax.ticklabel_format(style="sci", axis="x", scilimits=(4,4))   # 10^3単位の指数で表示
    if data_name == 'rewards':
        ax.set_ylabel('reward',fontsize=23)
        ax.yaxis.offsetText.set_fontsize(23)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(4,4))   # 10^3単位の指数で表示
    elif data_name == 'regrets':
        ax.set_ylabel('regret',fontsize=23)
    elif data_name == 'greedy_rate':
        ax.set_ylabel('greedy rate',fontsize=23)
    elif data_name == 'errors':
        ax.set_ylabel('mean percentage error',fontsize=23)
    elif data_name == 'entropy_of_reliability':
        ax.set_ylabel('entropy of reliability',fontsize=23)
    else:
        ax.set_ylabel(data_name,fontsize=23)
    #ax.spines["top"].set_linewidth(2)
    #ax.spines["left"].set_linewidth(2)
    #ax.spines["bottom"].set_linewidth(2)
    #ax.spines["right"].set_linewidth(2)
    if data_name == 'greedy_rate' or data_name == 'accuracy':
        leg = ax.legend(loc='lower right', fontsize=23)
    if data_name == 'errors':
        leg = ax.legend(loc='lower right', fontsize=23)
    else:
        leg = ax.legend(loc='upper left', fontsize=23)

    plt.tick_params(labelsize=23)
    #ax.grid(alpha=0.8,color = "gray", linestyle="--")

    fig.savefig(results_dir + data_name, bbox_inches='tight',
                pad_inches=0)

plt.clf()
