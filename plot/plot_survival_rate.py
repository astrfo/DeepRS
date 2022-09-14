import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def mean_result(df):
    """1日単位で合計、sim数で平均"""
    df_result = pd.DataFrame()
    len_df = len(df.columns)
    n = len_df//day #あまり分は切り捨て
    for i in range(n):
        df_day = df.iloc[:, 0+day*i:day*(i+1)]
        df_day_reward = df_day.sum(axis=1)
        df_result=pd.concat([df_result, df_day_reward], axis=1)

    #合計から生存したかいなか出す
    survival = np.where(df_result >= survival_rate, 1, 0)
    #sim数で平均。独立の生存率
    print(df_result)
    survival_result_tmp = survival.mean(axis=0)
    #本来の生存率を出す(前の生存率にかける)
    survival_result = np.zeros(len(survival_result_tmp))
    survival_result[0] = survival_result_tmp[0]

    for j in range(len(survival_result_tmp)-1):
        survival_result[j+1] = survival_result[j] * survival_result_tmp[j+1]


    return survival_result, n
        

"""定数系"""
n_steps = 100000
name = ['Regional LinRS', 'LinTS', 'LinUCB', 'Greedy']#ここを書き換える

"""引数確認"""
args = sys.argv
if len(args) <= 1:
    print('Usage: python plot.py [path of directory where csv is stored]')
    sys.exit()

""""生存率定義"""
survival_rate = float(args[2])

"""1日の単位"""
day = int(args[3])

"""csvデータの取得"""
directory = os.listdir(args[1])
files = [f for f in directory if os.path.isfile(os.path.join(args[1], f))]
print(directory)
policy_names = [file_name[:-4].replace('_', ' ') for file_name in files]
f = [files[policy_names.index(i)] for i in name]
policy_names = [file_name[:-4].replace('_', ' ') for file_name in f]

result_list = []
for file_name in f:
    df = pd.read_csv(args[1] + '/' + file_name, index_col=0)
    df, n = mean_result(df)
    df = df.tolist()
    #dict_type = df.to_dict(orient='list')
    result_list.append(df)
#result_list = pd.DataFrame(result_list)
print(result_list)#生存率

"""結果データのプロット"""
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

time_now = datetime.datetime.now()
results_dir = 'png/survival_rate/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
for j, policy_name in enumerate(policy_names):
    cmap = plt.get_cmap("tab10")
    ax.plot(np.linspace(1, n, num=n),
                    result_list[j],
                    label=policy_name, linewidth=3, alpha=0.8)
ax.set_ylim([-0.05,1.05])

ax.set_xlabel('day', fontsize=23)
ax.set_ylabel('survival rate', fontsize=23)
leg = ax.legend(loc='lower right', fontsize=23)
plt.tick_params(labelsize=23)
ax.grid(alpha=0.8,color = "gray", linestyle="--")

fig.savefig(results_dir + 'survival_rate', bbox_inches='tight',
                pad_inches=0)

plt.clf()
