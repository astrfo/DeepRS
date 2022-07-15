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
    n = len_df//day if len_df%day == 0 else (len_df//day) + 1
    for i in range(n):
        df_day = df.iloc[:, 0+day*i:day*(i+1)]
        df_day_reward = df_day.sum(axis=1)
        df_result=pd.concat([df_result, df_day_reward], axis=1)

        #print(df_day_reward)
        #print(df_result)
    #合計から生存したかいなか出す
    survival = np.where(df_result >= survival_rate, 1, 0)
    #sim数で平均。独立の生存率
    print(df_result)
    #print(survival)
    survival_result_tmp = survival.mean(axis=0)
    #print(survival_result_tmp)
    #本来の生存率を出す(前の生存率にかける)
    survival_result = np.zeros(len(survival_result_tmp))
    survival_result[0] = survival_result_tmp[0]

    for j in range(len(survival_result_tmp)-1):
        survival_result[j+1] = survival_result[j] * survival_result_tmp[j+1]

    #print(survival_result)

    return survival_result, n
        

"""定数系"""
n_steps = 100000

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
print(files)#時間が古い順にソートした方が良い
policy_names = [file_name[:-4].replace('_', ' ') for file_name in files]

result_list = []
for file_name in files:
    df = pd.read_csv(args[1] + '/' + file_name, index_col=0)
    df, n = mean_result(df)
    df = df.tolist()
    #dict_type = df.to_dict(orient='list')
    result_list.append(df)
#result_list = pd.DataFrame(result_list)
print(result_list)#生存率

#ここからしたやること
#シンプルにそれをplot、つまり下より簡単に

"""結果データのプロット"""
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

time_now = datetime.datetime.now()
results_dir = 'png/survival_rate/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
for j, policy_name in enumerate(policy_names):
    cmap = plt.get_cmap("tab10")
    ax.plot(np.linspace(1, n, num=n),
                    result_list[j],
                    label=policy_name, linewidth=1.5, alpha=0.8)
ax.set_ylim([0,1.1])

ax.set_xlabel('day', fontsize=14)
ax.set_ylabel('survival rate', fontsize=14)
leg = ax.legend(loc='upper left', fontsize=23)
plt.tick_params(labelsize=10)
ax.grid(axis='y')

fig.savefig(results_dir + 'survival_rate', bbox_inches='tight',
                pad_inches=0)

plt.clf()
