import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
任意のフォルダの全日付フォルダを平均するプログラム
python total_avg_plot.py sRSRS_sim10_alpha0.01_gamma0.9

sRSRS_sim10_alpha0.01_gamma0.9
|
├── 202301111729
└── 202301111731
↓↓↓
sRSRS_sim10_alpha0.01_gamma0.9
|
├── 202301111729
├── 202301111731
└── "total_average"
"""


def make_folder(args):
    log_path = f'./log/{args[1]}'
    total_avg_path = f'{log_path}/total_average'
    os.makedirs(total_avg_path, exist_ok=True)
    metric_list = [
        'epi100_reward',
        'fall_hole',
        'goal_step',
        'reward',
    ]
    for metric in metric_list:
        metric_csv_files = glob.glob(f'{log_path}/**/**/{metric}.csv')
        data_list = []
        for file in metric_csv_files:
            data = pd.read_csv(file, header=None)
            data_list.append(data)
        df = pd.concat(data_list, axis=1)
        mean_df = np.nanmean(df, axis=1)
        np.savetxt(f'{total_avg_path}/total_average_{metric}.csv', mean_df, delimiter=',')


def total_average_plot(args):
    total_avg_path = f'./log/{args[1]}/total_average'
    csv_files = glob.glob(f'{total_avg_path}/*.csv')
    for file in csv_files:
        fig = plt.figure(figsize=(12, 8))
        name = os.path.splitext(os.path.basename(file))[0]
        data = pd.read_csv(file)
        plt.plot(data, label=name)
        plt.title(name)
        plt.xlabel('Episode')
        plt.xlim(-1, len(data)+1)
        plt.savefig(f'{total_avg_path}/{name}.jpg')
        plt.close()


if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 1:
        print('wrong number of arguments')
        sys.exit()
    make_folder(args)
    total_average_plot(args)