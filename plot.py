import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot(args):
    csv_files = glob.glob('./log/' + args[1] + '/*.csv')
    for file in csv_files:
        name = os.path.splitext(os.path.basename(file))[0]
        data = pd.read_csv(file)
        plt.plot(data, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./log/' + args[1] + '/average_reward.png')
    plt.show()

if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 1:
        print('wrong number of arguments')
        sys.exit()
    plot(args)