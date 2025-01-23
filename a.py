import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
dqn_data = pd.read_csv('log/CartPole-v1/DQN_sim10/202410250818/average/average_reward.csv')
rs2_data = pd.read_csv('log/CartPole-v1/RSRSAlephQEpsRASChoiceDQN_sim10_epi1011_warmup50_zeta0.01_aleph_G0/202501111058/average/average_reward.csv')

# プロットの設定
plt.figure(figsize=(9, 6))

# DQNデータのプロット

# RS^2データのプロット
plt.plot(dqn_data, label='DQN')
plt.plot(rs2_data, label=r"$\mathrm{RS}^2$")

# 凡例の追加
plt.legend()

# タイトルと軸ラベルの設定
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Average Reward', fontsize=18)


# プロットの表示
plt.show()
