import pandas as pd
import matplotlib.pyplot as plt

# 3つのCSVファイルのパス
files = [
    "average1.csv", 
    "average2.csv", 
    "average3.csv"
]

# x軸のラベル
x_labels = [1, 2, 3]


# 各ファイルの全平均を計算
averages = [pd.read_csv(file, header=None).values.mean() for file in files]
plt.figure(figsize=(9, 6))
plt.plot(x_labels, averages, marker='o', label="Average Reward")
plt.xlabel("parameter xxx")
plt.ylabel("Average Reward")
plt.legend()
plt.savefig("average_param_plot.png")
plt.close()

# 各ファイルの最後から200episode分の平均を計算
last200_averages = [pd.read_csv(file, header=None).values[-200:].mean() for file in files]
plt.figure(figsize=(9, 6))
plt.plot(x_labels, last200_averages, marker='o', label="Last 200 Episode Average Reward")
plt.xlabel("parameter xxx")
plt.ylabel("Last200 Average Reward")
plt.legend()
plt.savefig("last200_average_param_plot.png")
plt.close()
