import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


state_features = pd.read_csv("log/CartPole-v1/DQN_epi100/1/embed.csv", header=None)
q_values = pd.read_csv("log/CartPole-v1/DQN_epi100/1/selected_q_value.csv", header=None)

# t-SNE による次元圧縮 (128次元 → 2次元)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
state_embeddings = tsne.fit_transform(state_features)

# t-SNE 空間上に Q 値をプロット
plt.figure(figsize=(9, 6))
sc = plt.scatter(state_embeddings[:, 0], state_embeddings[:, 1], c=q_values, cmap="jet", alpha=0.7)
plt.colorbar(sc, label="Q-value")

# 一部のデータポイントに状態画像を追加 (ここでは仮の座標)
example_indices = [100, 300, 500, 1000]
for i in example_indices:
    plt.text(state_embeddings[i, 0], state_embeddings[i, 1], i, fontsize=12, ha="center", va="center")

plt.show()
