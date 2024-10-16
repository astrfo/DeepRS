# DeepRS

## 使い方

GitHubからクローンして，`requirements.txt`でライブラリをインストールしてください．

```zsh
git clone https://github.com/astrfo/DeepRS.git
cd DeepRS
pip install -r requirements.txt
```

## パラメータ，環境の設定

必要に応じて`main.py`にあるパラメータと環境等を変更してください。

```python
"""
algo: 
DQN or DDQN or DuelingDQN or DuelingDDQN or
RSRSDQN or RSRSDDQN or RSRSDuelingDQN or RSRSDuelingDDQN or
ConvDQN or ConvDDQN or ConvRSRSDQN
"""
algo = ['DQN', 'ConvDQN']
sim = 1
epi = 1000
alpha = 0.01
gamma = 0.9
epsilon = 0.1
tau = 0.1
hidden_size = 8
memory_capacity = 10**4
batch_size = 32
neighbor_frames = 4
aleph = 0.5
warmup = 10
k = 5
zeta = 0.01
```

## 実行方法

```zsh
python main.py
```
