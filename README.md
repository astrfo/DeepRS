# DeepRS

## 使い方
GitHubからクローンして，frozenlakeブランチに切り替えて仮想環境を生成してください．
```
git clone https://github.com/takalabo/DeepRS.git
cd DeepRS
### frozenlakeブランチに移動
conda env create -f deeprs.yml
```

## パラメータ，環境の設定
[ここ](https://github.com/takalabo/DeepRS/blob/3681c0d966ddba3adf1d7baeb753ee0c176ce7ad/main.py#L114)から[ここ](https://github.com/takalabo/DeepRS/blob/3681c0d966ddba3adf1d7baeb753ee0c176ce7ad/main.py#L146)がパラメータと環境になっているため，ここを変更してください．
```
algo = 'sRSRS' #頭にsが付いているアルゴリズムはCNNなし
sim = 1
epi = 1000
alpha = 0.01
gamma = 0.9
epsilon = 0.1
tau = 0.1
hidden_size = 8
memory_capacity = 10**4
batch_size = 32
sync_interval = 20
neighbor_frames = 4
aleph = 0.5
warmup = 10
k = 5
zeta = 0.01
```

## 実行方法
```
python main.py
```
