# DeepRS

## 使い方

GitHubからクローンして，`requirements.txt`でライブラリをインストールしてください．

```zsh
git clone https://github.com/astrfo/DeepRS.git
cd DeepRS
pip install -r requirements.txt
```

## パラメータ，環境の設定

`config`ディレクトリ内に環境ごとのyamlファイルがあるので，これを編集するか，新たに作成してください．

```yaml
algo:
  - RS2AmbitionDQN
env: CartPole-v1
sim: 30
epi: 1000
param:
  # q-learning parameters
  gamma: null
  epsilon_fixed: null
  epsilon_start: null
  epsilon_end: null
  epsilon_decay: null

  # rsrs parameters
  epsilon_dash: null
  k: null
  zeta: null
  global_aleph: null
  global_value_size: null
  centroids_decay: null

  # optimizer parameters
  optimizer: null
  adam_learning_rate: null
  rmsprop_learning_rate: null
  rmsprop_alpha: null
  rmsprop_eps: null
  max_grad_norm: null

  # loss function parameters
  criterion: null
  mseloss_reduction: null

  # memory parameters
  replay_buffer_capacity: null
  episodic_memory_capacity: null

  # network parameters
  hidden_size: null
  embedding_size: null
  neighbor_frames: null

  # etc parameters
  sync_model_update: null
  warmup: null
  tau: null
  batch_size: null
  target_update_freq: null
```

## 実験用yamlファイルの使用方法

`config/main.yaml`を編集して，使用するyamlファイルを指定してください．

```yaml
experiments:
  - config/CartPole-v1/rsrs.yaml
```

## 実行方法

```zsh
python main.py
```
