# 文脈付きバンディット問題のシミュレーション
実世界データセット及び最適切な基準値が常に一定の人工データセットを用いた, 文脈付きバンディット問題のシミュレーションについて.
実世界データセット参考文献：[DEEP BAYESIAN BANDITS SHOWDOWN](https://arxiv.org/pdf/1802.09127.pdf)
 
# Requirement
* Python 3.7.3
* numpy 1.16.3
* tensorflow 2.1.0
* pandas 0.25.1
* matplotlib 3.0.3
* scikit-learn 0.0
* scipy 1.4.1

# Real-World Datasets
データセットの情報を setup_context.py で管理.  
データのサンプリングを data_sampler.py で行う.  
datasetsファイルに, 下記の実世界データ(3種類)と人工データ(2種類×5つの水準)が入っている. そのためdatasetの入手方法を示すが, 改めて入手する必要はない.

* Mushroom data
    * キノコの22の特徴から可食を判別する.(n=8124) 特徴はone-hotベクトルに変換するため, 特徴ベクトルは117次元. 行動は食べるか食べないかの2種類. 食用キノコを食べると正の報酬が得られ, 毒キノコを食べると確率 p で正の報酬, 確率 1-p で大きな負の報酬が得られる. 食べない時の報酬は0. すべての報酬, および p の値はカスタマイズ可能. データセットは[UCI Machin Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)より取得できる.
* Financial data(raw_stock data)
    * NYSEおよびNasdaqの上場企業 d = 21 社の過去14年間の株価からデータセットを作成(n = 3713). データの内容は各株式の日毎のセッションの開始時と終了時の価格差. 行動は k = 8 とし, 見込みのある有価証券を表す線型結合になるよう作成した. データは[こちら](https://storage.googleapis.com/bandits_datasets/raw_stock_contexts)から入手可能.
    * 注意：現在こちらは使っていない.
* Jester data
    * 合計73421人のユーザーからの100のジョークに対する[-10、10] の連続評価. データのうち, n = 19181 のユーザーが40個のジョークをすべて評価している. この評価のうち d = 32 をユーザーの特徴量として, 残りの k = 8 を行動として用いる. エージェントは1つのジョークを推薦し, 選択したジョークに対するユーザーの評価を報酬として取得する. データは[こちら](https://storage.googleapis.com/bandits_datasets/jester_data_40jokes_19181users.npy)から入手可能.

* 最適切な希求水準が一定となる人工データ(基準値 = {0.5, 0.6, 0.7, 0.8, 0.9})
    * 腕の本数 8, 特徴量の次元 128, 最適な希求水準 5種類, データサイズ 10万 でパターン1・2 で生成
    * パターン1 one-hot 特徴ベクトル h に対する報酬パラメータWの生成
        * datasize分のone-hotの特徴ベクトルを生成
        * 全ての腕のランキングの回数が同じになるような定数表(特徴次元数×腕の数だけある)を作る
        * 各腕へのランキングに応じた報酬パラメータW(sigmoid を噛ませると報酬確率になるもの)を設計する
            * [0, 定めた最適な希求水準 + 0.5] で均等に報酬確率 P が割り振られるように、各腕の報酬確率を設計
            * 報酬確率から sigmoid をかます前のパラメータWを推定
                * W = -log_e (1/P - 1) で求められる
    * パターン2 パターン1のデータセットを使って混合特徴量の生成
        * 1番高い報酬確率を持つ腕が一致する one-hot特徴ベクトルを抽出
        * 一致する one-hot 特徴ベクトル群から混合係数 λ を生成
            * 一致する one-hot 特徴ベクトル群の中で 1 が入っている次元の箇所に混合係数 λ を割り当てる
                * 平均0、分散0.1 の正規分布から値をサンプリング
                * サンプリングした値の総和が1になるように、合計値でそれぞれの値を割る
                * 算出した値を one-hot の 1 の箇所に割り当てる
            * one-hot の 0 の箇所は微小ノイズを割り当てる
                * 平均0、分散0.001 の正規分布から値をサンプリング
                * その値をそのまま割り当てる
        * 新しく生成した混合特徴ベクトルから報酬確率も計算
            * トップの値が変わってないか確認 (トップの本数のみ確認済み)
    
    * 生成方法
        * 腕の本数 / 特徴量の次元 / 最適な希求水準 / データサイズ は artificial_data_generator_new.py 内で変更可能
        * パターン1 の特徴ベクトルは artificial_data_generator_new.py 内の FLAG = False に設定して下記を実行すると生成できる
        ```bash
        python artificial_data_generator_new.py
        ```
        * パターン2 の特徴ベクトルは artificial_data_generator_new.py 内の FLAG = True に設定して下記を実行すると生成できる
            * artificial_feature_data_mixed.csv と artificial_param_mixed.csv が生成される
            * 注意：パターン1 のdataset を使って生成するため、パターン1を実行してから実行する必要あり
        ```bash
        python artificial_data_generator_new.py
        ```
        
        
# Usage
(1)実行
```bash
python real_world_main.py
```
用いるアルゴリズムや各種パラメータなどシミュレーション設定はreal_world_main.py内で変更可能.  
基本的な結果はcsv, 生存率の計算に必要な結果はcsv_reward_countに格納

(2)基本的な結果のプロット
```bash
python plot/plot.py csv/結果が入っているディレクトリ
```
実行した結果はpngフォルダに保存される. 
出力される図

* regrets.png
* rewards.png
* greedy_rate.png
    * エージェントが最適だと思う行動を選択した割合(greedy率)
* accuracy.png
* errors.png
    * RMSE誤差
    
    また,各アルゴリズムの 1 sim あたりの平均実行時間は 1 sim time: [ ] でprint される.

(3)生存率の結果のプロット
```bash
python plot/plot_survival_rate.py.py csv_reward_count/結果が入っているディレクトリ 生存ライン 1日のstep数
```
実行した結果はpng/survival_rateディレクトリに保存される. 
# Note
以下の点が整備しきれていないため注意が必要.

* financialデータセットに関するコードが整理しきれていない. 
    * 現状のコードではMushroom と Jester のみ使用可.

 
# Author
* 南 朱音
* 東京電機大学理工学研究科
* 21rmd35@ms.dendai.ac.jp

