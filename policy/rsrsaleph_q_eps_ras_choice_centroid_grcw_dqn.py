import sys
import numpy as np
import torch
from collections import deque

from policy.base_policy import BasePolicy


class RSRSAlephQEpsRASChoiceCentroidGRCwDQN(BasePolicy):
    def __init__(self, model_class, **kwargs):
        super().__init__(model_class, **kwargs)
        self.epsilon_dash = kwargs['epsilon_dash']
        self.k = kwargs['k']
        self.zeta = kwargs['zeta']
        self.global_aleph = kwargs['global_aleph']
        self.global_value_size = kwargs['global_value_size']
        self.centroids_decay = kwargs['centroids_decay']
        self.global_value = None
        self.global_value_list = None
        self.current_episode_actions = None
        self.current_episode_embeddings = None
        self.data_mean = None
        self.columns = None
        self.actions = None
        self.one_hot = None
        self.pseudo_counts = None
        self.weights = None
        self.centroids = None
        self.ras = None
        self.aleph = None

    def reset(self):
        super().reset()
        self.global_value = 0
        self.global_value_list = deque(maxlen=self.global_value_size)
        self.current_episode_actions = deque()
        self.current_episode_embeddings = deque()
        self.columns = self.k * self.action_space
        self.one_hot = np.identity(self.action_space, dtype=int)
        self.actions = np.repeat(self.one_hot, self.k, axis=0)
        self.pseudo_counts = np.zeros((self.columns, 1), dtype=float)
        self.weights = np.zeros((self.columns, 1), dtype=float)
        self.centroids = np.random.randn(self.columns, self.hidden_size)
        self.centroids /= np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.ras = np.zeros(self.action_space)

    def action(self, state):
        self.total_steps += 1
        if self.total_steps < self.warmup:
            action = np.random.choice(self.action_space)
        else:
            q_value = self.calc_q_value(state)
            beta = (self.global_aleph - self.global_value) / self.global_aleph
            beta = np.clip(beta, 0, 1)
            self.aleph = beta * self.global_aleph + (1 - beta) * q_value.max()
            diff = self.aleph - q_value
            if np.any(diff <= 0):
                rsrs = self.ras * (q_value - self.aleph)
                positive_rsrs = np.maximum(rsrs, 0)

                if np.sum(positive_rsrs) > 0:
                    self.pi = positive_rsrs / np.sum(positive_rsrs)
                else:
                    self.pi = np.ones_like(positive_rsrs) / len(positive_rsrs)
            else:
                z = 1.0 / np.sum(1.0 / diff)
                rho = z / diff
                b = self.ras / rho - 1.0 + sys.float_info.epsilon
                rsrs = (1.0 + max(b)) * rho - self.ras

                if min(rsrs) < 0:
                    rsrs -= min(rsrs)

                self.pi = rsrs / np.sum(rsrs)

            action = np.random.choice(self.action_space, p=self.pi)
        return action

    def update(self, state, action, reward, next_state, done):
        controllable_state = self.calc_embed(state)
        one_hot_action = np.zeros(self.action_space, dtype=int)
        one_hot_action[action] = 1
        self.current_episode_actions.append(one_hot_action)
        self.current_episode_embeddings.append(controllable_state)

        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size or len(self.replay_buffer.memory) < self.warmup:
            return
        
        self.calc_reliability(controllable_state)

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float64).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float64).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float64).to(self.device)
        d = torch.tensor(d, dtype=torch.float64).to(self.device)

        q = self.model(s)
        qa = q[np.arange(self.batch_size), a]
        with torch.no_grad():
            next_q_target = self.model_target(ns)
            next_qa_target = torch.amax(next_q_target, dim=1)
        
        target = r + self.gamma * next_qa_target * (1 - d)
        self.loss = self.criterion(qa, target)

        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
        self.optimizer.step()
        
        if self.sync_model_update == 'hard':
            if self.total_steps % self.target_update_freq == 0:
                self.sync_model_hard()
        elif self.sync_model_update == 'soft':
            self.sync_model_soft()
        else:
            raise ValueError(f'Invalid sync_model_update: {self.sync_model_update}')

    def update_episode(self, total_reward):
        self.global_value_list.append(total_reward)
        self.global_value = np.mean(self.global_value_list)
        self.update_centroid(np.array(self.current_episode_embeddings), np.array(self.current_episode_actions))
        self.current_episode_actions.clear()
        self.current_episode_embeddings.clear()

    def update_centroid(self, latents, x_actions):
        latents = self.normalize_data(latents)

        # 全ての擬似試行回数に忘却率をかける
        self.pseudo_counts *= self.centroids_decay
        self.weights *= self.centroids_decay
        self.centroids = self.normalize_data(self.centroids)

        # Action Class ごとに処理
        for cls in self.one_hot:
            # 同じアクションクラスを持つデータポイントをフィルタリング
            class_indices = np.where((self.actions == cls).all(axis=1))[0] #表の中から該当の class の index 取得
            class_centroids =  self.centroids[class_indices] # index を用いて重心から該当データを抽出
            x_class_indices = np.where((x_actions == cls).all(axis=1))[0] # 現在のデータの中から該当の class の index 取得
            x_class_points = latents[x_class_indices] # # index を用いてlatentから該当データを抽出
            if np.all(x_class_points) == 0:
                continue

            # クラスごとに最近傍の重心を見つけて，重心を更新
            if len(x_class_points) > 0:
                distances = np.linalg.norm(x_class_points[:, np.newaxis] - class_centroids, axis=2) #距離計算
                weights = 1 / (distances + 1e-10)
                for i, x in enumerate(x_class_points):
                    for j, weight in enumerate(weights[i]):
                        actual_index = class_indices[j]
                        self.centroids[actual_index] = (self.weights[actual_index] * self.centroids[actual_index] + weight * x) / (self.weights[actual_index] + weight)
                        self.weights[actual_index] += weight
                        self.pseudo_counts[actual_index] += 1

    def calc_reliability(self, latent):
        # 次元整形
        if len(latent.shape) == 1:
            latent = latent[np.newaxis, :] #(512) - > (1,512)
        
        # 1) 正規化 (中心化 + L2ノルムなど)
        latent = self.normalize_data(latent)  # ※region_nnと同じnormalize
        latent = torch.tensor(latent, dtype=torch.float32).to(self.device)
        centroids = self.normalize_data(self.centroids) #正規化
        centroids = torch.tensor(centroids, dtype=torch.float32).to(self.device)

        # 2) Centroid と距離を計算して類似度を求める
        distances = torch.norm(latent[:, None] - centroids, dim=2)  # shape=(1, n_centroids)
        similarities = 1.0 / (distances + 1e-10)
        similarities = similarities.cpu().numpy()

        # 3) 類似度に問題がある箇所は補正
        similarities[np.isnan(similarities)] = 0.0
        similarities = similarities.T
        similarities[similarities<=0] = 1e-8  # 負値があれば補正
        
        # 4) アクションごとに「類似したCentroidの疑似試行回数」を重み付き平均
        sim_num = np.zeros(self.one_hot.shape[0])
        for i, cls in enumerate(self.one_hot):
            # このアクションに該当するcentroidを検索
            class_indices = np.where((self.actions == cls).all(axis=1))[0]
            
            # 類似度の平均
            w_mean = np.average(similarities[class_indices], axis=0)  # shape=(1,n_centroids)なので similarities[0, ...]
            select_mean = np.average(self.pseudo_counts[class_indices], axis=0)
            sim_num[i] = w_mean * select_mean

        # 5) softmax で正規化
        y_max = np.max(sim_num)
        y = sim_num - y_max
        exp_y = np.exp(y)
        sum_exp_y = np.sum(exp_y)
        if sum_exp_y == 0:
            return np.ones_like(sim_num) / len(sim_num)  # 全部0の場合のfallback
        regional_confidence = exp_y / sum_exp_y
        self.ras = regional_confidence

    def normalize_data(self, data):
        if data.shape[0] != 1:
            # 中心化
            self.data_mean = np.mean(data, axis=0)
            centered_data = data - self.data_mean
            # 標準化
            self.norms = np.linalg.norm(centered_data, axis=1, keepdims=True)
            self.norms[self.norms==0.0] = 1e-8
            normalized_data = centered_data / self.norms
        else:
            if self.data_mean is not None:
                centered_data = data - self.data_mean
            else:
                centered_data = data
            # 標準化
            norms = np.linalg.norm(centered_data, axis=1, keepdims=True)
            norms[norms==0.0] = 1e-8
            normalized_data = centered_data / norms

        return normalized_data
