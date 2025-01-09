import uuid
import pickle as pkl
import numpy as np
import torch
import glob
import os
import pandas as pd


class Collector:
    def __init__(self, sim, epi, param, agent, policy, sma_window=50):
        self.sim = sim
        self.epi = epi
        self.param = param
        self.agent = agent
        self.policy = policy
        self.sma_window = sma_window
        self.is_aleph_state_in_policy = hasattr(self.policy, 'update_global_value')

        # step data
        self.reward_step_list = []
        self.survived_step_step_list = []
        self.q_value_step_list = []
        self.loss_step_list = []
        self.loss_sma_step_list = []
        if self.is_aleph_state_in_policy:
            self.aleph_state_step_list = []
            self.aleph_beta_step_list = []
            self.satisfy_unsatisfy_count_list = np.zeros(2)

        # episode data
        self.reward_epi_list = []
        self.reward_sma_epi_list = []
        self.survived_step_epi_list = []
        self.survived_step_sma_epi_list = []
    
    def format(self):
        data = {}
        data['param'] = self.param
        data['reward'] = self.reward_epi_list
        data['reward_sma'] = self.reward_sma_epi_list
        data['survived_step'] = self.survived_step_epi_list
        data['survived_step_sma'] = self.survived_step_sma_epi_list
        data['q_value'] = self.q_value_step_list
        data['loss'] = self.loss_step_list
        data['loss_sma'] = self.loss_sma_step_list
        if self.is_aleph_state_in_policy:
            data['aleph'] = self.aleph_state_step_list
            data['aleph_beta'] = self.aleph_beta_step_list
            data['satisfy_unsatisfy_count'] = self.satisfy_unsatisfy_count_list
        return data

    def initialize(self):
        self.reward_step_list = []
        self.survived_step_step_list = []
        self.q_value_step_list = []
        self.loss_step_list = []
        self.loss_sma_step_list = []
        self.reward_epi_list = []
        self.reward_sma_epi_list = []
        self.survived_step_epi_list = []
        self.survived_step_sma_epi_list = []

    def reset(self):
        self.reward_step_list = []
        self.survived_step_step_list = []

    def collect_step_data(self, reward, survived_step):
        self.reward_step_list.append(reward)
        self.survived_step_step_list.append(survived_step)
        q_value = self.agent.policy.q_value(self.agent.current_state)
        self.q_value_step_list.append(q_value)
        if self.policy.loss is not None:
            self.loss_step_list.append(self.policy.loss.item())
            self.loss_sma_step_list.append(self.calculate_sma(self.loss_step_list))
        if self.is_aleph_state_in_policy and self.agent.policy.aleph_state is not None:
            aleph = self.agent.policy.aleph_state
            self.aleph_state_step_list.append(aleph)
            self.aleph_beta_step_list.append(self.agent.policy.aleph_beta)
            if max(q_value) >= aleph: self.satisfy_unsatisfy_count_list[0] += 1
            else: self.satisfy_unsatisfy_count_list[1] += 1

    def collect_episode_data(self, total_reward, survived_step):
        self.reward_epi_list.append(total_reward)
        self.survived_step_epi_list.append(survived_step)
        self.reward_sma_epi_list.append(self.calculate_sma(self.reward_epi_list))
        self.survived_step_sma_epi_list.append(self.calculate_sma(self.survived_step_epi_list))

    def calculate_sma(self, data_list):
        if len(data_list) < self.sma_window:
            return np.mean(data_list) if len(data_list) > 0 else 0
        else:
            return np.mean(data_list[-self.sma_window:])

    def save_epi1000_data(self, sim_dir_path, epi):
        torch.save(self.agent.policy.model.state_dict(), sim_dir_path + f'model_episode{epi}.pth')
        np.savetxt(sim_dir_path + f'reward_epi{epi}.csv', self.reward_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + f'reward_sma_epi{epi}.csv', self.reward_sma_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + f'survived_step_epi{epi}.csv', self.survived_step_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + f'survived_step_sma_epi{epi}.csv', self.survived_step_sma_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + f'loss_epi{epi}.csv', self.loss_step_list, delimiter=',')
        np.savetxt(sim_dir_path + f'loss_sma_epi{epi}.csv', self.loss_sma_step_list, delimiter=',')
        if self.is_aleph_state_in_policy:
            np.savetxt(sim_dir_path + f'aleph_state_epi{epi}.csv', self.aleph_state_step_list, delimiter=',')
            np.savetxt(sim_dir_path + f'aleph_beta_epi{epi}.csv', self.aleph_beta_step_list, delimiter=',')
            np.savetxt(sim_dir_path + f'satisfy_unsatisfy_count_epi{epi}.csv', self.satisfy_unsatisfy_count_list, delimiter=',')
        
        episode_data = self.format()
        with open(sim_dir_path + f'episode{epi}_{uuid.uuid4().hex[:6]}.pickle', 'wb') as f:
            pkl.dump(episode_data, f)

    def save_episode_data(self, sim_dir_path):
        torch.save(self.agent.policy.model.state_dict(), sim_dir_path + f'{self.policy.__class__.__name__}_sim{self.sim}_epi{self.epi}.pth')
        np.savetxt(sim_dir_path + 'reward.csv', self.reward_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + 'reward_sma.csv', self.reward_sma_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + 'survived_step.csv', self.survived_step_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + 'survived_step_sma.csv', self.survived_step_sma_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + 'q_value.csv', self.q_value_step_list, delimiter=',')
        np.savetxt(sim_dir_path + 'loss.csv', self.loss_step_list, delimiter=',')
        np.savetxt(sim_dir_path + 'loss_sma.csv', self.loss_sma_step_list, delimiter=',')
        if self.is_aleph_state_in_policy:
            np.savetxt(sim_dir_path + 'aleph_state.csv', self.aleph_state_step_list, delimiter=',')
            np.savetxt(sim_dir_path + 'aleph_beta.csv', self.aleph_beta_step_list, delimiter=',')
            np.savetxt(sim_dir_path + 'satisfy_unsatisfy_count.csv', self.satisfy_unsatisfy_count_list, delimiter=',')

        episode_data = self.format()
        with open(sim_dir_path + f'episode{self.epi}_{uuid.uuid4().hex[:6]}.pickle', 'wb') as f:
            pkl.dump(episode_data, f)

    def save_simulation_data(self, result_dir_path):
        average_sim_dir_path = result_dir_path + 'average/'
        os.makedirs(average_sim_dir_path, exist_ok=True)

        metrics_list = ['reward', 'reward_sma', 'survived_step', 'survived_step_sma', 'q_value', 'loss', 'loss_sma']
        for metrics in metrics_list:
            search_pattern = os.path.join(result_dir_path, '**', f'{metrics}.csv')
            csv_files = glob.glob(search_pattern, recursive=True)
            df = [pd.read_csv(csv_file, header=None) for csv_file in csv_files]
            df_concat = pd.concat(df, axis=1)
            df_average = df_concat.mean(axis=1)
            np.savetxt(average_sim_dir_path + f'average_{metrics}.csv', df_average, delimiter=',')
