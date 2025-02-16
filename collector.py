import uuid
import pickle as pkl
import numpy as np
import torch
import glob
import os
import pandas as pd

from utils.create_folder_utils import create_sim_folder, create_average_folder
from plot.save_episode_plot import save_episode_plot
from plot.save_epi1000_plot import save_epi1000_plot
from plot.save_simulation_plot import save_simulation_plot


class Collector:
    def __init__(self, sim, epi, param, agent, policy, result_dir_path):
        self.sim = sim
        self.epi = epi
        self.param = param
        self.agent = agent
        self.policy = policy
        self.result_dir_path = result_dir_path

        # step data
        self.reward_step_list = []
        self.reward_greedy_step_list = []
        self.survived_step_step_list = []
        self.survived_step_greedy_step_list = []
        self.q_value_step_list = []
        self.embed_step_list = []
        self.loss_step_list = []
        self.pi_step_list = []
        self.terminal_state_count_step_list = []
        self.state_step_list = []

        # episode data
        self.reward_epi_list = []
        self.reward_greedy_epi_list = []
        self.survived_step_epi_list = []
        self.survived_step_greedy_epi_list = []
        self.terminal_state_ratio_epi_list = []

    def format(self):
        data = {}
        data['param'] = self.param
        data['reward'] = self.reward_epi_list
        data['reward_greedy'] = self.reward_greedy_epi_list
        data['survived_step'] = self.survived_step_epi_list
        data['survived_step_greedy'] = self.survived_step_greedy_epi_list
        data['q_value'] = self.q_value_step_list
        data['embed'] = self.embed_step_list
        data['loss'] = self.loss_step_list
        data['pi'] = self.pi_step_list
        return data

    def initialize(self, sim):
        self.sim_dir_path = create_sim_folder(self.result_dir_path, sim)
        self.reward_step_list = []
        self.reward_greedy_step_list = []
        self.survived_step_step_list = []
        self.survived_step_greedy_step_list = []
        self.q_value_step_list = []
        self.selected_q_value_step_list = []
        self.embed_step_list = []
        self.action_step_list = []
        self.loss_step_list = []
        self.pi_step_list = []
        self.terminal_state_count_step_list = []
        self.state_step_list = []
        self.reward_epi_list = []
        self.reward_greedy_epi_list = []
        self.survived_step_epi_list = []
        self.survived_step_greedy_epi_list = []
        self.terminal_state_ratio_epi_list = []

    def reset(self):
        self.reward_step_list = []
        self.reward_greedy_step_list = []
        self.survived_step_step_list = []
        self.survived_step_greedy_step_list = []
        self.terminal_state_count_step_list = []

    def collect_step_data(self, reward, survived_step):
        self.reward_step_list.append(reward)
        self.survived_step_step_list.append(survived_step)
        q_value = self.policy.calc_q_value(self.agent.current_state)
        self.q_value_step_list.append(q_value)
        self.selected_q_value_step_list.append(q_value[self.agent.current_action])
        self.action_step_list.append(self.agent.current_action)
        self.state_step_list.append(self.agent.current_state)
        if hasattr(self.policy, 'embed') and self.policy.embed is not None:
            embed = self.policy.embed(self.agent.current_state)
            self.embed_step_list.append(embed)
        if hasattr(self.policy, 'loss') and self.policy.loss is not None:
            self.loss_step_list.append(self.policy.loss.item())
        if hasattr(self.policy, 'pi') and self.policy.pi is not None:
            self.pi_step_list.append(self.policy.pi)
        if hasattr(self.policy, 'terminal_state_count'):
            self.terminal_state_count_step_list.append(self.policy.terminal_state_count)
    
    def collect_greedy_step_data(self, reward_greedy, survived_step_greedy):
        self.reward_greedy_step_list.append(reward_greedy)
        self.survived_step_greedy_step_list.append(survived_step_greedy)

    def collect_episode_data(self, total_reward, survived_step):
        self.reward_epi_list.append(total_reward)
        self.survived_step_epi_list.append(survived_step)
        self.terminal_state_ratio_epi_list.append(sum(self.terminal_state_count_step_list) / (survived_step * self.param['batch_size']))

    def collect_greedy_episode_data(self, total_reward_greedy, survived_step_greedy):
        self.reward_greedy_epi_list.append(total_reward_greedy)
        self.survived_step_greedy_epi_list.append(survived_step_greedy)

    def save_epi1000_data(self, epi):
        torch.save(self.agent.policy.model.state_dict(), self.sim_dir_path + f'model_episode{epi}.pth')
        np.savetxt(self.sim_dir_path + f'reward_epi{epi}.csv', self.reward_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'reward_greedy_epi{epi}.csv', self.reward_greedy_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'survived_step_epi{epi}.csv', self.survived_step_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'survived_step_greedy_epi{epi}.csv', self.survived_step_greedy_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'q_value_epi{epi}.csv', self.q_value_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'selected_q_value_epi{epi}.csv', self.selected_q_value_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'embed_epi{epi}.csv', self.embed_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'action_epi{epi}.csv', self.action_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'loss_epi{epi}.csv', self.loss_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'pi_epi{epi}.csv', self.pi_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'terminal_state_ratio_epi{epi}.csv', self.terminal_state_ratio_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + f'state_epi{epi}.csv', self.state_step_list, delimiter=',')
        save_epi1000_plot(self, epi)
        
        episode_data = self.format()
        with open(self.sim_dir_path + f'episode{epi}_{uuid.uuid4().hex[:6]}.pickle', 'wb') as f:
            pkl.dump(episode_data, f)

    def save_episode_data(self):
        torch.save(self.agent.policy.model.state_dict(), self.sim_dir_path + f'model_epi{self.epi}.pth')
        np.savetxt(self.sim_dir_path + 'reward.csv', self.reward_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'reward_greedy.csv', self.reward_greedy_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'survived_step.csv', self.survived_step_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'survived_step_greedy.csv', self.survived_step_greedy_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'q_value.csv', self.q_value_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'selected_q_value.csv', self.selected_q_value_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'embed.csv', self.embed_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'action.csv', self.action_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'loss.csv', self.loss_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'pi.csv', self.pi_step_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'terminal_state_ratio.csv', self.terminal_state_ratio_epi_list, delimiter=',')
        np.savetxt(self.sim_dir_path + 'state.csv', self.state_step_list, delimiter=',')
        save_episode_plot(self)

        episode_data = self.format()
        with open(self.sim_dir_path + f'episode{self.epi}_{uuid.uuid4().hex[:6]}.pickle', 'wb') as f:
            pkl.dump(episode_data, f)

    def save_simulation_data(self):
        average_sim_dir_path = create_average_folder(self.result_dir_path)

        metrics_list = ['reward', 'reward_greedy', 'survived_step', 'survived_step_greedy', 'terminal_state_ratio']
        for metrics in metrics_list:
            search_pattern = os.path.join(self.result_dir_path, '**', f'{metrics}.csv')
            csv_files = glob.glob(search_pattern, recursive=True)
            df = [pd.read_csv(csv_file, header=None) for csv_file in csv_files]
            df_concat = pd.concat(df, axis=1)
            df_average = df_concat.mean(axis=1)
            np.savetxt(average_sim_dir_path + f'average_{metrics}.csv', df_average, delimiter=',')
            save_simulation_plot(average_sim_dir_path, metrics, df_concat)
