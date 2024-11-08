import uuid
import pickle as pkl
import numpy as np
import torch


class Collector:
    def __init__(self, sim, epi, param, agent, policy):
        self.sim = sim
        self.epi = epi
        self.param = param
        self.agent = agent
        self.policy = policy
        self.is_aleph_s_in_policy = hasattr(self.policy, 'aleph_s')

        # step data
        self.reward_step_list = []
        self.survived_step_step_list = []
        self.q_value_step_list = []
        self.loss_step_list = []
        if self.is_aleph_s_in_policy:
            self.aleph_step_list = []
            self.satisfy_unsatisfy_count_list = np.zeros(2)

        # episode data
        self.reward_epi_list = []
        self.survived_step_epi_list = []

        # simulation data
        self.reward_sim_list = np.zeros(self.epi)
        self.survived_step_sim_list = np.zeros(self.epi)
    
    def format(self):
        data = {}
        data['param'] = self.param
        data['reward'] = self.reward_epi_list
        data['survived_step'] = self.survived_step_epi_list
        data['q_value'] = self.q_value_step_list
        data['loss'] = self.loss_step_list
        if self.is_aleph_s_in_policy:
            data['aleph'] = self.aleph_step_list
            data['satisfy_unsatisfy_count'] = self.satisfy_unsatisfy_count_list
        return data

    def initialize(self):
        self.reward_step_list = []
        self.survived_step_step_list = []
        self.q_value_step_list = []
        self.loss_step_list = []
        self.reward_epi_list = []
        self.survived_step_epi_list = []

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
        if self.is_aleph_s_in_policy:
            aleph = self.agent.policy.aleph_s(self.agent.current_state)
            self.aleph_step_list.append(aleph)
            if max(q_value) >= aleph: self.satisfy_unsatisfy_count_list[0] += 1
            else: self.satisfy_unsatisfy_count_list[1] += 1

    def collect_episode_data(self, total_reward, survived_step):
        self.reward_epi_list.append(total_reward)
        self.survived_step_epi_list.append(survived_step)

    def save_epi1000_data(self, sim_dir_path, epi):
        torch.save(self.agent.policy.model.state_dict(), sim_dir_path + f'{self.policy.__class__.__name__}_episode{epi}.pth')
        np.savetxt(sim_dir_path + f'reward_epi{epi}.csv', self.reward_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + f'survived_step_epi{epi}.csv', self.survived_step_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + f'loss_epi{epi}.csv', self.loss_step_list, delimiter=',')
        
        episode_data = self.format()
        with open(sim_dir_path + f'episode{epi}_{uuid.uuid4().hex[:6]}.pickle', 'wb') as f:
            pkl.dump(episode_data, f)

    def save_episode_data(self, sim_dir_path):
        self.reward_sim_list += self.reward_epi_list
        self.survived_step_sim_list += self.survived_step_epi_list
        torch.save(self.agent.policy.model.state_dict(), sim_dir_path + f'{self.policy.__class__.__name__}_sim{self.sim}_epi{self.epi}.pth')
        np.savetxt(sim_dir_path + 'reward.csv', self.reward_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + 'survived_step.csv', self.survived_step_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + 'q_value.csv', self.q_value_step_list, delimiter=',')
        np.savetxt(sim_dir_path + 'loss.csv', self.loss_step_list, delimiter=',')
        if self.is_aleph_s_in_policy:
            np.savetxt(sim_dir_path + 'aleph.csv', self.aleph_step_list, delimiter=',')
            np.savetxt(sim_dir_path + 'satisfy_unsatisfy_count.csv', self.satisfy_unsatisfy_count_list, delimiter=',')

        episode_data = self.format()
        with open(sim_dir_path + f'episode_{uuid.uuid4().hex[:6]}.pickle', 'wb') as f:
            pkl.dump(episode_data, f)

    def collect_simulation_data(self):
        self.reward_sim_list /= self.sim
        self.survived_step_sim_list /= self.sim

    def save_simulation_data(self, average_sim_dir_path):
        np.savetxt(average_sim_dir_path + 'average_reward.csv', self.reward_sim_list, delimiter=',')
        np.savetxt(average_sim_dir_path + 'average_survived_step.csv', self.survived_step_sim_list, delimiter=',')
