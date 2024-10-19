import uuid
import pickle as pkl
import numpy as np


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
        if self.is_aleph_s_in_policy:
            self.aleph_step_list = []

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
        if self.is_aleph_s_in_policy:
            data['aleph'] = self.aleph_step_list
        return data

    def initialize(self):
        self.reward_step_list = []
        self.survived_step_step_list = []
        self.reward_epi_list = []
        self.survived_step_epi_list = []

    def reset(self):
        self.reward_step_list = []
        self.survived_step_step_list = []

    def collect_step_data(self, reward, survived_step):
        self.reward_step_list.append(reward)
        self.survived_step_step_list.append(survived_step)
        self.q_value_step_list.append(self.agent.policy.q_value(self.agent.current_state))
        if self.is_aleph_s_in_policy:
            self.aleph_step_list.append(self.agent.policy.aleph_s(self.agent.current_state))

    def save_step_data(self, sim_dir_path):
        np.savetxt(sim_dir_path + 'q_value.csv', self.q_value_step_list, delimiter=',')
        if self.is_aleph_s_in_policy:
            np.savetxt(sim_dir_path + 'aleph.csv', self.aleph_step_list, delimiter=',')

    def collect_episode_data(self, total_reward, survived_step):
        self.reward_epi_list.append(total_reward)
        self.survived_step_epi_list.append(survived_step)

    def save_episode_data(self, sim_dir_path):
        self.reward_sim_list += self.reward_epi_list
        self.survived_step_sim_list += self.survived_step_epi_list
        np.savetxt(sim_dir_path + 'reward.csv', self.reward_epi_list, delimiter=',')
        np.savetxt(sim_dir_path + 'survived_step.csv', self.survived_step_epi_list, delimiter=',')

        episode_data = self.format()
        with open(sim_dir_path + f'episode_{uuid.uuid4().hex[:6]}.pickle', 'wb') as f:
            pkl.dump(episode_data, f)

    def collect_simulation_data(self):
        self.reward_sim_list /= self.sim
        self.survived_step_sim_list /= self.sim

    def save_simulation_data(self, average_sim_dir_path):
        np.savetxt(average_sim_dir_path + 'average_reward.csv', self.reward_sim_list, delimiter=',')
        np.savetxt(average_sim_dir_path + 'average_survived_step.csv', self.survived_step_sim_list, delimiter=',')
