import numpy as np


class Collector:
    def __init__(self, sim, epi):
        self.sim = sim
        self.epi = epi
        self.reward_epi_list = []
        self.survived_step_epi_list = []
        self.reward_sim_list = np.zeros(self.epi)
        self.survived_step_sim_list = np.zeros(self.epi)

    def reset(self):
        self.reward_epi_list = []
        self.survived_step_epi_list = []

    def collect_episodic_data(self, reward, survived_step):
        self.reward_epi_list.append(reward)
        self.survived_step_epi_list.append(survived_step)

    def sum_episodic_data(self):
        self.reward_sim_list += self.reward_epi_list
        self.survived_step_sim_list += self.survived_step_epi_list

    def collect_simulation_data(self):
        self.reward_sim_list /= self.sim
        self.survived_step_sim_list /= self.sim

    def save_simulation_data(self, sim_dir_path):
        np.savetxt(sim_dir_path + 'reward.csv', self.reward_sim_list, delimiter=',')
        np.savetxt(sim_dir_path + 'survived_step.csv', self.survived_step_sim_list, delimiter=',')
