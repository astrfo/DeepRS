import matplotlib.pyplot as plt

def save_simulation_plot(collector, average_sim_dir_path):
    plt.figure(figsize=(12, 8))
    plt.plot(collector.reward_sim_list, label='Average Reward')
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.xlim(-1, len(collector.reward_sim_list) + 1)
    plt.legend()
    plt.savefig(average_sim_dir_path + 'average_reward.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(collector.survived_step_sim_list, label='Average Survived Step')
    plt.title('Average Survived Step per Episode')
    plt.xlabel('Episode')
    plt.xlim(-1, len(collector.survived_step_sim_list) + 1)
    plt.legend()
    plt.savefig(average_sim_dir_path + 'average_survived_step.png')
    plt.close()
