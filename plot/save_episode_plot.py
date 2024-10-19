import matplotlib.pyplot as plt

def save_episode_plot(collector, sim_dir_path):
    plt.figure(figsize=(12, 8))
    plt.plot(collector.reward_sim_list, label='Reward')
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.xlim(-1, len(collector.reward_sim_list) + 1)
    plt.legend()
    plt.savefig(sim_dir_path + 'reward.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(collector.survived_step_sim_list, label='Survived Step')
    plt.title('Survived Step per Episode')
    plt.xlabel('Episode')
    plt.xlim(-1, len(collector.survived_step_sim_list) + 1)
    plt.legend()
    plt.savefig(sim_dir_path + 'survived_step.png')
    plt.close()
