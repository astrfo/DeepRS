import matplotlib.pyplot as plt

def save_step_plot(collector, sim_dir_path):
    plt.figure(figsize=(12, 8))
    plt.plot(collector.q_value_step_list, label='Q_value')
    plt.title('Q_value per Step')
    plt.xlabel('Step')
    plt.xlim(-1, len(collector.q_value_step_list) + 1)
    plt.legend()
    plt.savefig(sim_dir_path + 'q_value.png')
    plt.close()
    if not collector.is_aleph_s_in_policy:
        return
    plt.figure(figsize=(12, 8))
    plt.plot(collector.q_value_step_list, label='Q_value')
    plt.plot(collector.aleph_step_list, label='Aleph')
    plt.title('Q_value and Aleph per Step')
    plt.xlabel('Step')
    plt.xlim(-1, len(collector.aleph_step_list) + 1)
    plt.legend()
    plt.savefig(sim_dir_path + 'q_value_and_aleph.png')
    plt.close()
