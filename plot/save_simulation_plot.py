import matplotlib.pyplot as plt

def save_simulation_plot(average_sim_dir_path, metrics, df_average):
    plt.figure(figsize=(12, 8))
    plt.plot(df_average)
    plt.title(f'average {metrics}')
    plt.xlabel('episode or step')
    plt.xlim(-1, len(df_average) + 1)
    plt.savefig(average_sim_dir_path + f'average_{metrics}.png')
    plt.close()

