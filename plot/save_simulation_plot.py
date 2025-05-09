import matplotlib.pyplot as plt

def save_simulation_plot(average_sim_dir_path, metrics, df_concat):
    plt.figure(figsize=(12, 8))
    plt.plot(df_concat.mean(axis=1))
    plt.fill_between(range(df_concat.shape[0]), df_concat.mean(axis=1) - df_concat.std(axis=1), df_concat.mean(axis=1) + df_concat.std(axis=1), alpha=0.3)
    plt.title(f'average {metrics}')
    plt.xlabel('episode or step')
    plt.xlim(-1, len(df_concat) + 1)
    plt.savefig(average_sim_dir_path + f'average_{metrics}.png')
    plt.close()

