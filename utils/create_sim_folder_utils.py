import os


def create_sim_folder_utils(result_dir_path, sim):
    sim_dir_path = result_dir_path + f'{sim}/'
    os.makedirs(sim_dir_path, exist_ok=True)
    return sim_dir_path
