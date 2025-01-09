import os


def create_result_dir_path_folder(env_name, folder_name):
    result_dir_path = f'log/{env_name}/{folder_name}/'
    os.makedirs(result_dir_path, exist_ok=True)
    return result_dir_path


def create_sim_folder(result_dir_path, sim):
    sim_dir_path = result_dir_path + f'{sim}/'
    os.makedirs(sim_dir_path, exist_ok=True)
    return sim_dir_path


def create_average_folder(result_dir_path):
    average_sim_dir_path = result_dir_path + 'average/'
    os.makedirs(average_sim_dir_path, exist_ok=True)
    return average_sim_dir_path
