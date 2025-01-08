import os


def calculate_executed_sim(result_dir_path):
    executed_folders = [
        name for name in os.listdir(result_dir_path)
        if os.path.isdir(os.path.join(result_dir_path, name)) and name != 'average'
    ]

    return len(executed_folders)
