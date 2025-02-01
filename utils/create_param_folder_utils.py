import os
from utils.create_hyperparameter_list_utils import create_hyperparameter_list
from utils.create_folder_utils import create_result_dir_path_folder
from utils.trimming_trailing_zero import trimming_trailing_zero
from utils.default_param_utils import DEFAULT_PARAMS


def create_compare_base_param_folder(env_name, algo, ex_param):
    base_param = {
        'epi': 0,
        'algo': algo,
    }
    base_param.update(DEFAULT_PARAMS)
    
    folder_name = algo
    for base_param_key, base_param_value in base_param.items():
        if ex_param[base_param_key] != base_param_value:
            ex_param_base_param_value = trimming_trailing_zero(ex_param[base_param_key])
            folder_name += f'_{base_param_key}{ex_param_base_param_value}'
    result_dir_path = create_result_dir_path_folder(env_name, folder_name)
    return result_dir_path

def create_param_folder(env_name, algo, param):
    result_dir_path = create_compare_base_param_folder(env_name, algo, param)
    create_hyperparameter_list(result_dir_path, param)
    return result_dir_path
