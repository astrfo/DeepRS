import os
from datetime import datetime


def compare_base_make_folder(env_name, algo, ex_param):
    if algo == 'DQN' or algo == 'DDQN' or algo == 'DuelingDQN' or algo == 'DuelingDDQN':
        base_param = {
            'algo': algo,
            'sim': 100,
            'epi': 1000,
            'alpha': 0.001,
            'gamma': 0.99,
            'epsilon': 0.01,
            'tau': 0.01,
            'hidden_size': 128,
            'memory_capacity': 10**4,
            'batch_size': 32,
        }
        folder_name = algo
        for (base_k, base_v), (ex_k, ex_v) in zip(base_param.items(), ex_param.items()):
            if (base_k == ex_k) and (base_v != ex_v):
                folder_name += f'_{ex_k}{ex_v}'
        ex_folder_path = f'log/{env_name}/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    elif algo == 'ConvDQN' or algo == 'ConvDDQN':
        base_param = {
            'algo': algo,
            'sim': 100,
            'epi': 1000,
            'alpha': 0.001,
            'gamma': 0.99,
            'epsilon': 0.01,
            'tau': 0.01,
            'hidden_size': 128,
            'memory_capacity': 10**4,
            'batch_size': 32,
            'neighbor_frames': 4,
        }
        folder_name = algo
        for (base_k, base_v), (ex_k, ex_v) in zip(base_param.items(), ex_param.items()):
            if (base_k == ex_k) and (base_v != ex_v):
                folder_name += f'_{ex_k}{ex_v}'
        ex_folder_path = f'log/{env_name}/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    elif algo == 'RSRSDQN' or algo == 'RSRSDDQN' or algo == 'RSRSDuelingDQN' or algo == 'RSRSDuelingDDQN' or algo == 'RSRSAlephDQN' or algo == 'RSRSAlephQEpsDQN' or algo == 'RSRSAlephQEpsRASDQN' or algo == 'ConvRSRSDQN' or algo == 'ConvRSRSDynDQN' or algo == 'ConvRSRSAlephDQN':
        base_param = {
            'algo': algo,
            'sim': 100,
            'epi': 1000,
            'alpha': 0.001,
            'gamma': 0.99,
            'epsilon': 0.01,
            'tau': 0.01,
            'hidden_size': 128,
            'memory_capacity': 10**4,
            'batch_size': 32,
            'neighbor_frames': 4,
            'warmup': 10,
            'k': 5,
            'zeta': 0.008,
            'aleph_G': 2.0,
        }
        folder_name = algo
        for (base_k, base_v), (ex_k, ex_v) in zip(base_param.items(), ex_param.items()):
            if (base_k == ex_k) and (base_v != ex_v):
                folder_name += f'_{ex_k}{ex_v}'
        ex_folder_path = f'log/{env_name}/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    else:
        print(f'Cannot make file for algorithm {algo}')
        exit(1)
    return results_dir


def make_param_file(env_name, algo, param, model, policy, agent):
    result_dir_path = compare_base_make_folder(env_name, algo, param)
    f = open(result_dir_path + 'hyperparameter_list.txt', mode='w', encoding='utf-8')
    f.write(f'param: {param}\n')
    f.write(f'model: {model}\n')
    f.write(f'policy: {policy}\n')
    f.write(f'agent: {agent}\n')
    f.close()
    return result_dir_path
