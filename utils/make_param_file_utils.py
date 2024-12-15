import os
from datetime import datetime


def compare_base_make_folder(env_name, algo, ex_param):
    if algo == 'DQN' or algo == 'DDQN' or algo == 'DuelingDQN' or algo == 'DuelingDDQN' or algo == 'DQN_RND':
        base_param = {
            'algo': algo,
            'sim': 100,
            'epi': 1000,
            'gamma': 0.99,
            'epsilon_fixed': 0.01,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 1000000,
            'adam_learning_rate': 0.001,
            'rmsprop_learning_rate': 0.00025,
            'rmsprop_alpha': 0.95,
            'rmsprop_eps': 0.01,
            'mseloss_reduction': 'sum',
            'replay_buffer_capacity': 1000000,
            'hidden_size': 128,
            'sync_model_update': 'soft',
            'warmup': 0,
            'tau': 0.01,
            'batch_size': 32,
            'target_update_freq': 500,
        }
        folder_name = algo
        for base_param_key, base_param_value in base_param.items():
            if ex_param[base_param_key] != base_param_value:
                folder_name += f'_{base_param_key}{ex_param[base_param_key]}'
        ex_folder_path = f'log/{env_name}/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    elif algo == 'ConvDQN' or algo == 'ConvDDQN' or algo == 'ConvDQN_RND' or algo == 'ConvDQNAtari':
        base_param = {
            'algo': algo,
            'sim': 1,
            'epi': 1000,
            'gamma': 0.99,
            'epsilon_fixed': 0.01,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 1000000,
            'adam_learning_rate': 0.001,
            'rmsprop_learning_rate': 0.00025,
            'rmsprop_alpha': 0.95,
            'rmsprop_eps': 0.01,
            'max_grad_norm': 40,
            'mseloss_reduction': 'sum',
            'replay_buffer_capacity': 1000000,
            'hidden_size': 128,
            'neighbor_frames': 4,
            'sync_model_update': 'soft',
            'warmup': 50000,
            'tau': 0.01,
            'batch_size': 32,
            'target_update_freq': 10000,
        }
        folder_name = algo
        for base_param_key, base_param_value in base_param.items():
            if ex_param[base_param_key] != base_param_value:
                folder_name += f'_{base_param_key}{ex_param[base_param_key]}'
        ex_folder_path = f'log/{env_name}/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    elif algo == 'RSRSDQN' or algo == 'RSRSDDQN' or algo == 'RSRSDuelingDQN' or algo == 'RSRSDuelingDDQN' or algo == 'RSRSAlephDQN' or algo == 'RSRSAlephQEpsDQN' or algo == 'RSRSAlephQEpsRASDQN' or algo == 'RSRSAlephQEpsRASChoiceDQN' or algo == 'RSRSAlephQEpsRASChoiceDQN_RND' or algo == 'RSRSAlephQEpsRASChoiceCentroidDQN' or algo == 'RSRSAlephQEpsRASChoiceCentroidAlephGDQN' or algo == 'ConvRSRSDQN' or algo == 'ConvRSRSDynDQN' or algo == 'ConvRSRSAlephDQN' or algo == 'ConvRSRSAlephQEpsRASChoiceDQN_RND' or algo == 'ConvRSRSAlephQEpsRASChoiceDQNAtari' or algo == 'ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari':
        base_param = {
            'algo': algo,
            'sim': 100,
            'epi': 1000,
            'gamma': 0.99,
            'epsilon_dash': 0.01,
            'k': 5,
            'global_aleph': 500,
            'global_value_size': 500,
            'centroids_decay': 0.9,
            'adam_learning_rate': 0.001,
            'rmsprop_learning_rate': 0.00025,
            'rmsprop_alpha': 0.95,
            'rmsprop_eps': 0.01,
            'max_grad_norm': 40,
            'mseloss_reduction': 'sum',
            'replay_buffer_capacity': 1000000,
            'episodic_memory_capacity': 1000,
            'hidden_size': 128,
            'embedding_size': 8,
            'neighbor_frames': 4,
            'sync_model_update': 'soft',
            'warmup': 50000,
            'tau': 0.01,
            'batch_size': 32,
            'target_update_freq': 10000,
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
