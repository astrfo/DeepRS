import os
from utils.create_hyperparameter_list_utils import create_hyperparameter_list


def create_compare_base_param_folder(env_name, algo, ex_param):
    if algo == 'DQN' or algo == 'DDQN' or algo == 'DuelingDQN' or algo == 'DuelingDDQN' or algo == 'DQN_RND':
        base_param = {
            'epi': 0,
            'algo': algo,
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
    elif algo == 'ConvDQN' or algo == 'ConvDDQN' or algo == 'ConvDQN_RND' or algo == 'ConvDQNAtari':
        base_param = {
            'epi': 0,
            'algo': algo,
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
    elif algo == 'RSRSDQN' or algo == 'RSRSDDQN' or algo == 'RSRSDuelingDQN' or algo == 'RSRSDuelingDDQN' or algo == 'RSRSAlephDQN' or algo == 'RSRSAlephQEpsDQN' or algo == 'RSRSAlephQEpsRASDQN' or algo == 'RSRSAlephQEpsRASChoiceDQN' or algo == 'RSRSAlephQEpsCEChoiceDQN' or algo == 'RSRSAlephQEpsRASChoiceDQN_RND' or algo == 'RSRSAlephQEpsRASChoiceCentroidDQN' or algo == 'RSRSAlephQEpsRASChoiceCentroidAlephGDQN' or algo == 'ConvRSRSDQN' or algo == 'ConvRSRSDynDQN' or algo == 'ConvRSRSAlephDQN' or algo == 'ConvRSRSAlephQEpsRASChoiceDQN_RND' or algo == 'ConvRSRSAlephQEpsRASChoiceDQNAtari' or algo == 'ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari':
        base_param = {
            'epi': 0,
            'algo': algo,
            'gamma': 0.99,
            'epsilon_dash': 0.001,
            'k': 5,
            'global_aleph': 500,
            'global_value_size': 500,
            'centroids_decay': 0.9,
            'adam_learning_rate': 0.001,
            'rmsprop_learning_rate': 0.00025,
            'rmsprop_alpha': 0.95,
            'rmsprop_eps': 0.01,
            'max_grad_norm': 40,
            'mseloss_reduction': 'mean',
            'replay_buffer_capacity': 10000,
            'episodic_memory_capacity': 10000,
            'hidden_size': 128,
            'embedding_size': 8,
            'neighbor_frames': 4,
            'sync_model_update': 'soft',
            'warmup': 50,
            'tau': 0.01,
            'batch_size': 32,
            'target_update_freq': 1000,
        }
        folder_name = algo
        for base_param_key, base_param_value in base_param.items():
            if ex_param[base_param_key] != base_param_value:
                folder_name += f'_{base_param_key}{ex_param[base_param_key]}'
    else:
        print(f'Cannot make file for algorithm {algo}')
        exit(1)
    
    result_dir_path = f'log/{env_name}/{folder_name}/'
    os.makedirs(result_dir_path, exist_ok=True)
    return result_dir_path

def create_param_folder(env_name, algo, param):
    result_dir_path = create_compare_base_param_folder(env_name, algo, param)
    create_hyperparameter_list(result_dir_path, param)
    return result_dir_path
