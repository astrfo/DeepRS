DEFAULT_PARAMS = {
    'gamma': 0.99,
    'epsilon_fixed': 0.01,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 1000000,
    'epsilon_dash': 0.001,
    'k': 5,
    'zeta': 0.01,
    'global_aleph': 500,
    'global_value_size': 500,
    'centroids_decay': 0.99,
    'optimizer': 'rmsprop',
    'adam_learning_rate': 0.001,
    'rmsprop_learning_rate': 0.00025,
    'rmsprop_alpha': 0.95,
    'rmsprop_eps': 0.01,
    'max_grad_norm': 40,
    'criterion': 'mseloss',
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
    'target_update_freq': 500
}


def apply_default_params(param):
    for key, default_value in DEFAULT_PARAMS.items():
        if param[key] is None:
            param[key] = [default_value]
        elif key not in param:
            ValueError(f'param[{key}] is None')
    return param
