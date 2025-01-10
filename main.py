from simulator import simulation, conv_simulation, atari_simulation
from agent import Agent
from collector import Collector
from utils.get_screen_utils import get_screen  # TODO: frame_shape廃止時に削除
from utils.space2size_utils import space2size
from utils.create_param_folder_utils import create_param_folder
from utils.calculate_executed_sim_utils import calculate_executed_sim
from utils.create_env_utils import create_env
from utils.algo_class_utils import algo_class


if __name__ == '__main__':
    """
    algo: 
    DQN or DDQN
    RSRSAlephQEpsRASChoiceDQN or RSRSAlephQEpsRASChoiceCentroidDQN
    ConvDQN or ConvDDQN or ConvDQN_RND or ConvDQNAtari or ConvRSRSDQN or ConvRSRSDynDQN or ConvRSRSAlephDQN or ConvRSRSAlephQEpsRASChoiceDQN_RND or ConvRSRSAlephQEpsRASChoiceDQNAtari or ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari
    """
    # environment parameters
    env_name = 'CartPole-v1'
    algos = ['DQN', 'DDQN', 'RSRSAlephQEpsRASChoiceDQN', 'RSRSAlephQEpsRASChoiceCentroidDQN']
    sim = 10
    epi = 1000

    # q-learning parameters
    gamma = 0.99
    epsilon_fixed = 0.01
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1000000
    
    # rsrs parameters
    epsilon_dash = 0.001
    k = 5
    zeta = 0.01
    global_aleph = 500
    global_value_size = 500
    centroids_decay = 0.9

    # optimizer parameters
    adam_learning_rate = 0.001
    rmsprop_learning_rate = 0.00025
    rmsprop_alpha = 0.95
    rmsprop_eps = 0.01
    max_grad_norm = 40

    # loss function parameters
    mseloss_reduction = 'mean'

    # memory parameters
    replay_buffer_capacity = 10000
    episodic_memory_capacity = 10000

    # network parameters
    hidden_size = 128
    embedding_size = 8
    neighbor_frames = 4

    # etc parameters
    sync_model_update = 'soft'
    warmup = 50
    tau = 0.01
    batch_size = 32
    target_update_freq = 1000
    

    for algo in algos:
        env = create_env(env_name, algo)
        env.reset()
        init_frame = get_screen(env)  # TODO: frame_shape廃止時に削除

        model, policy_class = algo_class.get(algo)

        param = {
            'env': env_name,
            'algo': algo,
            'sim': sim,
            'epi': epi,
            'gamma': gamma,
            'epsilon_fixed': epsilon_fixed,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'epsilon_dash': epsilon_dash,
            'k': k,
            'zeta': zeta,
            'global_aleph': global_aleph,
            'global_value_size': global_value_size,
            'centroids_decay': centroids_decay,
            'adam_learning_rate': adam_learning_rate,
            'rmsprop_learning_rate': rmsprop_learning_rate,
            'rmsprop_alpha': rmsprop_alpha,
            'rmsprop_eps': rmsprop_eps,
            'max_grad_norm': max_grad_norm,
            'mseloss_reduction': mseloss_reduction,
            'replay_buffer_capacity': replay_buffer_capacity,
            'episodic_memory_capacity': episodic_memory_capacity,
            'hidden_size': hidden_size,
            'embedding_size': embedding_size,
            'neighbor_frames': neighbor_frames,
            'sync_model_update': sync_model_update,
            'warmup': warmup,
            'tau': tau,
            'batch_size': batch_size,
            'target_update_freq': target_update_freq,
            'action_space': space2size(env.action_space),
            'state_space': space2size(env.observation_space),
            'frame_shape': init_frame.shape,  # TODO: frame_shape廃止時に削除
            'model': model
        }

        result_dir_path = create_param_folder(env_name, algo, param)
        executed_sims = calculate_executed_sim(result_dir_path)
        policy = policy_class(**param)
        agent = Agent(policy)
        collector = Collector(sim, epi, param, agent, policy, result_dir_path)

        # TODO: Atariアルゴリズムを採用，Convを削除，のちにAtari→Convに変更
        if 'Conv' in algo:
            if 'Atari' in algo:
                atari_simulation(sim, epi, env, agent, collector, result_dir_path)
            else:
                conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        else:
            simulation(sim, executed_sims, epi, env, agent, collector, result_dir_path)
