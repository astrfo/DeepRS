import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

from simulator import simulation, conv_simulation, atari_simulation
from agent import Agent
from collector import Collector
from utils.get_screen_utils import get_screen
from utils.space2size_utils import space2size
from utils.make_param_file_utils import make_param_file

from policy.dqn import DQN
from policy.ddqn import DDQN
from policy.duelingdqn import DuelingDQN
from policy.duelingddqn import DuelingDDQN
from policy.dqn_rnd import DQN_RND
from policy.rsrs_dqn import RSRSDQN
from policy.rsrs_ddqn import RSRSDDQN
from policy.rsrs_duelingdqn import RSRSDuelingDQN
from policy.rsrs_duelingddqn import RSRSDuelingDDQN
from policy.rsrsaleph_dqn import RSRSAlephDQN
from policy.rsrsaleph_q_eps_ras_dqn import RSRSAlephQEpsRASDQN
from policy.rsrsaleph_q_eps_ras_choice_dqn import RSRSAlephQEpsRASChoiceDQN
from policy.rsrsaleph_q_eps_dqn import RSRSAlephQEpsDQN
from policy.rsrsaleph_q_eps_ras_choice_dqn_rnd import RSRSAlephQEpsRASChoiceDQN_RND
from policy.rsrsaleph_q_eps_ras_choice_centroid_dqn import RSRSAlephQEpsRASChoiceCentroidDQN
from policy.conv_dqn import ConvDQN
from policy.conv_ddqn import ConvDDQN
from policy.conv_dqn_rnd import ConvDQN_RND
from policy.conv_dqn_atari import ConvDQNAtari
from policy.conv_rsrs_dqn import ConvRSRSDQN
from policy.conv_rsrsdyn_dqn import ConvRSRSDynDQN
from policy.conv_rsrsaleph_dqn import ConvRSRSAlephDQN
from policy.conv_rsrsaleph_q_eps_ras_choice_dqn_rnd import ConvRSRSAlephQEpsRASChoiceDQN_RND
from policy.conv_rsrsaleph_q_eps_ras_choice_dqn_atari import ConvRSRSAlephQEpsRASChoiceDQNAtari
from policy.conv_rsrsaleph_q_eps_ras_choice_centroid_dqn_atari import ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari

from network.qnet import QNet
from network.duelingnet import DuelingNet
from network.rsrsnet import RSRSNet
from network.rsrsalephnet import RSRSAlephNet
from network.rsrs_duelingnet import RSRSDuelingNet
from network.conv_qnet import ConvQNet
from network.conv_atari_qnet import ConvQAtariNet
from network.conv_rsrsnet import ConvRSRSNet
from network.conv_rsrsalephnet import ConvRSRSAlephNet
from network.conv_atari_rsrsnet import ConvRSRSAtariNet


if __name__ == '__main__':
    """
    algo: 
    DQN or DDQN or DuelingDQN or DuelingDDQN or DQN_RND
    RSRSDQN or RSRSDDQN or RSRSDuelingDQN or RSRSDuelingDDQN or RSRSAlephDQN or RSRSAlephQEpsDQN or RSRSAlephQEpsRASDQN or RSRSAlephQEpsRASChoiceDQN or RSRSAlephQEpsRASChoiceDQN_RND or RSRSAlephQEpsRASChoiceCentroidDQN
    ConvDQN or ConvDDQN or ConvDQN_RND or ConvDQNAtari or ConvRSRSDQN or ConvRSRSDynDQN or ConvRSRSAlephDQN or ConvRSRSAlephQEpsRASChoiceDQN_RND or ConvRSRSAlephQEpsRASChoiceDQNAtari or ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari
    """
    # environment parameters
    env_name = 'BreakoutNoFrameskip-v4'
    algos = ['ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari']
    sim = 1
    epi = 10000

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
    aleph_G = 0

    # optimizer parameters
    adam_learning_rate = 0.001
    rmsprop_learning_rate = 0.00025
    rmsprop_alpha = 0.95
    rmsprop_eps = 0.01
    max_grad_norm = 40

    # loss function parameters
    mseloss_reduction = 'sum'

    # memory parameters
    replay_buffer_capacity = 1000000
    episodic_memory_capacity = 1000

    # network parameters
    hidden_size = 128
    embedding_size = 8
    neighbor_frames = 4

    # etc parameters
    sync_model_update = 'soft'
    warmup = 0
    tau = 0.01
    batch_size = 32
    target_update_freq = 500
    

    for algo in algos:
        if 'Conv' in algo:
            if 'Atari' in algo:
                env = gym.make(env_name, render_mode='rgb_array')
                env = AtariPreprocessing(env, frame_skip=4, grayscale_newaxis=False, scale_obs=True)
                env = FrameStackObservation(env, stack_size=4)
            else:
                env = gym.make(env_name, render_mode='rgb_array', frameskip=4).unwrapped
        else:
            if 'ALE' in env_name:
                env = gym.make(env_name, render_mode='rgb_array', obs_type='ram')
            else:
                env = gym.make(env_name, render_mode='rgb_array')
        env.reset()
        init_frame = get_screen(env)

        if algo == 'DQN' or algo == 'DDQN' or algo == 'DQN_RND':
            model = QNet
        elif algo == 'ConvDQN' or algo == 'ConvDDQN' or algo == 'ConvDQN_RND':
            model = ConvQNet
        elif algo == 'ConvDQNAtari':
            model = ConvQAtariNet
        elif algo == 'RSRSDQN' or algo == 'RSRSDDQN' or algo == 'RSRSAlephQEpsDQN' or algo == 'RSRSAlephQEpsRASDQN' or algo == 'RSRSAlephQEpsRASChoiceDQN' or algo == 'RSRSAlephQEpsRASChoiceDQN_RND' or algo == 'RSRSAlephQEpsRASChoiceCentroidDQN':
            model = RSRSNet
        elif algo == 'RSRSAlephDQN':
            model = RSRSAlephNet
        elif algo == 'DuelingDQN' or algo == 'DuelingDDQN':
            model = DuelingNet
        elif algo == 'RSRSDuelingDQN' or algo == 'RSRSDuelingDDQN':
            model = RSRSDuelingNet
        elif algo == 'ConvRSRSDQN' or algo == 'ConvRSRSDynDQN' or algo == 'ConvRSRSAlephQEpsRASChoiceDQN_RND':
            model = ConvRSRSNet
        elif algo == 'ConvRSRSAlephDQN':
            model = ConvRSRSAlephNet
        elif algo == 'ConvRSRSAlephQEpsRASChoiceDQNAtari' or algo == 'ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari':
            model = ConvRSRSAtariNet
        else:
            print(f'Not found network {algo}')
            exit(1)

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
            'aleph_G': aleph_G,
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
            'frame_shape': init_frame.shape,
            'model': model
        }

        if algo == 'DQN':
            policy = DQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'DDQN':
            policy = DDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSDQN':
            policy = RSRSDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSAlephDQN':
            policy = RSRSAlephDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSAlephQEpsDQN':
            policy = RSRSAlephQEpsDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSAlephQEpsRASDQN':
            policy = RSRSAlephQEpsRASDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSAlephQEpsRASChoiceDQN':
            policy = RSRSAlephQEpsRASChoiceDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSAlephQEpsRASChoiceDQN_RND':
            policy = RSRSAlephQEpsRASChoiceDQN_RND(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSAlephQEpsRASChoiceCentroidDQN':
            policy = RSRSAlephQEpsRASChoiceCentroidDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'DuelingDQN':
            policy = DuelingDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'DuelingDDQN':
            policy = DuelingDDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'DQN_RND':
            policy = DQN_RND(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSDDQN':
            policy = RSRSDDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSDuelingDQN':
            policy = RSRSDuelingDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'RSRSDuelingDDQN':
            policy = RSRSDuelingDDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'ConvDQN':
            policy = ConvDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        elif algo == 'ConvDDQN':
            policy = ConvDDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        elif algo == 'ConvDQN_RND':
            policy = ConvDQN_RND(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        elif algo == 'ConvDQNAtari':
            policy = ConvDQNAtari(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            atari_simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'ConvRSRSDQN':
            policy = ConvRSRSDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        elif algo == 'ConvRSRSDynDQN':
            policy = ConvRSRSDynDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        elif algo == 'ConvRSRSAlephDQN':
            policy = ConvRSRSAlephDQN(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        elif algo == 'ConvRSRSAlephQEpsRASChoiceDQN_RND':
            policy = ConvRSRSAlephQEpsRASChoiceDQN_RND(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, collector, neighbor_frames, result_dir_path)
        elif algo == 'ConvRSRSAlephQEpsRASChoiceDQNAtari':
            policy = ConvRSRSAlephQEpsRASChoiceDQNAtari(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            atari_simulation(sim, epi, env, agent, collector, result_dir_path)
        elif algo == 'ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari':
            policy = ConvRSRSAlephQEpsRASChoiceCentroidDQNAtari(**param)
            agent = Agent(policy)
            collector = Collector(sim, epi, param, agent, policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            atari_simulation(sim, epi, env, agent, collector, result_dir_path)
        else:
            print(f'Not found algorithm {algo}')
            exit(1)
