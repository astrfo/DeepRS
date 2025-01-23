import gymnasium as gym
import ale_py

from simulator import simulation, conv_simulation
from agent import Agent
from collector import Collector
from utils.get_screen_utils import get_screen
from utils.space2size_utils import space2size
from utils.make_param_file_utils import make_param_file

from policy.dqn import DQN
from policy.ddqn import DDQN
from policy.duelingdqn import DuelingDQN
from policy.duelingddqn import DuelingDDQN
from policy.rsrs_dqn import RSRSDQN
from policy.rsrs_ddqn import RSRSDDQN
from policy.rsrs_duelingdqn import RSRSDuelingDQN
from policy.rsrs_duelingddqn import RSRSDuelingDDQN
from policy.rsrsaleph_dqn import RSRSAlephDQN
from policy.rsrsaleph_q_eps_ras_dqn import RSRSAlephQEpsRASDQN
from policy.rsrsaleph_q_eps_ras_choice_dqn import RSRSAlephQEpsRASChoiceDQN
from policy.rsrsaleph_q_eps_dqn import RSRSAlephQEpsDQN
from policy.conv_dqn import ConvDQN
from policy.conv_ddqn import ConvDDQN
from policy.conv_rsrs_dqn import ConvRSRSDQN
from policy.conv_rsrsdyn_dqn import ConvRSRSDynDQN
from policy.conv_rsrsaleph_dqn import ConvRSRSAlephDQN

from network.qnet import QNet
from network.duelingnet import DuelingNet
from network.rsrsnet import RSRSNet
from network.rsrsalephnet import RSRSAlephNet
from network.rsrs_duelingnet import RSRSDuelingNet
from network.conv_qnet import ConvQNet
from network.conv_rsrsnet import ConvRSRSNet
from network.conv_rsrsalephnet import ConvRSRSAlephNet


if __name__ == '__main__':
    """
    algo: 
    DQN or DDQN or DuelingDQN or DuelingDDQN or
    RSRSDQN or RSRSDDQN or RSRSDuelingDQN or RSRSDuelingDDQN or RSRSAlephDQN or RSRSAlephQEpsDQN or RSRSAlephQEpsRASDQN or RSRSAlephQEpsRASChoiceDQN or
    ConvDQN or ConvDDQN or ConvRSRSDQN or ConvRSRSDynDQN or ConvRSRSAlephDQN
    """
    env_name = 'CartPole-v1'
    algos = ['RSRSAlephQEpsRASChoiceDQN']
    sim = 10
    epi = 1021
    alpha = 0.001
    gamma = 0.99
    epsilon = 0.01
    tau = 0.01
    hidden_size = 128
    memory_capacity = 10**4
    batch_size = 32
    neighbor_frames = 4
    warmup = 50
    k = 5
    zeta = 0.01
    aleph_G = 0
    for algo in algos:
        if 'Conv' in algo:
            env = gym.make(env_name, render_mode='rgb_array').unwrapped
        else:
            env = gym.make(env_name, render_mode='rgb_array')
        env.reset()
        init_frame = get_screen(env)

        if algo == 'DQN' or algo == 'DDQN':
            model = QNet
        elif algo == 'ConvDQN' or algo == 'ConvDDQN':
            model = ConvQNet
        elif algo == 'RSRSDQN' or algo == 'RSRSDDQN' or algo == 'RSRSAlephQEpsDQN' or 'RSRSAlephQEpsRASDQN' or algo == 'RSRSAlephQEpsRASChoiceDQN':
            model = RSRSNet
        elif algo == 'RSRSAlephDQN':
            model = RSRSAlephNet
        elif algo == 'DuelingDQN' or algo == 'DuelingDDQN':
            model = DuelingNet
        elif algo == 'RSRSDuelingDQN' or algo == 'RSRSDuelingDDQN':
            model = RSRSDuelingNet
        elif algo == 'ConvRSRSDQN' or algo == 'ConvRSRSDynDQN':
            model = ConvRSRSNet
        elif algo == 'ConvRSRSAlephDQN':
            model = ConvRSRSAlephNet
        else:
            print(f'Not found network {algo}')
            exit(1)

        param = {
            'algo': algo,
            'sim': sim,
            'epi': epi,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon,
            'tau': tau,
            'hidden_size': hidden_size,
            'memory_capacity': memory_capacity,
            'batch_size': batch_size,
            'neighbor_frames': neighbor_frames,
            'warmup': warmup,
            'k': k,
            'zeta': zeta,
            'aleph_G': aleph_G,
            'action_space': space2size(env.action_space),
            'state_space': space2size(env.observation_space),
            'frame_shape': init_frame.shape,
            'env': env_name,
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
        else:
            print(f'Not found algorithm {algo}')
            exit(1)
