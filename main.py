import os
from datetime import datetime
import gymnasium as gym
from gymnasium.spaces.discrete import Discrete
import ale_py

from simulator import simulation, conv_simulation, get_screen
from agent import Agent

from policy.dqn import DQN
from policy.ddqn import DDQN
from policy.duelingdqn import DuelingDQN
from policy.duelingddqn import DuelingDDQN
from policy.rsrs_dqn import RSRSDQN
from policy.rsrs_ddqn import RSRSDDQN
from policy.rsrs_duelingdqn import RSRSDuelingDQN
from policy.rsrs_duelingddqn import RSRSDuelingDDQN
from policy.conv_dqn import ConvDQN
from policy.conv_ddqn import ConvDDQN
from policy.conv_rsrs_dqn import ConvRSRSDQN
from policy.conv_rsrsdyn_dqn import ConvRSRSDynDQN
from policy.conv_rsrsaleph_dqn import ConvRSRSAlephDQN

from network.qnet import QNet
from network.duelingnet import DuelingNet
from network.rsrsnet import RSRSNet
from network.rsrs_duelingnet import RSRSDuelingNet
from network.conv_qnet import ConvQNet
from network.conv_rsrsnet import ConvRSRSNet
from network.conv_rsrsalephnet import ConvRSRSAlephNet


def space2size(space):
    if type(space) is Discrete:
        size = space.n
    else:
        size = 1
        for s in space.shape:
            size *= s
    return size


def compare_base_make_folder(env_name, algo, ex_param):
    if algo == 'DQN' or algo == 'DDQN' or algo == 'DuelingDQN' or algo == 'DuelingDDQN':
        base_param = {
            'algo': algo,
            'sim': 100,
            'epi': 1000,
            'alpha': 0.001,
            'gamma': 0.999,
            'epsilon': 0.01,
            'tau': 0.01,
            'hidden_size': 8,
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
            'gamma': 0.999,
            'epsilon': 0.01,
            'tau': 0.01,
            'hidden_size': 8,
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
    elif algo == 'RSRSDQN' or algo == 'RSRSDDQN' or algo == 'RSRSDuelingDQN' or algo == 'RSRSDuelingDDQN' or algo == 'ConvRSRSDQN' or algo == 'ConvRSRSDynDQN' or algo == 'ConvRSRSAlephDQN':
        base_param = {
            'algo': algo,
            'sim': 100,
            'epi': 1000,
            'alpha': 0.001,
            'gamma': 0.999,
            'epsilon': 0.01,
            'tau': 0.01,
            'hidden_size': 8,
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
        print(f'Not found algorithm {algo}')
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


if __name__ == '__main__':
    """
    algo: 
    DQN or DDQN or DuelingDQN or DuelingDDQN or
    RSRSDQN or RSRSDDQN or RSRSDuelingDQN or RSRSDuelingDDQN or
    ConvDQN or ConvDDQN or ConvRSRSDQN or ConvRSRSDynDQN or ConvRSRSAlephDQN
    """
    env_name = 'ALE/Breakout-v5'
    algos = ['ConvRSRSDQN']
    sim = 1
    epi = 500
    alpha = 0.001
    gamma = 0.99
    epsilon = 0.01
    tau = 0.1
    hidden_size = 64
    memory_capacity = 10**4
    batch_size = 32
    neighbor_frames = 4
    warmup = 10
    k = 5
    zeta = 0.01
    aleph_G = 0
    for algo in algos:
        env = gym.make(env_name, render_mode="rgb_array")
        env.reset()
        init_frame = get_screen(env)

        if algo == 'DQN' or algo == 'DDQN':
            model = QNet
        elif algo == 'ConvDQN' or algo == 'ConvDDQN':
            model = ConvQNet
        elif algo == 'RSRSDQN' or algo == 'RSRSDDQN':
            model = RSRSNet
        elif algo == 'DuelingDQN' or algo == 'DuelingDDQN':
            model = DuelingNet
        elif algo == 'RSRSDuelingDQN' or algo == 'RSRSDuelingDDQN':
            model = RSRSDuelingNet
        elif algo == 'ConvRSRSDQN' or algo == 'ConvRSRSDynDQN':
            model = ConvRSRSNet
        elif algo == 'ConvRSRSAlephDQN':
            model = ConvRSRSAlephNet
        else:
            print(f'Not found algorithm {algo}')
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
            'env': env,
            'model': model
        }

        if algo == 'DQN':
            policy = DQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'DDQN':
            policy = DDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'RSRSDQN':
            policy = RSRSDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'DuelingDQN':
            policy = DuelingDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'DuelingDDQN':
            policy = DuelingDDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'RSRSDDQN':
            policy = RSRSDDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'RSRSDuelingDQN':
            policy = RSRSDuelingDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'RSRSDuelingDDQN':
            policy = RSRSDuelingDDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            simulation(sim, epi, env, agent, result_dir_path)
        elif algo == 'ConvDQN':
            policy = ConvDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
        elif algo == 'ConvDDQN':
            policy = ConvDDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
        elif algo == 'ConvRSRSDQN':
            policy = ConvRSRSDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(env_name, algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
        elif algo == 'ConvRSRSDynDQN':
            policy = ConvRSRSDynDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
        elif algo == 'ConvRSRSAlephDQN':
            policy = ConvRSRSAlephDQN(**param)
            agent = Agent(policy)
            result_dir_path = make_param_file(algo, param, model, policy, agent)
            conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
        else:
            print(f'Not found algorithm {algo}')
            exit(1)
