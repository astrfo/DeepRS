import os
from datetime import datetime
import gym
from gym.spaces.discrete import Discrete

from simulator import simulation, conv_simulation, get_screen
from agent import Agent
from policy import DQN, DDQN, ConvDQN, ConvDDQN, QNet, ConvQNet, RSRS, ConvRSNet


def space2size(space):
    if type(space) is Discrete:
        size = space.n
    else:
        size = 1
        for s in space.shape:
            size *= s
    return size


def compare_base_make_folder(algo, ex_param):
    if algo == 'sDQN' or algo == 'sDDQN':
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
            'sync_interval': 2,
        }
        folder_name = algo
        for (base_k, base_v), (ex_k, ex_v) in zip(base_param.items(), ex_param.items()):
            if (base_k == ex_k) and (base_v != ex_v):
                folder_name += f'_{ex_k}{ex_v}'
        ex_folder_path = f'log/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    elif algo == 'DQN' or algo == 'DDQN':
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
            'sync_interval': 2,
            'neighbor_frames': 4,
        }
        folder_name = algo
        for (base_k, base_v), (ex_k, ex_v) in zip(base_param.items(), ex_param.items()):
            if (base_k == ex_k) and (base_v != ex_v):
                folder_name += f'_{ex_k}{ex_v}'
        ex_folder_path = f'log/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    elif algo == 'RSRS':
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
            'sync_interval': 2,
            'neighbor_frames': 4,
            'aleph': 0.7,
            'warmup': 10,
            'k': 5,
            'zeta': 0.008,
        }
        folder_name = algo
        for (base_k, base_v), (ex_k, ex_v) in zip(base_param.items(), ex_param.items()):
            if (base_k == ex_k) and (base_v != ex_v):
                folder_name += f'_{ex_k}{ex_v}'
        ex_folder_path = f'log/{folder_name}/'
        os.makedirs(ex_folder_path, exist_ok=True)
        time_now = datetime.now()
        results_dir = f'{ex_folder_path}{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
    else:
        print(f'Not found algorithm {algo}')
        exit(1)
    return results_dir


def make_param_file(algo, param, model, policy, agent):
    result_dir_path = compare_base_make_folder(algo, param)
    f = open(result_dir_path + 'hyperparameter_list.txt', mode='w', encoding='utf-8')
    f.write(f'param: {param}\n')
    f.write(f'model: {model}\n')
    f.write(f'policy: {policy}\n')
    f.write(f'agent: {agent}\n')
    f.close()
    return result_dir_path


if __name__ == '__main__':
    algo = 'sDQN' #sDQN or sDDQN or DQN or DDQN or RSRS
    sim = 1
    epi = 10000
    alpha = 0.01
    gamma = 0.9
    epsilon = 0.1
    tau = 0.01
    hidden_size = 64
    memory_capacity = 10**4
    batch_size = 32
    sync_interval = 20
    neighbor_frames = 4
    aleph = 0.7
    warmup = 10
    k = 5
    zeta = 0.008
    desc=[
        'SFFFFFFFF',
        'FFFFFFFFG',
    ]
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode='rgb_array').unwrapped

    env.reset()
    init_frame = get_screen(env)

    if algo == 'sDQN' or algo == 'sDDQN':
        model = QNet
    elif algo == 'DQN' or algo == 'DDQN':
        model = ConvQNet
    elif algo == 'RSRS':
        model = ConvRSNet
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
        'sync_interval': sync_interval,
        'neighbor_frames': neighbor_frames,
        'aleph': aleph,
        'warmup': warmup,
        'k': k,
        'zeta': zeta,
        'action_space': space2size(env.action_space),
        'state_space': space2size(env.observation_space),
        'frame_shape': init_frame.shape,
        'env': env,
        'model': model
    }

    if algo == 'sDQN':
        policy = DQN(**param)
        agent = Agent(policy)
        result_dir_path = make_param_file(algo, param, model, policy, agent)
        simulation(sim, epi, env, agent, result_dir_path)
    elif algo == 'sDDQN':
        policy = DDQN(**param)
        agent = Agent(policy)
        result_dir_path = make_param_file(algo, param, model, policy, agent)
        simulation(sim, epi, env, agent, result_dir_path)
    elif algo == 'DQN':
        policy = ConvDQN(**param)
        agent = Agent(policy)
        result_dir_path = make_param_file(algo, param, model, policy, agent)
        conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
    elif algo == 'DDQN':
        policy = ConvDDQN(**param)
        agent = Agent(policy)
        result_dir_path = make_param_file(algo, param, model, policy, agent)
        conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
    elif algo == 'RSRS':
        policy = RSRS(**param)
        agent = Agent(policy)
        result_dir_path = make_param_file(algo, param, model, policy, agent)
        conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
    else:
        print(f'Not found algorithm {algo}')
        exit(1)