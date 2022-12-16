import os
from datetime import datetime
import gym

from simulator import simulation, conv_simulation, get_screen
from agent import Agent
from policy import DQN, ConvDQN, QNet, ConvQNet


if __name__ == '__main__':
    sim = 1
    epi = 400
    alpha = 0.0005
    gamma = 0.98
    epsilon = 0.1
    hidden_size = 128
    embed_size = 64
    sync_interval = 20
    neighbor_frames = 4
    memory_capacity = 10**4
    batch_size = 32
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    env.reset()
    init_frame = get_screen(env)

    param = {
        'sim': sim,
        'epi': epi,
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'hidden_size': hidden_size,
        'embed_size': embed_size,
        'action_space': env.action_space.n,
        'state_shape': env.observation_space.shape[0],
        'frame_shape': init_frame.shape,
        'sync_interval': sync_interval,
        'neighbor_frames': neighbor_frames,
        'memory_capacity': memory_capacity,
        'batch_size': batch_size,
        'env': env,
        # 'model': QNet,
        'model': ConvQNet,
    }

    # policy = DQN(**param)
    policy = ConvDQN(**param)
    agent = Agent(policy)

    time_now = datetime.now()
    result_dir_path = f'log/{time_now:%Y%m%d%H%M}/'
    os.makedirs(result_dir_path, exist_ok=True)
    f = open(result_dir_path + 'hyperparameter_list.txt', mode='w', encoding='utf-8')
    f.write(f'param: {param}\n')
    f.close()

    # simulation(sim, epi, env, agent, result_dir_path)
    conv_simulation(sim, epi, env, agent, neighbor_frames, result_dir_path)
