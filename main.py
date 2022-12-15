import os
from datetime import datetime
import gym
from simulator import simulation
from agent import Agent
from policy import DQN
from replay_buffer import ReplayBuffer


if __name__ == '__main__':
    sim = 1
    epi = 300
    alpha = 0.0005
    gamma = 0.98
    epsilon = 0.1
    hidden_size = 128
    action_space = 2
    state_shape = 4
    sync_interval = 20
    memory_capacity = 10**4
    batch_size = 32
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    param = {
        'sim': sim,
        'epi': epi,
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'hidden_size': hidden_size,
        'action_space': env.action_space.n,
        'state_shape': env.observation_space.shape[0],
        'sync_interval': sync_interval,
        'memory_capacity': memory_capacity,
        'batch_size': batch_size,
        'env': env,
    }

    policy = DQN(**param)
    agent = Agent(policy)

    time_now = datetime.now()
    result_dir_path = f'log/{time_now:%Y%m%d%H%M}/'
    os.makedirs(result_dir_path, exist_ok=True)
    f = open(result_dir_path + 'hyperparameter_list.txt', mode='w', encoding='utf-8')
    f.write(f'param: {param}\n')
    f.close()

    simulation(sim, epi, env, agent, result_dir_path)
