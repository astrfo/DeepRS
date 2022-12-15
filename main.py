import os
from datetime import datetime
import gym
from simulator import simulation
from agent import Agent
from policy import DQN
from replay_buffer import ReplayBuffer


if __name__ == '__main__':
    sim = 5
    epi = 10
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    policy = DQN()
    agent = Agent(policy)

    time_now = datetime.now()
    result_dir_path = f'log/{time_now:%Y%m%d%H%M}/'
    os.makedirs(result_dir_path, exist_ok=True)
    f = open(result_dir_path + 'hyperparameter_list.txt', mode='w', encoding='utf-8')
    f.write(f'sim:{sim}\nepi:{epi}\nenv:{env}\n')
    f.close()

    simulation(sim, epi, env, agent, result_dir_path)
