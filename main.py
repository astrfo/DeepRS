import os
from datetime import datetime
import gym
from simulator import simulation

def main():
    sim = 5
    epi = 10
    aleph = 0.6
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped

    result_dir_path = make_folder()
    f = open(result_dir_path + 'hyperparameter_list.txt', mode='w', encoding='utf-8')
    f.write(f'sim:{sim}\nepi:{epi}\naleph:{aleph}\nenv:{env}\n')
    f.close()

    simulation(sim, epi, aleph, env, result_dir_path)


def make_folder():
    time_now = datetime.now()
    result_dir_path = f'log/{time_now:%Y%m%d%H%M}/'
    os.makedirs(result_dir_path, exist_ok=True)
    return result_dir_path

if __name__ == '__main__':
    main()
