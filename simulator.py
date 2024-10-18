import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import deque
import torchvision.transforms as T


def get_screen(env):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(size=(84, 84)),
                    T.Grayscale(num_output_channels=1)])
    screen = resize(env.render())
    screen = np.expand_dims(np.asarray(screen), axis=2).transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float64) / 255
    return screen


def plus_csv_plot(metrics, metric, path, name):
    metrics += metric
    np.savetxt(path + f'{name}.csv', metric, delimiter=',')
    sub_plot(path, name, metric)
    return metrics


def divide_csv_plot(metrics, path, name, sims):
    os.makedirs(path + 'average', exist_ok=True)
    average_path = path + 'average/'
    metrics /= sims
    np.savetxt(average_path + f'average_{name}.csv', metrics, delimiter=',')
    sub_plot(average_path, f'average_{name}', metrics)


def sub_plot(sim_dir_path, name, thing):
    plt.figure(figsize=(12, 8))
    plt.plot(thing, label=name)
    plt.title(name)
    plt.xlabel('Episode')
    plt.xlim(-1, len(thing)+1)
    plt.savefig(sim_dir_path + f'{name}.png')
    plt.close()


def eg_alephg_plot(sim_dir_path, eg, alephg):
    plt.figure(figsize=(12, 8))
    plt.plot(eg, label='eg')
    plt.plot(alephg, label='alephg')
    plt.title('eg and alephg')
    plt.xlabel('Episode')
    plt.xlim(-1, len(eg)+1)
    plt.legend()
    plt.savefig(sim_dir_path + f'eg_alephg.png')
    plt.close()


def qvalue_plot(sim_dir_path, name, thing):
    plt.figure(figsize=(12, 8))
    plt.plot(thing, label=['LEFT', 'DOWN', 'RIGHT', 'UP'])
    plt.title(name)
    plt.xlabel('Step')
    plt.legend()
    plt.savefig(sim_dir_path + f'{name}.png')
    plt.close()


def pi_plot(sim_dir_path, name, thing):
    plt.figure(figsize=(12, 8))
    plt.plot(thing, label=['LEFT', 'DOWN', 'RIGHT', 'UP'], alpha=0.2)
    plt.title(name)
    plt.xlabel('Step')
    plt.legend()
    plt.savefig(sim_dir_path + f'{name}.png')
    plt.close()


def simulation(sims, epis, env, agent, collector, result_dir_path):
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        agent.reset()
        collector.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            state, _ = env.reset()
            total_reward, survived_step = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = agent.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                survived_step += 1
            collector.collect_episodic_data(total_reward, survived_step)
        collector.sum_episodic_data()
    collector.collect_simulation_data()
    collector.save_simulation_data(sim_dir_path)
    env.close()


def conv_simulation(sims, epis, env, agent, collector, neighbor_frames, result_dir_path):
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        agent.reset()
        collector.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            env.reset()
            frame = get_screen(env)
            frames = deque([frame]*neighbor_frames, maxlen=neighbor_frames)
            state = np.stack(frames, axis=1)[0,:]
            total_reward, survived_step = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = agent.action(state)
                _, reward, terminated, truncated, _ = env.step(action)
                frame = get_screen(env)
                frames.append(frame)
                next_state = np.stack(frames, axis=1)[0,:]
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                survived_step += 1
            collector.collect_episodic_data(total_reward, survived_step)
        collector.sum_episodic_data()
    collector.collect_simulation_data()
    collector.save_simulation_data(sim_dir_path)
    env.close()
