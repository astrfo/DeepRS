import numpy as np
import os
import sys
from tqdm import tqdm
from collections import deque

from utils.get_screen_utils import get_screen
from plot.save_episode_plot import save_episode_plot
from plot.save_epi1000_plot import save_epi1000_plot
from plot.save_simulation_plot import save_simulation_plot


def simulation(sims, epis, env, agent, collector, result_dir_path):
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        agent.initialize()
        collector.initialize()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            collector.reset()
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
                collector.collect_step_data(reward, survived_step)
            collector.collect_episode_data(total_reward, survived_step)
            if (epi+1) % 1000 == 0:
                collector.save_epi1000_data(sim_dir_path, epi)
                save_epi1000_plot(collector, sim_dir_path, epi)
            if hasattr(agent.policy, 'update_global_value'):
                agent.policy.update_global_value(total_reward)
        collector.save_episode_data(sim_dir_path)
        save_episode_plot(collector, sim_dir_path)
    average_sim_dir_path = result_dir_path + 'average/'
    os.makedirs(average_sim_dir_path, exist_ok=True)
    collector.collect_simulation_data()
    collector.save_simulation_data(average_sim_dir_path)
    save_simulation_plot(collector, average_sim_dir_path)
    env.close()


def conv_simulation(sims, epis, env, agent, collector, neighbor_frames, result_dir_path):
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        agent.initialize()
        collector.initialize()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            collector.reset()
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
                collector.collect_step_data(reward, survived_step)
            collector.collect_episode_data(total_reward, survived_step)
        collector.save_episode_data(sim_dir_path)
        save_episode_plot(collector, sim_dir_path)
    average_sim_dir_path = result_dir_path + 'average/'
    os.makedirs(average_sim_dir_path, exist_ok=True)
    collector.collect_simulation_data()
    collector.save_simulation_data(average_sim_dir_path)
    save_simulation_plot(collector, average_sim_dir_path)
    env.close()


def atari_simulation(sims, epis, env, agent, collector, result_dir_path):
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        agent.initialize()
        collector.initialize()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            collector.reset()
            env.reset()
            total_reward, survived_step = 0, 0
            terminated, truncated = False, False

            no_op_steps = np.random.randint(1, 30)
            for _ in range(no_op_steps):
                action = 0
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    state, _ = env.reset()

            while not (terminated or truncated):
                action = agent.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                survived_step += 1
                collector.collect_step_data(reward, survived_step)
            collector.collect_episode_data(total_reward, survived_step)
            if (epi+1) % 1000 == 0:
                collector.save_epi1000_data(sim_dir_path, epi)
                save_epi1000_plot(collector, sim_dir_path, epi)
        collector.save_episode_data(sim_dir_path)
        save_episode_plot(collector, sim_dir_path)
    average_sim_dir_path = result_dir_path + 'average/'
    os.makedirs(average_sim_dir_path, exist_ok=True)
    collector.collect_simulation_data()
    collector.save_simulation_data(average_sim_dir_path)
    save_simulation_plot(collector, average_sim_dir_path)
    env.close()
