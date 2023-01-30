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
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return screen


def sub_plot(sim_dir_path, name, thing):
    plt.plot(thing, label=name)
    plt.title(name)
    plt.xlabel('Episode')
    plt.savefig(sim_dir_path + f'{name}.png')
    plt.close()


def simulation(sims, epis, env, agent, result_dir_path):
    average_reward_list = np.zeros(epis)
    for sim in range(sims):
        total_reward_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            state = env.reset()[0]
            total_reward, step = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated) and (step < 500):
                action = agent.action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                step += 1
            total_reward_list.append(total_reward)
        average_reward_list += total_reward_list
    average_reward_list /= sims
    np.savetxt(result_dir_path + 'average_reward.csv', average_reward_list, delimiter=',')
    env.close()


def conv_simulation(sims, epis, env, agent, neighbor_frames, result_dir_path):
    average_reward_list = np.zeros(epis)
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        total_reward_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            discrete_state = env.reset()[0]
            frame = get_screen(env)
            frames = deque([frame]*neighbor_frames, maxlen=neighbor_frames)
            state = np.stack(frames, axis=1)[0,:]

            total_reward, step = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated) and (step < 500):
                action = agent.action(state)
                discrete_state, reward, terminated, truncated, info = env.step(action)

                frame = get_screen(env)
                frames.append(frame)
                next_state = np.stack(frames, axis=1)[0,:]
                
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                step += 1
            total_reward_list.append(total_reward)
        average_reward_list += total_reward_list
        np.savetxt(sim_dir_path + 'reward.csv', total_reward_list, delimiter=",")
        sub_plot(sim_dir_path, 'reward', total_reward_list)
    average_reward_list /= sims
    np.savetxt(result_dir_path + 'average_reward.csv', average_reward_list, delimiter=',')
    env.close()


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
    state = env.reset()
    print(f'state: {state}') #shape: (4,)
    env.close()