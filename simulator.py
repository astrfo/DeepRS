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


def sub_plot(sim_dir_path, name, thing):
    plt.plot(thing, label=name)
    plt.title(name)
    plt.xlabel('Episode')
    plt.savefig(sim_dir_path + f'{name}.png')
    plt.close()


def qvalue_plot(sim_dir_path, name, thing):
    plt.plot(thing, label=['left', 'down', 'right', 'up'])
    plt.title(name)
    plt.xlabel('Step')
    plt.legend()
    plt.savefig(sim_dir_path + f'{name}.png')
    plt.close()


def one_hot(discrete_state, state_space):
    one_hot_array = np.zeros(state_space)
    one_hot_array[discrete_state] = 1
    return one_hot_array


def frozenlake_position(env, discrete_state):
    X = discrete_state % env.ncol
    Y = discrete_state // env.ncol
    return env.desc[Y, X]


def simulation(sims, epis, env, agent, result_dir_path):
    average_reward_list = np.zeros(epis)
    average_goal_step_list = np.zeros(epis)
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        total_reward_list = []
        total_goal_step_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            discrete_state = env.reset()[0]
            state = one_hot(discrete_state, agent.policy.state_space)
            step, total_reward = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated) and (step < 500):
                action = agent.action(state, discrete_state)
                discrete_next_state, reward, terminated, truncated, info = env.step(action)
                next_state = one_hot(discrete_next_state, agent.policy.state_space)
                agent.update(state, action, reward, next_state, terminated)
                discrete_state = discrete_next_state
                state = next_state
                total_reward += reward
                step += 1
            agent.policy.EG_update(total_reward, step)
            total_reward_list.append(total_reward)
            total_goal_step_list.append(step)
        for i in range(agent.policy.state_space):
            np.savetxt(sim_dir_path + f'qvalue{i}.csv', agent.policy.q_list[i], delimiter=',')
            qvalue_plot(sim_dir_path, f'qvalue{i}', agent.policy.q_list[i])
        np.savetxt(sim_dir_path + 'reward.csv', total_reward_list, delimiter=',')
        sub_plot(sim_dir_path, 'reward', total_reward_list)
        np.savetxt(sim_dir_path + 'goal_step.csv', total_goal_step_list, delimiter=',')
        sub_plot(sim_dir_path, 'goal_step', total_goal_step_list)
        average_reward_list += total_reward_list
        average_goal_step_list += total_goal_step_list
    average_reward_list /= sims
    average_goal_step_list /= sims
    average_dir_path = result_dir_path + 'average/'
    os.makedirs(average_dir_path, exist_ok=True)
    np.savetxt(average_dir_path + 'average_reward.csv', average_reward_list, delimiter=',')
    sub_plot(average_dir_path, 'average_reward', average_reward_list)
    np.savetxt(average_dir_path + 'average_goal_step.csv', average_goal_step_list, delimiter=',')
    sub_plot(average_dir_path, 'average_goal_step', average_goal_step_list)
    env.close()


def conv_simulation(sims, epis, env, agent, neighbor_frames, result_dir_path):
    average_reward_list = np.zeros(epis)
    average_goal_step_list = np.zeros(epis)
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        total_reward_list = []
        total_goal_step_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            discrete_state = env.reset()[0]
            frame = get_screen(env)
            frames = deque([frame]*neighbor_frames, maxlen=neighbor_frames)
            state = np.stack(frames, axis=1)[0,:]

            step, total_reward = 0, 0
            terminated, truncated = False, False
            while not (terminated or truncated) and (step < 500):
                action = agent.action(state, discrete_state)
                discrete_next_state, reward, terminated, truncated, info = env.step(action)

                frame = get_screen(env)
                frames.append(frame)
                next_state = np.stack(frames, axis=1)[0,:]
                
                agent.update(state, action, reward, next_state, terminated)
                discrete_state = discrete_next_state
                state = next_state
                total_reward += reward
                step += 1
            agent.policy.EG_update(total_reward)
            total_reward_list.append(total_reward)
            total_goal_step_list.append(step)
        for i in range(agent.policy.state_space):
            np.savetxt(sim_dir_path + f'qvalue{i}.csv', agent.policy.q_list[i], delimiter=',')
            qvalue_plot(sim_dir_path, f'qvalue{i}', agent.policy.q_list[i])
        np.savetxt(sim_dir_path + 'reward.csv', total_reward_list, delimiter=',')
        sub_plot(sim_dir_path, 'reward', total_reward_list)
        np.savetxt(sim_dir_path + 'goal_step.csv', total_goal_step_list, delimiter=',')
        sub_plot(sim_dir_path, 'goal_step', total_goal_step_list)
        average_reward_list += total_reward_list
        average_goal_step_list += total_goal_step_list
    average_reward_list /= sims
    average_goal_step_list /= sims
    average_dir_path = result_dir_path + 'average/'
    os.makedirs(average_dir_path, exist_ok=True)
    np.savetxt(average_dir_path + 'average_reward.csv', average_reward_list, delimiter=',')
    sub_plot(average_dir_path, 'average_reward', average_reward_list)
    np.savetxt(average_dir_path + 'average_goal_step.csv', average_goal_step_list, delimiter=',')
    sub_plot(average_dir_path, 'average_goal_step', average_goal_step_list)
    env.close()


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
    state = env.reset()
    print(f'state: {state}') #shape: (4,)
    env.close()