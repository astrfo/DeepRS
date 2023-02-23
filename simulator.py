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


def one_hot(discrete_state, state_space):
    one_hot_array = np.zeros(state_space)
    one_hot_array[discrete_state] = 1
    return one_hot_array


def frozenlake_position(env, discrete_state):
    X = discrete_state % env.ncol
    Y = discrete_state // env.ncol
    return env.desc[Y, X]


def simulation(sims, epis, env, agent, result_dir_path, max_step):
    average_reward_list = np.zeros(epis)
    average_goal_step_list = np.zeros(epis)
    average_fall_hole_list = np.zeros(epis)
    average_epi100_reward_list = np.zeros(epis//100)
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        total_reward_list = []
        total_goal_step_list = []
        total_fall_hole_list = []
        epi100_reward_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            if (epi+1) % 100 == 0:
                discrete_state = env.reset()[0]
                state = one_hot(discrete_state, agent.policy.state_space)
                step, total_reward, goal_step, fall_hole = 0, 0, np.nan, 0
                terminated, truncated = False, False
                while not (terminated or truncated) and (step < max_step):
                    action = agent.greedy_action(state, discrete_state)
                    discrete_next_state, reward, terminated, truncated, info = env.step(action)
                    letter = frozenlake_position(env, discrete_next_state)
                    if letter == b'G':
                        reward = +1
                        goal_step = step
                    if letter == b'H':
                        reward = -1
                        fall_hole += 1
                    # if letter == b'G':
                    #     if discrete_next_state == 2: reward = 8
                    #     elif discrete_next_state == 6: reward = 1
                    #     elif discrete_next_state == 18: reward = 3
                    #     elif discrete_next_state == 26: reward = 5
                    #     elif discrete_next_state == 54: reward = 6
                    #     elif discrete_next_state == 62: reward = 4
                    #     elif discrete_next_state == 74: reward = 2
                    #     elif discrete_next_state == 78: reward = 7
                    #     goal_step = step
                    next_state = one_hot(discrete_next_state, agent.policy.state_space)
                    discrete_state = discrete_next_state
                    state = next_state
                    total_reward += reward
                    step += 1
                epi100_reward_list.append(total_reward)
            discrete_state = env.reset()[0]
            state = one_hot(discrete_state, agent.policy.state_space)
            step, total_reward, goal_step, fall_hole = 0, 0, np.nan, 0
            terminated, truncated = False, False
            while not (terminated or truncated) and (step < max_step):
                action = agent.action(state, discrete_state)
                discrete_next_state, reward, terminated, truncated, info = env.step(action)
                letter = frozenlake_position(env, discrete_next_state)
                if letter == b'G':
                    reward = +1
                    goal_step = step
                if letter == b'H':
                    reward = -1
                    fall_hole += 1
                # if letter == b'G':
                #     if discrete_next_state == 2: reward = 8
                #     elif discrete_next_state == 6: reward = 1
                #     elif discrete_next_state == 18: reward = 3
                #     elif discrete_next_state == 26: reward = 5
                #     elif discrete_next_state == 54: reward = 6
                #     elif discrete_next_state == 62: reward = 4
                #     elif discrete_next_state == 74: reward = 2
                #     elif discrete_next_state == 78: reward = 7
                #     goal_step = step
                next_state = one_hot(discrete_next_state, agent.policy.state_space)
                agent.update(state, action, reward, next_state, terminated)
                discrete_state = discrete_next_state
                state = next_state
                total_reward += reward
                step += 1
            agent.policy.EG_update(total_reward, step)
            total_reward_list.append(total_reward)
            total_goal_step_list.append(goal_step)
            total_fall_hole_list.append(fall_hole)
        average_reward_list = plus_csv_plot(average_reward_list, total_reward_list, sim_dir_path, 'reward')
        average_goal_step_list = plus_csv_plot(average_goal_step_list, total_goal_step_list, sim_dir_path, 'goal_step')
        average_fall_hole_list = plus_csv_plot(average_fall_hole_list, total_fall_hole_list, sim_dir_path, 'fall_hole')
        average_epi100_reward_list = plus_csv_plot(average_epi100_reward_list, epi100_reward_list, sim_dir_path, 'epi100_reward')
        # for i in range(agent.policy.state_space):
        #     np.savetxt(sim_dir_path + f'qvalue{i}.csv', agent.policy.q_list[i], delimiter=',')
        #     qvalue_plot(sim_dir_path, f'qvalue{i}', agent.policy.q_list[i])
        # np.savetxt(sim_dir_path + 'batchreward.csv', agent.policy.batch_reward_list, delimiter=',')
        # sub_plot(sim_dir_path, f'batchreward', agent.policy.batch_reward_list)
        # np.savetxt(sim_dir_path + f'pi.csv', agent.policy.pi_list, delimiter=',') #DQNのときはコメント
        # pi_plot(sim_dir_path, f'pi', agent.policy.pi_list) #DQNのときはコメント
        np.savetxt(sim_dir_path + f'eg.csv', agent.policy.E_G_list, delimiter=',') #DQNのときはコメント
        np.savetxt(sim_dir_path + f'alephg.csv', agent.policy.aleph_G_list, delimiter=',') #DQNのときはコメント
        eg_alephg_plot(sim_dir_path, agent.policy.E_G_list, agent.policy.aleph_G_list)
    divide_csv_plot(average_reward_list, result_dir_path, 'reward', sims)
    divide_csv_plot(average_goal_step_list, result_dir_path, 'goal_step', sims)
    divide_csv_plot(average_fall_hole_list, result_dir_path, 'fall_hole', sims)
    divide_csv_plot(average_epi100_reward_list, result_dir_path, 'epi100_reward', sims)
    env.close()


def conv_simulation(sims, epis, env, agent, neighbor_frames, result_dir_path, max_step):
    average_reward_list = np.zeros(epis)
    average_goal_step_list = np.zeros(epis)
    average_fall_hole_list = np.zeros(epis)
    average_epi100_reward_list = np.zeros(epis//100)
    for sim in range(sims):
        sim_dir_path = result_dir_path + f'{sim+1}/'
        os.makedirs(sim_dir_path, exist_ok=True)
        total_reward_list = []
        total_goal_step_list = []
        total_fall_hole_list = []
        epi100_reward_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            if (epi+1) % 100 == 0:
                discrete_state = env.reset()[0]
                frame = get_screen(env)
                frames = deque([frame]*neighbor_frames, maxlen=neighbor_frames)
                state = np.stack(frames, axis=1)[0,:]
                step, total_reward, goal_step, fall_hole = 0, 0, np.nan, 0
                terminated, truncated = False, False
                while not (terminated or truncated) and (step < max_step):
                    action = agent.greedy_action(state, discrete_state)
                    discrete_next_state, reward, terminated, truncated, info = env.step(action)
                    letter = frozenlake_position(env, discrete_next_state)
                    if letter == b'G':
                        reward = +1
                        goal_step = step
                    if letter == b'H':
                        reward = -1
                        fall_hole += 1
                    # if letter == b'G':
                    #     if discrete_next_state == 2: reward = 8
                    #     elif discrete_next_state == 6: reward = 1
                    #     elif discrete_next_state == 18: reward = 3
                    #     elif discrete_next_state == 26: reward = 5
                    #     elif discrete_next_state == 54: reward = 6
                    #     elif discrete_next_state == 62: reward = 4
                    #     elif discrete_next_state == 74: reward = 2
                    #     elif discrete_next_state == 78: reward = 7
                    #     goal_step = step
                    frame = get_screen(env)
                    frames.append(frame)
                    next_state = np.stack(frames, axis=1)[0,:]
                    discrete_state = discrete_next_state
                    state = next_state
                    total_reward += reward
                    step += 1
                epi100_reward_list.append(total_reward)
            discrete_state = env.reset()[0]
            frame = get_screen(env)
            frames = deque([frame]*neighbor_frames, maxlen=neighbor_frames)
            state = np.stack(frames, axis=1)[0,:]
            step, total_reward, goal_step, fall_hole = 0, 0, np.nan, 0
            terminated, truncated = False, False
            while not (terminated or truncated) and (step < max_step):
                action = agent.action(state, discrete_state)
                discrete_next_state, reward, terminated, truncated, info = env.step(action)
                letter = frozenlake_position(env, discrete_next_state)
                if letter == b'G':
                    reward = +1
                    goal_step = step
                if letter == b'H':
                    reward = -1
                    fall_hole += 1
                # if letter == b'G':
                #     if discrete_next_state == 2: reward = 8
                #     elif discrete_next_state == 6: reward = 1
                #     elif discrete_next_state == 18: reward = 3
                #     elif discrete_next_state == 26: reward = 5
                #     elif discrete_next_state == 54: reward = 6
                #     elif discrete_next_state == 62: reward = 4
                #     elif discrete_next_state == 74: reward = 2
                #     elif discrete_next_state == 78: reward = 7
                #     goal_step = step
                frame = get_screen(env)
                frames.append(frame)
                next_state = np.stack(frames, axis=1)[0,:]
                agent.update(state, action, reward, next_state, terminated)
                discrete_state = discrete_next_state
                state = next_state
                total_reward += reward
                step += 1
            agent.policy.EG_update(total_reward, step)
            total_reward_list.append(total_reward)
            total_goal_step_list.append(goal_step)
            total_fall_hole_list.append(fall_hole)
        average_reward_list = plus_csv_plot(average_reward_list, total_reward_list, sim_dir_path, 'reward')
        average_goal_step_list = plus_csv_plot(average_goal_step_list, total_goal_step_list, sim_dir_path, 'goal_step')
        average_fall_hole_list = plus_csv_plot(average_fall_hole_list, total_fall_hole_list, sim_dir_path, 'fall_hole')
        average_epi100_reward_list = plus_csv_plot(average_epi100_reward_list, epi100_reward_list, sim_dir_path, 'epi100_reward')
        # for i in range(agent.policy.state_space):
        #     np.savetxt(sim_dir_path + f'qvalue{i}.csv', agent.policy.q_list[i], delimiter=',')
        #     qvalue_plot(sim_dir_path, f'qvalue{i}', agent.policy.q_list[i])
        # np.savetxt(sim_dir_path + 'batchreward.csv', agent.policy.batch_reward_list, delimiter=',')
        # sub_plot(sim_dir_path, f'batchreward', agent.policy.batch_reward_list)
        # np.savetxt(sim_dir_path + f'pi.csv', agent.policy.pi_list, delimiter=',') #DQNのときはコメント
        # pi_plot(sim_dir_path, f'pi', agent.policy.pi_list) #DQNのときはコメント
        np.savetxt(sim_dir_path + f'eg.csv', agent.policy.E_G_list, delimiter=',') #DQNのときはコメント
        np.savetxt(sim_dir_path + f'alephg.csv', agent.policy.aleph_G_list, delimiter=',') #DQNのときはコメント
        eg_alephg_plot(sim_dir_path, agent.policy.E_G_list, agent.policy.aleph_G_list)
    divide_csv_plot(average_reward_list, result_dir_path, 'reward', sims)
    divide_csv_plot(average_goal_step_list, result_dir_path, 'goal_step', sims)
    divide_csv_plot(average_fall_hole_list, result_dir_path, 'fall_hole', sims)
    divide_csv_plot(average_epi100_reward_list, result_dir_path, 'epi100_reward', sims)
    env.close()


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
    state = env.reset()
    print(f'state: {state}') #shape: (4,)
    env.close()