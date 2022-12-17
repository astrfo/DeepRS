import numpy as np
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


def simulation(sims, epis, env, agent, result_dir_path):
    average_reward_list = np.zeros(epis)
    for sim in range(sims):
        total_reward_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            total_reward = 0
            state = env.reset()[0]
            terminated, truncated = False, False
            while not(terminated or truncated):
                action = agent.action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                if total_reward >= 500:
                    break
            if epi % agent.policy.sync_interval == 0:
                agent.policy.sync_model()
            total_reward_list.append(total_reward)
        average_reward_list += total_reward_list
    average_reward_list /= sims
    np.savetxt(result_dir_path + 'average_reward.csv', average_reward_list, delimiter=',')
    env.close()


def conv_simulation(sims, epis, env, agent, neighbor_frames, result_dir_path):
    average_reward_list = np.zeros(epis)
    for sim in range(sims):
        total_reward_list = []
        agent.reset()
        for epi in tqdm(range(epis), 
                        bar_format='{desc}:{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} episode, {elapsed}/{remaining}, {rate_fmt}{postfix}',
                        desc=f'[{sys._getframe().f_code.co_name}_{agent.policy.__class__.__name__} {sim+1}/{sims} agent]'):
            total_reward = 0
            env.reset()
            frame = get_screen(env)
            frames = deque([frame]*neighbor_frames, maxlen=neighbor_frames)
            state = np.stack(frames, axis=1)[0,:]

            terminated, truncated = False, False
            while not(terminated or truncated):
                action = agent.action(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                frame = get_screen(env)
                frames.append(frame)
                next_state = np.stack(frames, axis=1)[0,:]
                
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                if total_reward >= 500:
                    break
            if epi % agent.policy.sync_interval == 0:
                agent.policy.sync_model()
            total_reward_list.append(total_reward)
        average_reward_list += total_reward_list
    average_reward_list /= sims
    np.savetxt(result_dir_path + 'average_reward.csv', average_reward_list, delimiter=',')
    env.close()


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
    state = env.reset()
    print(f'state: {state}') #shape: (4,)
    env.close()

