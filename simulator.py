import numpy as np
from tqdm import tqdm

def simulation(sims, epis, env, agent, result_dir_path):
    average_reward_list = np.zeros(epis)
    for sim in range(sims):
        total_reward_list = []
        agent.reset()
        for epi in tqdm(range(epis)):
            total_reward = 0
            state = env.reset()[0]
            terminated, truncated = False, False
            while not(terminated or truncated):
                action = agent.action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
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

