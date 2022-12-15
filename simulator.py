import numpy as np

def simulation(sims, epis, aleph, env, result_dir_path):
    average_reward_list = np.zeros(epis)
    for sim in range(sims):
        total_reward_list = []
        for epi in range(epis):
            total_reward = 0
            observation = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                observation, reward, done, _, info = env.step(action)
                total_reward += reward
            total_reward_list.append(total_reward)
        average_reward_list += total_reward_list
    average_reward_list /= sims
    np.savetxt(result_dir_path + 'average_reward.csv', average_reward_list, delimiter=",")