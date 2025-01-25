import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

from wrapper.mountain_car_reward_wrapper import MountainCarRewardWrapper


def create_env(env_name, algo):
    # TODO: Atariアルゴリズムを採用，Convを削除，のちにAtari→Convに変更
    if 'Atari' in algo:
        env = gym.make(env_name, render_mode='rgb_array')
        env = AtariPreprocessing(env, frame_skip=4, grayscale_newaxis=False, scale_obs=True)
        env = FrameStackObservation(env, stack_size=4)
    else:
        env = gym.make(env_name, render_mode='rgb_array')
        if 'MountainCar' in env_name:
            env = MountainCarRewardWrapper(env)
    return env
