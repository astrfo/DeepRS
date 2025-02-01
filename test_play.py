import torch
import gymnasium as gym
import ale_py
import imageio

from network.conv_qnet import ConvQNet


if __name__ == '__main__':
    model_path = 'xxx.pth'

    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    action_space = env.action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_net = ConvQNet(input_size=None, hidden_size=None, output_size=action_space).to(device)
    target_net.load_state_dict(torch.load(model_path, map_location=device))

    video_filename = 'breakout_simulation.mp4'
    frames = []

    state, info = env.reset()
    state_frames = torch.zeros((1, 4, 84, 84), dtype=torch.float32, device=device)
    next_state, reward, terminated, truncated, info = env.step(1)

    next_state_frame = next_state / 255.0
    next_state_frame = torch.tensor(next_state_frame, dtype=torch.float32, device=device).unsqueeze(0)
    state_frames = next_state_frame

    terminated, truncated = False, False
    while not (terminated or truncated):
        action = target_net(state_frames).argmax(dim=1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

        next_state_frame = next_state / 255.0
        next_state_frame = torch.tensor(next_state_frame, dtype=torch.float32, device=device).unsqueeze(0)
        state_frames = next_state_frame

    with imageio.get_writer(video_filename, fps=30) as video:
        for frame in frames:
            video.append_data(frame)

    print(f'Simulation video saved as {video_filename}')
