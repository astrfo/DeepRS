import numpy as np
import matplotlib.pyplot as plt
import os

def process_and_save_sma(csv_path, sma_window, output_dir):
    data = np.loadtxt(csv_path, delimiter=',')
    sma = np.convolve(data, np.ones(sma_window)/sma_window, mode='valid')
    
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(output_dir + 'loss_sma.csv', sma, delimiter=',')
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(sma_window - 1, len(data)), sma, label=f'loss')
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.xlim(-1, len(data)+1)
    plt.legend()
    plt.savefig(output_dir + 'loss_sma.png')
    plt.close()

if __name__ == '__main__':
    csv_path = 'log/CartPole-v1/RSRSAlephQEpsRASChoiceCentroidDQN/202412091421/1/loss_epi200.csv'
    output_dir = 'sma_log/'
    sma_window = 50
    
    process_and_save_sma(csv_path, sma_window, output_dir)
