import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 32
        self.hidden_size = 128
        self.device = torch.device('cpu')
        #state, action_spaceが必要
        self.model = QNet(input_size=0, hidden_size=self.hidden_size, output_size=0)
        self.model.to(self.device)
        self.model_target = QNet(input_size=0, hidden_size=self.hidden_size, output_size=0)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def reset(self):
        pass

    def update(self):
        pass

    def temporal_difference(self):
        pass


if __name__ == '__main__':
    print('started policy')
    qnet = QNet(input_size=2, hidden_size=3, output_size=2)
    print('finished policy')