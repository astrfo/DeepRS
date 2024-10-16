import torch.nn as nn
import torch.nn.functional as F


class DuelingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, output_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        advantage = self.advantage(x)
        v = self.v(x)
        return advantage + v
