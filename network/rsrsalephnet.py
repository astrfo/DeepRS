import torch.nn as nn
import torch.nn.functional as F


class RSRSAlephNet(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.embed = nn.Linear(hidden_size, embedding_size)
        self.head_q = nn.Linear(embedding_size, output_size)
        self.head_aleph = nn.Linear(embedding_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.embed(x))
        q_values = self.head_q(x)
        aleph_s = self.head_aleph(x)
        return q_values, aleph_s

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = self.embed(x)
        return x
