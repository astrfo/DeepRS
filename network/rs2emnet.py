import torch.nn as nn
import torch.nn.functional as F


class RS2EMNet(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.embed = nn.Linear(hidden_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.embed(x))
        x = self.fc2(x)
        return x

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = self.embed(x)
        return x
