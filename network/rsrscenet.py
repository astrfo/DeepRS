import torch.nn as nn
import torch.nn.functional as F


class RSRSCENet(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.embed = nn.Linear(hidden_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, output_size)
        self.confidence = nn.Linear(embedding_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.embed(x))
        q_values = self.fc2(x)
        confidence = self.confidence(x)
        return q_values, confidence

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = self.embed(x)
        return x
