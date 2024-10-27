import torch.nn as nn
import torch.nn.functional as F


class ConvRNDNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, neighbor_frames):
        super().__init__()
        C, H, W = input_size
        self.kernel_sizes = [8, 4, 3]
        self.strides = [4, 2, 1]

        self.conv1 = nn.Conv2d(in_channels=C*neighbor_frames, out_channels=16, kernel_size=self.kernel_sizes[0], stride=self.strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_sizes[1], stride=self.strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_sizes[2], stride=self.strides[2])
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(W, n_layer=3)
        convh = self.conv2d_size_out(H, n_layer=3)
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.embed = nn.Linear(512, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.embed(x)
        x = self.head(x)
        return x

    def conv2d_size_out(self, size, n_layer):
        cnt = 0
        size_out = size
        while cnt < n_layer:
            size_out = (size_out - self.kernel_sizes[cnt]) // self.strides[cnt] + 1
            cnt += 1
        return size_out
