import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4 * 29 * 29, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        print("input       :", x.shape)

        x = F.relu(self.conv1(x))
        print("after conv1 :", x.shape)

        x = F.max_pool2d(x, 2)
        print("after pool1 :", x.shape)

        x = F.relu(self.conv2(x))
        print("after conv2 :", x.shape)

        x = F.max_pool2d(x, 2)
        print("after pool2 :", x.shape)

        x = self.dropout(x)
        print("after drop  :", x.shape)

        x = x.view(x.shape[0], -1)
        print("after flat  :", x.shape)

        x = F.relu(self.fc1(x))
        print("after fc1   :", x.shape)

        x = self.fc2(x)
        print("after fc2   :", x.shape)

        return x


if __name__ == "__main__":
    model = SimpleCNN()
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    print("final output:", y.shape)
