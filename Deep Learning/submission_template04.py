import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Сверточные слои и пулинг
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Вычисляем размер входа в полносвязный слой
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # Исправленный размер входа в fc1
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Сверточные и пулинг слои
        x = self.pool1(F.relu(self.conv1(x)))  # conv1 -> ReLU -> maxpool1
        x = self.pool2(F.relu(self.conv2(x)))  # conv2 -> ReLU -> maxpool2

        # Преобразуем в вектор
        x = self.flatten(x)

        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
