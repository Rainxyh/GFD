import os
from collections import OrderedDict

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from argparse import ArgumentParser
from data.data_interface import DatasetInterface


class CnngruNet(nn.Module):
    def __init__(self):
        super(CnngruNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.GRU_layer = nn.GRU(batch_first=True, input_size=128, hidden_size=128, num_layers=3,
                                bidirectional=False)
        self.fc1 = nn.Linear(128, 64)  # 双向
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.to(torch.float32)  # [32, 8, 4096]
        x = x.reshape(x.shape[0], x.shape[1], 64, 64)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.reshape(x.shape[0], -1, 128)
        x, hidden_state = self.GRU_layer(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    pass