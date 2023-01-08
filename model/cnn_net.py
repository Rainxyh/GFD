import os
from collections import OrderedDict

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from argparse import ArgumentParser
from data.data_interface import DatasetInterface


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        self.conv4 = nn.Conv2d(48, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 256, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # x为32 * 64*64 * 1
        x = x.to(torch.float32)
        x = x.reshape(x.shape[0], x.shape[2], 64, 64)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    pass