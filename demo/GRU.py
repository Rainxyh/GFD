from torch.utils.data import Dataset
import numpy as np
import torch
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class GRU_network(nn.Module):
    """
            Parameters：
            - input_size: feature size
            - hidden_size: number of hidden units
            - output_size: number of output
            - num_layers: layers of LSTM to stack
        """

    def __init__(self, input_dim, hidden_num=1, output_dim=1, neurons=[16]):
        super(GRU_network, self).__init__()
        # batch_first=False (seq_len, batch_size, input_size) batch_first=True (batch_size, seq_len, input_size)
        self.GRU_layer = nn.GRU(input_size=input_dim, hidden_size=hidden_num, num_layers=len(neurons))
        self.linear_1 = nn.Linear(hidden_num, neurons[0])
        self.linear_2 = nn.Linear(neurons[0], output_dim)

    def forward(self, _x):
        x, hidden_state = self.GRU_layer(_x)  # _x is input, size (seq_len, batch, input_size)
        seq_len, batch_size, hidden_size = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(seq_len * batch_size, hidden_size)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = x.view(seq_len, batch_size, -1)
        return x
