import torch
import torch.nn as nn


class GruNet(nn.Module):
    """
            Parameters：
            - input_size: feature size
            - hidden_size: number of hidden units
            - output_size: number of output
            - num_layers: layers of LSTM to stack
        """

    def __init__(self, input_dim=8, hidden_num=128, output_dim=10, fc_hidden_dim=64, num_layers=5):
        super(GruNet, self).__init__()
        self.input_dim = input_dim
        # batch_first=False (seq_len, batch_size, input_size) batch_first=True (batch_size, seq_len, input_size)
        self.GRU_layer = nn.GRU(batch_first=True, input_size=input_dim, hidden_size=hidden_num, num_layers=num_layers)
        self.linear_1 = nn.Linear(hidden_num, fc_hidden_dim)
        self.linear_2 = nn.Linear(fc_hidden_dim, output_dim)

    # batch_first = True
    def forward(self, _x):
        _x = _x.to(torch.float32) # [32, 8, 4096]
        _x = _x.permute(0, 2, 1)
        _x = _x.reshape((_x.shape[0], -1, self.input_dim)) # (batch_size, seq_len, input_size)
        x, hidden_state = self.GRU_layer(_x) # [32, 256, 128]
        x = x[:, -1, :]
        x = self.linear_1(x)
        output = self.linear_2(x)
        # soft_output = torch.softmax(output, dim=-1)
        return output

    # batch_first = False
    # def forward(self, _x):
    #     x, hidden_state = self.GRU_layer(_x)  # _x is input, size (seq_len, batch_size, input_size)
    #     seq_len, batch_size, hidden_size = x.shape  # x is output, size (seq_len, batch, hidden_size)
    #     x = x.view(seq_len * batch_size, hidden_size)
    #     x = self.linear_1(x)
    #     x = self.linear_2(x)
    #     x = x.view(seq_len, batch_size, -1)
    #     return x
