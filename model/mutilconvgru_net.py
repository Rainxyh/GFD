import os

import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable

from data.gearbox_data import GearboxData
from data.wavelet_transform import multi_sensor_signal2cube


class Flatten(nn.Module):
    # 把输入reshape成（batch_size,dim_length）
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype=torch.FloatTensor):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # same padding
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim+hidden_dim,
                                    out_channels=2*self.hidden_dim,  # 对应于GRU中的更新门与重制门
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_candidate = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                         out_channels=self.hidden_dim,  # for candidate neural memory
                                         kernel_size=kernel_size,
                                         padding=self.padding,
                                         bias=self.bias)

        self.conv_follow_up = nn.Sequential(OrderedDict(
            [
                ("bn", nn.BatchNorm2d(num_features=self.hidden_dim)),
                ("active", nn.ReLU()),
                ("pool", nn.MaxPool2d(kernel_size=2))
            ]
        ))

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, self.height, self.width).type(self.dtype)
        # return torch.zeros(batch_size, self.hidden_dim, self.height, self.width)

    def forward(self, input_tensor, h_cur):
        """
        :param self:   除了深度不相同 其余维度必须相同 将深度进行拼接
        :param input_tensor: (b, c, h, w)  # 当前输入张量  （1， 4， 64， 64）
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)  # 当前隐含层状态   （1， 16， 64， 64）
            current hidden and cell states respectively
        :return: h_next,  # 下一时刻隐含层状态
            next hidden state
        """
        # print(input_tensor.is_cuda, h_cur.is_cuda)  # false true
        # print('Is model on gpu: ', next(self.conv_gates.parameters()).is_cuda)
        # print(input_tensor.shape, h_cur.shape)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # 将x与h在深度维度进行拼接（1， 20， 64， 64）
        combined_conv = self.conv_gates(combined)  # 集合卷积操作 生成的张量具有两个门的深度 该层先不使用池化（1， 32， 64， 64）

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)  # （1， 16， 64， 64）* 2
        reset_gate = torch.sigmoid(gamma)  # 重制门（1， 16， 64， 64）
        update_gate = torch.sigmoid(beta)  # 更新门（1， 16， 64， 64）

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)  # 对隐含层状态使用重制门进行更新后再在深度维度进行拼接
        cc_cnm = self.conv_candidate(combined)  # （1， 16， 64， 64）
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm

        h_next_pooled = self.conv_follow_up(h_next)  # 进行BN、激活、池化等操作
        return h_next_pooled, h_next  # 池化后的 也就是下一层GRU的输入， 未经过池化的 也就是当前层GRU的输入


class MutilconvgruNet(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias,
                 num_layers, dtype, batch_first, return_all_layers, class_num):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(MutilconvgruNet, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):  # 循环对每一层网络进行构建 层次越深通过池化后深层长度与宽度减半
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]  # 除第层外其余层次的输入深度等于上一层的输出深度
            cell_list.append(ConvGRUCell(input_size=(self.height//pow(2, i), self.width//pow(2, i)),  # 张量长宽
                                         input_dim=cur_input_dim,  # 张量深度
                                         hidden_dim=self.hidden_dim[i],  # 隐含层深度
                                         kernel_size=self.kernel_size[i],  # 卷积核大小
                                         bias=self.bias,  # 是否偏置
                                         dtype=self.dtype))  # 数据类型

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

        self.flatten = Flatten()
        self.feature_length = self.height*self.width//pow(2, self.num_layers*2)*hidden_dim[-1]
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(self.feature_length, self.feature_length//2)),
                    ("fc2", nn.Linear(self.feature_length//2, class_num)),
                ]
            )
        )

    def forward_conv(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t, b, c, h, w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_cur_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        # print(input_tensor.shape)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_cur_state_list = []
        last_next_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):  # 遍历每一层
            cur_layer_hidden = hidden_state[layer_idx]  # 当前层初始隐藏层状态
            next_layer_hidden = hidden_state[layer_idx]  # 需要传递到下一层的初始隐藏层状态
            next_layer_hidden_list = []  # 当前层内部隐含状态列表
            for t in range(seq_len):  # 遍历单层中的每个时间步
                """input current hidden and cell state then compute the next hidden and cell state
                through ConvLSTMCell forward function"""
                # 输入当前隐藏和单元状态，然后通过 ConvLSTMCell 前向函数计算下一个隐藏和单元状态
                next_layer_hidden, cur_layer_hidden = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],  # (b, c, h, w) (1, 4, 8 ,8)
                                              h_cur=cur_layer_hidden)  # 同一layer中 通过h状态向后传递 进行迭代， 池化后的hidden状态作为下一层的输入
                next_layer_hidden_list.append(next_layer_hidden)  # 将池化后的hidden状态保存起来 供下一层作为输入

            layer_output = torch.stack(next_layer_hidden_list, dim=1)  # 在深度维度进行拼接
            cur_layer_input = layer_output  # 下一层的输入x 对应上一层的隐含层状态h

            layer_output_list.append(layer_output)  # 各层的 "池化后的隐含层状态列表" 所构成的列表
            last_cur_state_list.append([cur_layer_hidden])  # 各层 "未经过池化的末端时间步隐含层状态" 构成的列表
            last_next_state_list.append([next_layer_hidden])  # 各层 "经过池化后的末端时间步隐含层状态" 构成的列表

        if not self.return_all_layers:  # 只返回最后一层
            layer_output_list = layer_output_list[-1:]  # 最后一层的 "池化后的隐含层状态列表"
            last_cur_state_list = last_cur_state_list[-1:]  # 最后一层的 "未经过池化的末端时间步隐含层状态"
            last_next_state_list = last_next_state_list[-1:]  # 最后一层的 "经过池化后的末端时间步隐含层状态"

        return last_next_state_list

    def forward(self, input_tensor, hidden_state=None):
        input_tensor = input_tensor.type(self.dtype)
        last_next_state_list = self.forward_conv(input_tensor, hidden_state)
        if not self.return_all_layers:
            last_state = last_next_state_list[0][0]
            feature = self.flatten(last_state)
            self.feature_length = feature.shape[0]
            output = self.fc(feature)
            # soft_output = torch.softmax(output, dim=-1)
            return output

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    n = 5
    height = width = pow(2, n)
    channels = 4
    hidden_dim = 16
    kernel_size = (3, 3)  # kernel size for two stacked hidden layer
    num_layers = 3  # number of stacked hidden layer
    model = MutilconvgruNet(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=torch.FloatTensor,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)

    data = GearboxData()
    data, label = data.__getitem__(0)
    data_noise = data
    print(data.shape)
    exit()

