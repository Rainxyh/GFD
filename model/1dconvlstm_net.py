import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell. 初始化ConvLSTM单元

        Parameters
        ----------
        input_dim: int  输入维度
            Number of channels of input tensor.  输入张量的通道数
        hidden_dim: int
            Number of channels of hidden state.  隐藏状态的通道数
        kernel_size: (int, int)
            Size of the convolutional kernel.  卷积核的大小
        bias: bool
            Whether or not to add the bias.  是否添加偏移量

        In the case more layers are present but a single value is provided, this is replicated for all the layers.
        For example, in the following snippet each of the three layers has a different hidden dimension
        but the same kernel size.
        如果存在更多层但提供了单个值，则会为所有层复制此值。
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # ✖4为了将输出的通道数分为i,f,o,g四个部分
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis 在相应维度进行拼接

        # 32*80*4 -> 32*64*5
        combined = combined.permute(0, 2, 1)
        combined_conv = self.conv(combined)
        combined_conv = combined_conv.permute(0, 2, 1)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)  # 手动计算LSTM不同门控单元的数值
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g  # 记忆单元状态
        h_next = o * torch.tanh(c_next)  # 隐藏单元状态

        return h_next, c_next

    def init_hidden(self, batch_size, seq_size):  # 权重初始化为0表明初始状态对未来完全无先验知识
        return (torch.zeros(batch_size, self.hidden_dim, seq_size, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, seq_size, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input  输入张量的通道数
        hidden_dim: Number of hidden channels  隐藏状态的通道数
        kernel_size: Size of kernel in convolutions  卷积核的大小
        num_layers: Number of LSTM layers stacked on each other  彼此堆叠的LSTM层数
        batch_first: Whether or not dimension 0 is the batch or not  第一维度是否为batch
        bias: Bias or no bias in Convolution  卷积层的偏置值
        return_all_layers: Return the list of computations for all layers  返回所有层的计算结果
        Note: Will do same padding.  将做相同的填充

    Input:
        A tensor of size (B, T, C, H, W) or (T, B, C, H, W)
        B batch size, T length, C channel, H row, W col
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output  长度为T的输出列表构成的列表
            1 - last_state_list is the list of last states  最终状态构成的列表
                    each element of the list is a tuple (h, c) for hidden state and memory  每个元素由一个元组构成(隐藏状态， 记忆)
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(input_dim=64, hidden_dim=16, kernel_size=3, num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):  # 对每一层构建对应参数控制的网络层
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]  # 当前层的输入 如果不为首次输入 则使用前一层的输出作为输入

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:  # 如果batch size不为第一维度
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        b, _, _, h = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             seq_size=h)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # 序列长度
        cur_layer_input = input_tensor  # 输出张量

        for layer_idx in range(self.num_layers):  # 遍历每一层

            h, c = hidden_state[layer_idx]  # 第index层隐藏状态h与记忆状态c
            output_inner = []
            for t in range(seq_len):  # 对序列长度进行遍历
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                 cur_state=[h, c])  # 下一步隐藏状态与记忆状态继续迭代
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)  # 将隐藏状态h在相应维度拼接为一个矩阵
            cur_layer_input = layer_output  # 供下一层网络继续迭代

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:  # 如果不需要全部输出 则只输出最后一层网络信息
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, seq_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, seq_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):  # 将单卷积核转换为大小等同于网络层数的列表
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    x = torch.rand((32, 10, 64, 4))
    convlstm = ConvLSTM(input_dim=64, hidden_dim=16, kernel_size=4, num_layers=1, batch_first=True, bias=True,
                           return_all_layers=False)
    _, last_states = convlstm(x)
    h = last_states[0][0]  # 0 for layer index, 0 for h index
    print(h)
    print(h.size())

