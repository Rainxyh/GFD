import os
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from argparse import ArgumentParser
from data.data_interface import DatasetInterface


class Flatten(nn.Module):
    # 把输入reshape成（batch_size,dim_length）
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class OnedcnnNet(nn.Module):
    def __init__(self, layer_num=5, in_channels=8, out_channels=[16, 32, 64, 64, 64], kernel_sizes=[64, 3, 3, 3, 3],
                 strides=[16, 1, 1, 1, 1], conv_paddings=[0, 'same', 'same', 'same', 'valid'], batchnorm_num_features=[16, 32, 64, 64, 64],
                 pool_kernel_sizes=[2, 2, 2, 2, 2], fc_hidden_dim=100, class_num=10,
                        ):
        super(OnedcnnNet, self).__init__()
        self.layer_num = layer_num
        self.feature = None
        self.conv = nn.ModuleDict()
        for i in range(self.layer_num):
            self.conv['layer'+str(i)] = nn.Sequential(OrderedDict(
                [
                    ("conv"+str(i), nn.Conv1d(in_channels=in_channels if i == 0 else out_channels[i-1], out_channels=out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=conv_paddings[i] if i != self.layer_num else conv_paddings[-1])),
                    ("bn"+str(i), nn.BatchNorm1d(num_features=batchnorm_num_features[i])),
                    ("active"+str(i), nn.ReLU()),
                    ("pool"+str(i), nn.MaxPool1d(kernel_size=pool_kernel_sizes[i]))
                ]
            ))

        self.flatten = Flatten()

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(64, 32)),
                    ("fc2", nn.Linear(32, class_num)),
                ]
            )
        )

    def forward(self, x_):
        x_ = x_.to(torch.float32)
        x_ = x_.reshape(x_.shape[0], -1, x_.shape[2])
        x = list()
        x.append(x_)
        for i in range(self.layer_num):
            t = self.conv['layer'+str(i)](x[i])
            x.append(t)
        # x = self.conv['layer0'](x)
        # x = self.conv['layer1'](x)
        # x = self.conv['layer2'](x)
        # x = self.conv['layer3'](x)
        # x = self.conv['layer4'](x)
        self.feature = self.flatten(x[-1])

        output = self.fc(self.feature)
        # soft_output = torch.softmax(output, dim=-1)
        return output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--dataset', default='gearbox_data', type=str)
    parser.add_argument('--data_dir', default=r'E:/JetBrains/PycharmProjects/GFD/data/gearbox'.format(os.getcwd()).replace("\\", "/"), type=str)
    parser.add_argument('--ref_dir', default=r'E:/JetBrains/PycharmProjects/GFD/data/ref'.format(os.getcwd()).replace("\\", "/"), type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--standard_scalar', default=True, type=bool)
    parser.add_argument('--class_num', default=20, type=int)
    parser.add_argument('--train_mode', default=True, type=bool)
    parser.add_argument('--sample_number', default=1000, type=int)
    parser.add_argument('--sample_length', default=512, type=int)

    args = parser.parse_args()
    dataset_module = DatasetInterface(**vars(args))  # 数据集模型
    dataset_module.setup(stage='fit')
    train_loader = dataset_module.train_dataloader()

    net = CnnNet(layer_num=1, in_channels=4)

    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs[0].reshape(32, 4, -1).to(torch.float32)
        print(inputs.shape, labels.shape)
        # (batchsize, 4, -1) (batchsize, 20,)
        outputs = net.forward(inputs)
        print(outputs.shape)
        exit()


