import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from GRU import GRU_network

from datetime import datetime
from torch.utils.data import DataLoader
from utils.utils import MyDataset
torch.set_default_tensor_type(torch.DoubleTensor)

batch_size = 32
max_epoch = 100
num_classes = 2
lr_init = 0.001

path = r'data/Gearbox'

time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')

train_data = MyDataset(data_path=path, model=0)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

net = GRU_network(input_dim=864, hidden_num=32, output_dim=num_classes, neurons=[16])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)

acc = []
epoch_acc = []
for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    # scheduler.step()  # 更新学习率

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.reshape(-1, inputs.shape[0], inputs.shape[1])
        labels = torch.argmax(labels, dim=1).view(-1)

        outputs = net(inputs).squeeze(0)
        # outputs = torch.softmax(outputs, dim=1)
        # softmax_function = nn.Softmax(dim=1)
        # outputs = softmax_function(outputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, dim=1)  # value index
        total += labels.size(0)

        correct += predicted.eq(labels.view_as(predicted)).squeeze().sum().numpy()
        loss_sigma += loss.item()
        acc.append(correct / total)

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, acc[-1]))
    epoch_acc.append(np.mean(acc[:-32]))
            # # 记录训练loss
            # writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # # 记录learning rate
            # writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # # 记录Accuracy
            # writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)
np.save("Acc", np.array(epoch_acc))