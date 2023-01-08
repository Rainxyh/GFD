import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import pandas as pd
from model import ModelInterface
from model.onedcnn_net import OnedcnnNet
from PIL import Image


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def target_category_loss(x, category_index, nb_classes):
    return torch.mul(x, F.one_hot(category_index, nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        # print('output.size()', output.size())
        return output[target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            output = output.squeeze()
            target_category = np.argmax(output.cpu().data.numpy())
            # print('output:', output)
            # print('target_category:', target_category)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        # print('loss', loss)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        # print('activations', activations)
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        #weights = np.mean(grads, axis=(0))
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
             cam += w * activations[i, :]

        # print('cam.shape', cam.shape)
        # print('input_tensor.shape', input_tensor.shape)
        cam = np.interp(np.linspace(0, cam.shape[0], input_tensor.shape[2]), np.linspace(0, cam.shape[0], cam.shape[0]), cam)   #Change it to the interpolation algorithm that numpy comes with.
        # cam = resize_1d(cam, (input_tensor.shape[2]))

        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)#归一化处理
        # print(heatmap.shape)
        return heatmap


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=1)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=1)
        return weights


def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

if __name__ == '__main__':
    # from pytorch_grad_cam.utils.image import preprocess_image
    model = OnedcnnNet()
    model_interface = ModelInterface.load_from_checkpoint(r'{}/2fold_logs/onedcnn_net/version_1/checkpoints/best-val_acc=0.980.ckpt'.format(os.getcwd()))
    model = model_interface.network_module
    target_layer = model.conv['layer2'][0]
    net = GradCAM(model, target_layer)

    input_tensor = {}
    data_dir = r'{}/data/gearset'.format(os.getcwd()).replace("\\", "/")
    filenames = os.listdir(data_dir)
    file_name = [name for name in filenames if name.find('.csv') != -1]  # 文件名
    file_name.sort()
    for name in file_name:
        input_tensor[name] = []
        file_path = os.path.join(data_dir, name)
        file = pd.read_csv(file_path, skiprows=20, header=None, sep='\t')
        for i in range(8):
            input_tensor[name].append(file[i].ravel()[:4096])
        input_tensor[name] = np.array(input_tensor[name])
        input_tensor[name] = torch.from_numpy(input_tensor[name])

    input = input_tensor['Health_20_0.csv']

    input = input.reshape(1, input.shape[0], input.shape[1])
    print(input.shape)
    output = net(input)
    input_tensor = input.numpy().squeeze()

    # plt.plot(range(4096), output, color='r', label="CAM")  # s-:方形
    # # plt.plot(range(4096), input_tensor[0], color='g', label="input")  # o-:圆形
    # plt.xlabel("step")  # 横坐标名字
    # plt.ylabel("y")  # 纵坐标名字
    # plt.legend(loc="best")  # 图例
    # plt.show()


    root_path = r'{}/results/CAM'.format(os.getcwd()).replace("\\", "/")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    mat_path = r'{}/cam.mat'.format(root_path).replace("\\", "/")
    scio.savemat(mat_path, mdict={'cam': output, 'data': input_tensor[0]})

    # data = scio.loadmat(mat_path)
    #
    # a = data['data']
    # plt.imshow(k)
    # plt.show()
