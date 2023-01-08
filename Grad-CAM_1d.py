import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio


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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
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


def resize_1d(array, shape):
    res = np.zeros(shape)
    if array.shape[0] >= shape:
        ratio = array.shape[0] / shape
        for i in range(array.shape[0]):
            res[int(i / ratio)] += array[i] * (1 - (i / ratio - int(i / ratio)))
            if int(i / ratio) != shape - 1:
                res[int(i / ratio) + 1] += array[i] * (i / ratio - int(i / ratio))
            else:
                res[int(i / ratio)] += array[i] * (i / ratio - int(i / ratio))
        res = res[::-1]
        array = array[::-1]
        for i in range(array.shape[0]):
            res[int(i / ratio)] += array[i] * (1 - (i / ratio - int(i / ratio)))
            if int(i / ratio) != shape - 1:
                res[int(i / ratio) + 1] += array[i] * (i / ratio - int(i / ratio))
            else:
                res[int(i / ratio)] += array[i] * (i / ratio - int(i / ratio))
        res = res[::-1] / (2 * ratio)
        array = array[::-1]
    else:
        ratio = shape / array.shape[0]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i / ratio):
                left += 1
                right += 1
            if right > array.shape[0] - 1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                          (i - left * ratio) / ratio + array[left] * (right * ratio - i) / ratio
        res = res[::-1]
        array = array[::-1]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i / ratio):
                left += 1
                right += 1
            if right > array.shape[0] - 1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                          (i - left * ratio) / ratio + array[left] * (right * ratio - i) / ratio
        res = res[::-1] / 2
        array = array[::-1]
    return res


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

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
        print(output.shape)
        print(target_category)
        return output[0][target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        # weights = np.mean(grads, axis=(0))
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :]
        # cam = activations.T.dot(weights)    #maybe better
        # print(input_tensor.shape[1])
        cam = resize_1d(cam, (input_tensor.shape[2]))
        cam = np.maximum(cam, 0)
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
        return heatmap


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations, grads):
        return np.mean(grads, axis=1)

if __name__ == '__main__':
    model = CnnNet()
    target_layer = model.conv1
    net = GradCAM(model, target_layer)

    input_tensor = torch.randn((5, 8, 64, 64))
    input_tensor = input_tensor[0:1, :]
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    # plt.figure(figsize=(5, 1))
    output = net(input_tensor)

    input_tensor = input_tensor.numpy().squeeze()
    dataNew = "G:\\datanew.mat"
    scio.savemat(dataNew, mdict={'cam': output, 'data': input_tensor})