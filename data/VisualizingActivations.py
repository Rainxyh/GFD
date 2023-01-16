# 对单个图像可视化
import imageio
import pandas as pd
import pywt
import scipy
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
import os

from data.signal2img import CWT_time_frequency_diagram

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from data.wavelet_transform import multi_sensor_signal2cube, signal2matrix
from model import ModelInterface
from model.mutilconvgru_net import MutilconvgruNet
from sklearn.preprocessing import MinMaxScaler


def time_freq_img(data, name, time=1, sampling_rate=128):
    t = np.arange(0, time, 1.0 / sampling_rate)
    wavename = 'morl'  # "cmorB-C" where B is the bandwidth and C is the center frequency.
    totalscal = 16  # scale
    fc = pywt.central_frequency(wavename)  # central frequency
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(1, totalscal + 1)

    [cwtmatr_l, frequencies_l] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)  # continuous wavelet transform

    plt.figure(figsize=(8, 4))
    plt.contourf(t, frequencies_l, abs(cwtmatr_l), cmap='jet', levels=np.linspace(0, 1, 20), extend='both')
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.title(name)
    plt.colorbar()
    plt.show()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
GPU = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
sensor_number = 1

# 1.加载模型
# model = resnet50(pretrained=True)
model_interface = ModelInterface.load_from_checkpoint(r'{}/2fold_logs/mutilconvgru_net/pow6_nlay4_ksz[(9, 9), (7, 7), (5, 5), (3, 3)]_slen4096_ep100/version_0/checkpoints/best-val_acc=0.960.ckpt'.format(os.getcwd())).to(device)
model = model_interface.network_module

# 2.选择目标层
# target_layer = [model.layer4]


min_max_scaler = MinMaxScaler(feature_range=(0, 255))  # 根据需要设置最大最小值，这里设置最大值为1.最小值为0
# 3. 构建输入图像的Tensor形式
input_tensor = {}
data_dir = r'{}/data/gearset'.format(os.getcwd()).replace("\\", "/")
filenames = os.listdir(data_dir)
file_name = [name for name in filenames if name.find('.csv') != -1]  # 文件名
file_name.sort()
for sensor in range(8):
    for name in file_name:
        input_tensor[name] = []
        file_path = os.path.join(data_dir, name)
        file = pd.read_csv(file_path, skiprows=5000, header=None, sep='\t')
        for i in range(sensor_number):
            input_tensor[name].append(file[sensor].ravel()[:4096])
        input_tensor[name] = np.array(input_tensor[name])

        # img_a = signal2matrix(input_tensor[name][0], n=6)
        # img_b = signal2matrix(input_tensor[name][1], n=6)
        # img_c = signal2matrix(input_tensor[name][2], n=6)
        # rgb_img = np.stack((img_a, img_b, img_c), axis=-1)
        # normal_rgb = min_max_scaler.fit_transform(rgb_img.reshape(-1, 1))
        # rgb_img = normal_rgb.reshape(rgb_img.shape)  # 标准化，注意这里的values是array
        # image_path = '{}/time_frequency/example/{}.png'.format(os.getcwd(), name[:-4])
        # cv2.imwrite(image_path, rgb_img)

    # for name in file_name:
    #     for i in range(sensor_number):
    #         time_freq_img(input_tensor[name][i][:128], name+str(i))
    #     break

    #     image_path = '{}/tf_img/{}.png'.format(os.getcwd(), name[:-4])
    #     rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]   # 1是读取rgb
    #     rgb_img = np.float32(rgb_img) / 255


    # preprocess_image作用：归一化图像，并转成tensor
    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
    #                                              std=[0.229, 0.224, 0.225])   # torch.Size([1, 3, 224, 224])
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    for i in range(4):
        target_layer = [model.cell_list[i].conv_candidate]
        # Construct the CAM object once, and then re-use it on many images:
        # 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
        cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

        # If target_category is None, the highest scoring category
        # will be used for every image in the batch.
        # target_category can also be an integer, or a list of different integers
        # for every image in the batch.
        # 5.选定目标类别，如果不设置，则默认为分数最高的那一类
        target_category = None  # 281

        for name in file_name:
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            # 6. 计算cam

            input = multi_sensor_signal2cube(input_tensor[name].reshape(1, sensor_number, 4096), 6)
            input = np.transpose(input, (0, 2, 1, 3, 4))
            input = torch.Tensor(input)
            grayscale_cam = cam(input_tensor=input, targets=target_category)  # [batch, 224,224]

            # In this example grayscale_cam has only one image in the batch:
            # 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
            grayscale_cam = grayscale_cam[0]
            path = '{}/time_frequency/result{}/layer{}'.format(os.getcwd(),sensor, i)
            if not os.path.exists(path):
                os.makedirs(path)
            visualization = show_cam_on_image(np.zeros_like(grayscale_cam)[:, :, None], grayscale_cam)  # (224, 224, 3)
            vis_path = '{}/cam_{}.png'.format(path, name[:-4])
            cv2.imwrite(vis_path, visualization)

            if i == 3:
                vis_path = '{}/tf_{}.png'.format(path, name[:-4])
                specgram_path = '{}/specgram_{}.png'.format(path, name[:-4])
                heatmap_path = '{}/shmap_{}.png'.format(path, name[:-4])
                CWT_time_frequency_diagram(input_tensor[name], vis_path, specgram_path, heatmap_path)
