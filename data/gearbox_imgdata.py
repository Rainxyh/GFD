import os
import sys
import pandas as pd
import torch
import numpy as np
import pickle
import torch.utils.data as data
from torch.utils import data
from torchvision import datasets, transforms
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
from matplotlib import image
from pyts.image import GramianAngularField


class GearboxImgdata(data.Dataset):
    def __init__(self,
                 data_dir,
                 GAFpath,
                 transforms=None,
                 train_mode=True,
                 img_sz=224,
                 window_sz=256,
                 imgs_pre_file=10,
                 seed=1234,
                 train_size=0.8,
                 class_num=10):
        # Set all input args as attributes
        self.__dict__.update(locals())  # locals()以字典类型返回当前位置的全部局部变量 self.__dict__.update()加载字典
        self.train_set_len = 0
        self.test_set_len = 0
        self.X_train_set = []
        self.y_train_set = []
        self.X_test_set = []
        self.y_test_set = []
        self.transforms = transforms
        self.preprocess(datapath=data_dir, GAFpath=GAFpath)

    def preprocess(self, datapath, GAFpath):
        self.rowsignal2img(datapath, GAFpath)
        self.make_path_label_file(datapath, GAFpath)
        img_list = []
        labels = []
        txt_file = os.path.join(GAFpath, "img_path_label_list.txt").replace("\\", "/")
        with open(txt_file, "r") as fr:
            for line in fr:
                img_path, cls_name = line.strip().split("\t")
                img_list.append(img_path)
                one_hot_label = self.to_one_hot(int(cls_name))
                labels.append(one_hot_label)
        labels = np.array(labels)
        self.data_slice(img_list, labels)

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out

    def rowsignal2img(self, datapath, GAFpath):
        if not os.path.exists(GAFpath):
            filename_list = [name for name in os.listdir(datapath) if name.find('.csv') != -1]  # 文件名
            for i, name in enumerate(filename_list):
                filepath = "{}/{}".format(datapath, name)  # 要处理的文件路径
                img_sz = self.img_sz  # 生成的 GAF 图片的大小 (the size of each GAF image)
                # 如果 滑动窗口的大小 等于 滑动步长 则滑动窗口之间没有重叠
                window_sz = self.window_sz  # 滑动窗口的大小，需要满足 window_sz > img_sz
                step = window_sz  # 滑动窗口的步长 (step of slide window)
                assert window_sz >= img_sz, "window_sz < img_sz（滑动窗口大小 小于 GAF 图片尺寸）。"
                method = 'summation'  # GAF 图片的类型，可选 'summation'（默认）和 'difference'

                # 以下是 GAF 生成的代码
                print("GAF 生成方法：%s，图片大小：%d * %d" % (method, img_sz, img_sz))
                img_path = "{}/images/{}".format(GAFpath, name)  # 可视化图片保存的文件夹
                data_path = "{}/textdata/{}".format(GAFpath, name)  # 数据文件保存的文件夹
                if not os.path.exists(img_path):
                    os.makedirs(img_path)  # 如果文件夹不存在就创建一个
                if not os.path.exists(data_path):
                    os.makedirs(data_path)  # 如果文件夹不存在就创建一个

                print("开始生成...")
                print("可视化图片保存在文件夹 %s 中，数据文件保存在文件夹 %s 中。" % (img_path, data_path))
                gaf = GramianAngularField(image_size=img_sz, method=method)
                data = np.loadtxt("{}/{}".format(datapath, name), delimiter=",", skiprows=1)
                n_sample, n_channels = data.shape

                img_num = 0  # 生成图片的总数 (the total numbers of GAF images)
                start_index, end_index = 0, window_sz  # 序列开头以及末尾索引
                while end_index <= n_sample:
                    img_num += 1
                    if img_num % self.imgs_pre_file == 0:
                        break
                    sub_series = data[start_index:end_index, :]  # 获得当前滑动窗口中的数据
                    gaf_images = gaf.fit_transform(sub_series.T)  # 转化为 GAF 图片
                    for c in range(n_channels):  # 保存每个 channel 的图片
                        gaf_img = gaf_images[c, :, :]  # 得到第 c 个 channel 的数据

                        img_save_path = "{}/{}_{}.png".format(img_path, img_num, c)
                        data_save_path = "{}/{}_{}.csv".format(data_path, img_num, c)
                        image.imsave(img_save_path, gaf_img)  # 保存图片 (save image)
                        np.savetxt(data_save_path, gaf_img, delimiter=',')  # 保存数据为 csv 文件

                    # 滑动窗口向后移动
                    start_index += step
                    end_index += step
                print("No.{}: {} 文件处理完成,生成 {} 张图片".format(i, name, img_num * n_channels))
        else:
            print("已存在GAF转换数据，未进行操作")

    def make_path_label_file(self, datapath, GAFpath):
        txt_path = os.path.join(GAFpath, "img_path_label_list.txt").replace("\\", "/")
        if not os.path.exists(txt_path):
            file_label_dict_path = os.path.join(self.ref_dir, "file_label_dict.pkl")
            with open(file_label_dict_path, 'rb') as fo:
                file_label_dict = pickle.load(fo)
            filename_list = [name for name in os.listdir(datapath) if name.find('.csv') != -1]  # 文件名
            all_file = []
            for i, cls_name in enumerate(filename_list):
                file_dir = os.path.join(GAFpath, "images", cls_name).replace("\\", "/")
                for img_name in os.listdir(file_dir):
                    if img_name.endswith(".png"):
                        img_path = os.path.join(file_dir, img_name)
                        label = file_label_dict[cls_name]
                        all_file.append([img_path, label])  # 图片路径和标签

            file_str = ""
            for img_path, cls_name in all_file:
                file_str += img_path + "\t" + cls_name + "\n"
            with open(txt_path, "w") as fw:
                fw.write(file_str)
        else:
            print("已存在图片路径标签列表，未进行操作")

    def data_slice(self, img_list, labels):
        self.X_train_set, self.X_test_set, self.y_train_set, self.y_test_set = train_test_split(img_list, labels, train_size=self.train_size, random_state=self.seed)
        self.train_set_len, self.test_set_len = len(self.y_train_set), len(self.y_test_set)

    def standard_scalar(self):
        self.X_train_set = preprocessing.StandardScaler().fit_transform(self.X_train_set)
        self.X_test_set = preprocessing.StandardScaler().transform(self.X_test_set)

    def __len__(self):
        return self.train_set_len if self.train_mode else self.test_set_len

    def __getitem__(self, index):
        if self.train_mode:
            img_path = self.X_train_set[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transforms['train'](img)
            label = self.y_train_set[index]
        else:
            img_path = self.X_test_set[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transforms['val'](img)
            label = self.y_test_set[index]
        return img, label


if __name__ == "__main__":
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    datapath = "{}/gearbox".format(os.getcwd()).replace("\\", "/")
    GAFpath = "{}/GAF".format(datapath).replace("\\", "/")  # GAF 图片保存的路径
    data = GearboxImgdata(datapath, GAFpath, transforms=data_transforms)
    img, label = data.__getitem__(0)
    print(img)


