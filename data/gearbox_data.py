import os
import sys
import pandas as pd
import torch
import numpy as np
import pickle
import torch.utils.data as data

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

from data.wavelet_transform import multi_sensor_signal2cube

class GearboxData(data.Dataset):
    def __init__(self,
                 data_dir=r'{}/gearbox'.format(os.getcwd()),
                 ref_dir=r'{}/ref'.format(os.getcwd()),
                 class_num=10,
                 train_mode=True,
                 sample_number=1000,
                 sample_length=4096,
                 seed=1234,
                 train_size=0.8,
                 sensor_number=8,
                 wavelet_packet=False,
                 power=5):
        # Set all input args as attributes
        self.__dict__.update(locals())  # locals()以字典类型返回当前位置的全部局部变量 self.__dict__.update()加载字典
        self.train_set_len = 0
        self.test_set_len = 0
        self.X_train_set = []
        self.y_train_set = []
        self.X_test_set = []
        self.y_test_set = []
        self.preprocess()

    """
    每个文件一种标签 1.将多个文件读取出来 2.以文件为单位添加标签 3.将带有标签的数据合并起来 4.数据划分
    """
    def label_judge(self):
        """以单个文件为单位生成字典并写入文件
        :param
        :return: dict{file_name: label}
        """
        path = os.path.join(self.ref_dir, "file_label_dict.pkl")
        if not os.path.exists(path):
            filenames = os.listdir(self.data_dir)
            file_name = [name for name in filenames if name.find('.csv') != -1]  # 文件名
            file_name.sort()
            # file_label = [0 if name[0] == 'h' else 1 for name in filenames]   # 二分类
            file_label = list(range(len(filenames)))  # 多分类
            file_label_dict = dict(zip(file_name, file_label))

            with open(path, 'wb') as fo:  # 将标签数据写入pickle文件 一个字典 其中一个键值对 对应一类标签相同的样本
                pickle.dump(file_label_dict, fo)

    def csv_to_dict(self):
        """读取csv文件，打上标签，并返回字典
        :param
        :return: 数据字典 {'feature':key-文件名 value-特征矩阵, 'label':key-文件名 value-标签}
        """
        files = {}
        path = os.path.join(self.ref_dir, "dict_files.pkl")
        if not os.path.exists(path):
            self.label_judge()
            label_path = os.path.join(self.ref_dir, "file_label_dict.pkl")

            with open(label_path, 'rb') as fo:
                file_label_dict = pickle.load(fo)

            filenames = os.listdir(self.data_dir)
            for name in filenames:
                if name[-4:] != '.csv':
                    continue
                files[name] = {}
                files[name]['feature'] = []
                # 存放数据
                file_path = os.path.join(self.data_dir, name)
                file = pd.read_csv(file_path, skiprows=20, header=None, sep='\t')
                for i in range(self.sensor_number):
                    files[name]['feature'].append(file[i].ravel())
                # 存放标签
                files[name]['label'] = file_label_dict[name]

                file_save = open(path, 'wb')
                pickle.dump(files, file_save)
                file_save.close()
        else:
            file_read = open(path, 'rb')
            files = pickle.load(file_read)
            file_read.close()
        return files

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out

    def time_series_slice_and_append(self):
        """将一整条时间序列切割成长度为length的一段一段数据
        :param
        :return:
        """
        sliced_data, sliced_target = [], []
        sliced_data_path = os.path.join(self.ref_dir, "sliced_data.pkl")
        sliced_target_path = os.path.join(self.ref_dir, "sliced_target.pkl")
        if not os.path.exists(sliced_data_path) or not os.path.exists(sliced_target_path):
            data_files = self.csv_to_dict()
            keys = data_files.keys()
            for name in keys:
                time_series_end = len(data_files[name]['feature'][0])
                for i in range(self.sample_number):  # 采样次数
                    random_start = np.random.randint(low=0, high=(time_series_end-self.sample_length))
                    sample = []
                    for j in range(self.sensor_number):  # 传感器数量
                        sample.append(data_files[name]['feature'][j][random_start:random_start+self.sample_length])
                    sliced_data.append(sample)

                    one_hot_label = self.to_one_hot(data_files[name]['label'])
                    sliced_target.append(one_hot_label)
            sliced_data, sliced_target = np.array(sliced_data), np.array(sliced_target)

            file_save = open(sliced_data_path, 'wb')
            pickle.dump(sliced_data, file_save)
            file_save = open(sliced_target_path, 'wb')
            pickle.dump(sliced_target, file_save)
            file_save.close()
        else:
            file_read = open(sliced_data_path, 'rb')
            sliced_data = pickle.load(file_read)
            file_read = open(sliced_target_path, 'rb')
            sliced_target = pickle.load(file_read)
            file_read.close()

        return sliced_data, sliced_target

    def data_slice(self):
        """对数据进行划分
        :param
        :return:
        """
        feature, target = self.time_series_slice_and_append()
        self.X_train_set, self.X_test_set, self.y_train_set, self.y_test_set = train_test_split(feature, target, train_size=self.train_size, random_state=self.seed)
        self.train_set_len, self.test_set_len = len(self.y_train_set), len(self.y_test_set)

    def preprocess(self):
        self.data_slice()
        if self.wavelet_packet:
            self.X_train_set = multi_sensor_signal2cube(self.X_train_set, self.power)
            self.X_test_set = multi_sensor_signal2cube(self.X_test_set, self.power)
            self.y_train_set = np.array(self.y_train_set)
            self.y_test_set = np.array(self.y_test_set)
            # print(self.X_train_set.shape, self.X_test_set.shape, self.y_train_set.shape, self.y_test_set.shape)
            self.X_train_set = np.transpose(self.X_train_set, (0, 2, 1, 3, 4))
            self.X_test_set = np.transpose(self.X_test_set, (0, 2, 1, 3, 4))

    def standard_scalar(self):
        self.X_train_set = preprocessing.StandardScaler().fit_transform(self.X_train_set)
        self.X_test_set = preprocessing.StandardScaler().transform(self.X_test_set)

    def __len__(self):
        return self.train_set_len if self.train_mode else self.test_set_len

    def __getitem__(self, idx):
        if self.train_mode:
            inputs = self.X_train_set[idx]
            labels = self.y_train_set[idx]
        else:
            inputs = self.X_test_set[idx]
            labels = self.y_test_set[idx]
        return inputs, labels

from model.mutilconvgru_net import *
if __name__ == "__main__":
    n = 5
    height = width = pow(2, n)
    channels = 4
    hidden_dim = 16
    kernel_size = (3, 3)  # kernel size for two stacked hidden layer
    num_layers = 3  # number of stacked hidden layer
    class_num = 20
    model = MutilconvgruNet(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=torch.FloatTensor,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False,
                    class_num=class_num,)

    data = GearboxData()
    data, label = data.__getitem__(0)
    print(data.shape)
    data_noise = data[np.newaxis, :]  # 添加batch维度
    input_tensor = data_noise
    print(input_tensor.shape)
    soft_output = model(input_tensor)
    print(soft_output)
    print(soft_output.shape)



