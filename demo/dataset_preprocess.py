from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import matplotlib.pyplot as plt

def preprocess(d_path, length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enhancement=True, enhancement_step=28):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enhancement: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enhancement_step: 增强数据集采样顺延间隔
    :return: train_X, train_Y, valid_X, valid_Y, test_X, test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enhancement=True,
                                                                    enhancement_step=28)
    ```
    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def csv_to_dict(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = pd.read_csv('./{}'.format(file_path))
            files[i] = file['a1'].ravel()
        return files


    def data_slice(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单条数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if enhancement:
                enhancement_time = length // enhancement_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enhancement_time):
                        samp_step += 1
                        random_start += enhancement_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            Y += [label] * len(x)
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(train_Y, test_Y):
        train_Y = np.array(train_Y).reshape([-1, 1])
        test_Y = np.array(test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(train_Y)
        train_Y = Encoder.transform(train_Y).toarray()
        test_Y = Encoder.transform(test_Y).toarray()
        train_Y = np.asarray(train_Y, dtype=np.int32)
        test_Y = np.asarray(test_Y, dtype=np.int32)
        return train_Y, test_Y

    # 用训练集标准差标准化训练集以及测试集
    def standard_scalar(train_X, test_X):
        scalar = preprocessing.StandardScaler().fit(train_X)
        train_X = scalar.transform(train_X)
        test_X = scalar.transform(test_X)
        return train_X, test_X

    def valid_test_slice(test_X, test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in sss.split(test_X, test_Y):
            X_valid, X_test = test_X[train_index], test_X[test_index]
            Y_valid, Y_test = test_Y[train_index], test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = csv_to_dict(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = data_slice(data)
    # 为训练集制作标签，返回X，Y
    train_X, train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    test_X, test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    train_Y, test_Y = one_hot(train_Y, test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        train_X, test_X = standard_scalar(train_X, test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        train_X = np.asarray(train_X)
        test_X = np.asarray(test_X)
    # 将测试集切分为验证集合和测试集.
    valid_X, valid_Y, test_X, test_Y = valid_test_slice(test_X, test_Y)
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数

if __name__ == "__main__":
    path = r'data/Gearbox'
    # train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess(d_path=path,
    #                                                             length=864,
    #                                                             number=1000,
    #                                                             normal=False,
    #                                                             rate=[0.5, 0.25, 0.25],
    #                                                             enhancement=False,
    #                                                             enhancement_step=28)
    # print(train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape, test_X.shape, test_Y.shape)

    filenames = os.listdir(path)
    files = {}
    for i in filenames:
        # 文件路径
        file_path = os.path.join(path, i)
        file = pd.read_csv('./{}'.format(file_path))
        files[i] = file['a1'].ravel()

    start, end = 0, min(len(files['h30hz0.csv']), len(files['b30hz0.csv']))
    start, end = 0, 500
    x = np.arange(start, end)
    window_size = 1
    average1 = moving_average(interval=files['h30hz0.csv'], window_size=window_size)
    average2 = moving_average(interval=files['h30hz90.csv'], window_size=window_size)
    plt.plot(x, average1[start:end], color='red', label='Healthy-load0')
    plt.plot(x, average2[start:end], color='blue', label='Healthy-load90')

    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    plt.xlabel('$\t{Step}$', font1)
    plt.ylabel('$\t{Vibration}$', font1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig('./load_compare.png', bbox_inches='tight')
    plt.show()

    # Acc = np.load("Acc.npy")
    # plt.plot(range(len(Acc)), Acc, color='red')
    # plt.rcParams['figure.figsize'] = (6.0, 4.0)
    # plt.rcParams['savefig.dpi'] = 200  # 图片像素
    # plt.rcParams['figure.dpi'] = 200  # 分辨率
    # plt.xlabel('$\t{Epoch}$', font1)
    # plt.ylabel('$\t{Acc}$', font1)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend()
    # plt.savefig('./Acc.png', bbox_inches='tight')
    # plt.show()


