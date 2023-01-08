import os
import pywt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyts.image import RecurrencePlot
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from data.gearbox_data import GearboxData
from data.wavelet_transform import signal2matrix

'''
读取时间序列的数据
怎么读取需要你自己写
X为ndarray类型数据
'''

def seq2img(idx, X):
    if idx == 1:
        recurrence_plot(X)
    elif idx == 2:
        markov_transition_field(X)
    elif idx == 3:
        gramian_angular_field(X)
    elif idx == 0:
        recurrence_plot(X)
        markov_transition_field(X)
        gramian_angular_field(X)

    CWT_time_frequency_diagram(X)

# 1.Recurrence Plot （递归图）--------------------------------------------------------------------------------
def recurrence_plot(X):
    # Recurrence plot transformation
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(X)

    # Show the results for the first time series
    plt.figure(figsize=(5, 5))
    plt.imshow(X_rp[0], cmap='binary', origin='lower')
    plt.title('Recurrence Plot', fontsize=16)
    plt.tight_layout()
    plt.show()


# 2.Markov Transition Field （马尔可夫变迁场）--------------------------------------------------------------------------------
def markov_transition_field(X):
    # MTF transformation
    mtf = MarkovTransitionField(image_size=64)
    X_mtf = mtf.fit_transform(X)

    # Show the image for the first time series
    plt.figure(figsize=(5, 5))
    plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
    plt.title('Markov Transition Field', fontsize=18)
    plt.colorbar(fraction=0.0457, pad=0.04)
    plt.tight_layout()
    plt.show()


# 3. Gramian Angular Field （格拉米角场）--------------------------------------------------------------------------------
def gramian_angular_field(X):
    # Transform the time series into Gramian Angular Fields
    gasf = GramianAngularField(image_size=64, method='summation')
    X_gasf = gasf.fit_transform(X)
    gadf = GramianAngularField(image_size=64, method='difference')
    X_gadf = gadf.fit_transform(X)

    # Show the results for the first time series
    axs = plt.subplots()
    plt.subplot(211)
    plt.imshow(X_gasf[0], cmap='rainbow', origin='lower')
    plt.title("GASF", fontsize=16)
    plt.subplot(212)
    plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
    plt.title("GADF", fontsize=16)

    # plt.axes((left, bottom, width, height)
    cax = plt.axes([0.7, 0.1, 0.02, 0.8])
    plt.colorbar(cax=cax)
    plt.suptitle('Gramian Angular Fields', y=0.98, fontsize=16)
    plt.tight_layout()
    plt.show()

def CWT_time_frequency_diagram(data, save_path='./', specgram_path='./', heatmap_path='./'):
    data = data.reshape(data.shape[-1])
    sampling_rate = data.shape[-1]
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    # f1 = 100
    # f2 = 200
    # f3 = 300
    # data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
    #                     [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
    #                      lambda t: np.sin(2 * np.pi * f3 * t)])
    wavename = 'cgau8'
    totalscal = 256
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel("time(s)")
    plt.title("time-frequency diagram", fontsize=20)
    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel("frequency(Hz)")
    plt.xlabel("time(s)")
    plt.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig(save_path)

    plt.subplots()
    plt.specgram(data, NFFT=256, Fs=4096, noverlap=0)
    # plt.show()
    plt.savefig(specgram_path)

    matrix = signal2matrix(data, 6)
    fig, ax = plt.subplots(figsize=(9, 9))
    # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
    # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
    # sns.heatmap(
    #     pd.DataFrame(np.round(matrix, 2), columns=['img0', 'img1', 'img2', 'img3'], index=['img0', 'img1', 'img2', 'img3']),
    #     annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues")
    sns.heatmap(np.round(matrix, 2), annot=False, square=True, cmap="YlGnBu")
    # ax.set_title('二维数组热力图', fontsize = 18)
    ax.set_ylabel('freq', fontsize=18)
    ax.set_xlabel('time', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
    # plt.show()
    plt.savefig(heatmap_path)

if __name__ == '__main__':
    input_tensor = {}
    data_dir = r'{}/gearset'.format(os.getcwd()).replace("\\", "/")
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
        # input_tensor[name] = torch.from_numpy(input_tensor[name])

    X = input_tensor['Health_20_0.csv'][4].reshape(1, -1)
    CWT_time_frequency_diagram(X)
