import os

import numpy as np
import json
import matplotlib.pyplot as plt
import csv

import pandas as pd
from scipy.fftpack import fft, ifft

# 读入数据
datafile = r'{}/data/gearset/Health_20_0.csv'.format(os.getcwd()).replace("\\", "/")
# 从文件数据，sep参数是数据分割字符串
# dtype=np.int32,
data = []
file = pd.read_csv(datafile, skiprows=5000, header=None, sep='\t')
for i in range(8):
    data.append(file[i].ravel()[:40960])
data = np.array(data)

# 读取波形需要的相关参数
ch_jump = 0  # 跳过前面多少个通道
ch_num = 4  # 总共绘制几个通道的波形
left_cursor = 1000  # 波形绘制左游标
right_cusor = 2000  # 波形绘制右游标

sample_dot = 10000  # 采样频率

dot_num = right_cusor - left_cursor

ch = np.empty((ch_num, dot_num))
ch_fft = np.empty((ch_num, dot_num))

x = range(dot_num)

for i in range(ch_num):
    ch[i] = data[ch_jump + i, left_cursor:right_cusor]
    f = plt.subplot(ch_num, 1, i + 1)
    case = i % 4
    if case == 0:
        plt.plot(x, ch[i], color='y')
    elif case == 1:
        plt.plot(x, ch[i], color='g')
    elif case == 2:
        plt.plot(x, ch[i], color='r')
    else:
        plt.plot(x, ch[i], color='k')

plt.show()

for i in range(ch_num):
    ch_fft[i] = fft(ch[i])
    fft_x = np.arange(dot_num)  # 频率个数
    half_fft_x = fft_x[range(int(dot_num / 2))]  # 取一半区间

    half_fft_x = (sample_dot // dot_num) * half_fft_x

    ch_hz_abs = np.abs(ch_fft[i])  # 取复数的绝对值，即复数的模(双边频谱)
    cha_hz_angle = np.angle(ch_fft[i])  # 取复数的角度
    normalization_y = ch_hz_abs / dot_num  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(dot_num / 2))]  # 由于对称性，只取一半区间（单边频谱）

    normalization_half_y[1:] = 2 * normalization_half_y[1:]
    plot_sub = 50
    f = plt.subplot(ch_num, 1, i + 1)
    case = i % 4
    if case == 0:
        plt.plot(half_fft_x[:plot_sub], normalization_half_y[:plot_sub], color='y')
    elif case == 1:
        plt.plot(half_fft_x[:plot_sub], normalization_half_y[:plot_sub], color='g')
    elif case == 2:
        plt.plot(half_fft_x[:plot_sub], normalization_half_y[:plot_sub], color='r')
    else:
        plt.plot(half_fft_x[:plot_sub], normalization_half_y[:plot_sub], color='k')

plt.show()
