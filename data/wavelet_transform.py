import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet

import sys
sys.path.append('~/Github/GFD/data')
from swpt import SWPT


def signal2matrix(signal, n=3):
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=n)
    re = []  # 第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    matrix = np.vstack(re)
    return matrix


def multi_sensor_signal2cube(multi_sensor_signal, n=3):
    # multi_sensor_signal = np.array(multi_sensor_signal.cpu())
    all_batch_cube = []
    for b in range(multi_sensor_signal.shape[0]):  # 遍历每个batch
        cube = []
        for i in range(multi_sensor_signal.shape[1]):  # 遍历传感器数量
            time_step_cube_list = []
            for t in range(multi_sensor_signal.shape[2]//pow(2, 2*n)):  # 遍历时间步
                # 选中第t个时间步中的信号段 并通过小波变换转换为二维时频
                matrix = signal2matrix(multi_sensor_signal[b][i][pow(2, 2*n)*t:pow(2, 2*n)*(t+1)], n)
                time_step_cube_list.append(matrix)
            np.concatenate(time_step_cube_list, axis=0)
            cube.append(time_step_cube_list)
        all_batch_cube.append(cube)
    all_batch_cube = np.array(all_batch_cube)
    return all_batch_cube


# 小波树图
def wavelet_tree_plt(signal, n=3):
    # wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=n)

    # 计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = dict()
    map[1] = signal
    for row in range(1, n + 1):
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data

    # 作图
    plt.figure(figsize=(15, 10))
    plt.subplot(n + 1, 1, 1)  # 绘制第一个图
    plt.title('Wavelet Tree Diagram')
    plt.plot(map[1])
    for i in range(1, n + 1):
        level_num = pow(2, i)  # 从第二行图开始，计算上一行图的2的幂次方
        # 获取每一层分解的node：比如第三层['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i, 'freq')]
        for j in range(1, level_num + 1):
            plt.subplot(n + 1, level_num, level_num * i + j)
            plt.plot(map[re[j - 1]])  # 列表从0开始
            plt.title(re[j - 1], y=-0.2)
    plt.show()


# 小波包能量特征提取 并绘制小波能量特征柱形图，注意这里的节点顺序不是自然分解的顺序，而是频率由低到高的顺序
def wavelet_packet_feature_extraction(signal, n=3, percentage=True):
    # wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=n)

    re = []  # 第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    # 第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i, ord=None), 2))

    if percentage:
        energy = 100.0*np.array(energy)/sum(energy)

    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(10, 7), dpi=80)
    plt.bar(x=np.arange(pow(2, n)), height=energy, width=0.45, label="num", color="#87CEFA", data=energy)
    plt.xlabel('frequency band clusters')
    plt.ylabel('energy percentage (%)')
    plt.title('Wavelet Packet Energy Spectrum Analysis')
    plt.xticks(np.arange(pow(2, n)), np.arange(pow(2, n)-1, pow(2, n+1)-1))
    plt.legend(loc="upper right")
    plt.show()

    return energy


def wavelet_packet_transform(x, level=3, mother_wavelet='dmey'):
    wp = pywt.WaveletPacket(data=x, wavelet=mother_wavelet, mode='symmetric', maxlevel=level)
    node_name_list = [node.path for node in wp.get_level(level, 'natural')]
    rec_results = []
    for i in node_name_list:
        new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet=mother_wavelet, mode='symmetric')
        new_wp[i] = wp[i].data
        x_i = new_wp.reconstruct(update=True)
        rec_results.append(x_i)
    output = np.array(rec_results)
    return output


def show_np_array(y):
    x = np.arange(y.shape[-1])

    n = 1
    if y.ndim == 2:
        n = y.shape[0]

    plt.figure(figsize=(30, 20))
    for i in range(1, n+1):
        plt.subplot(n, 1, i)
        plt.plot(x, y if n == 1 else y[i-1], color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Num. {}'.format(i))
    plt.show()


def show_origin_denoise_refactor(data_noise):
    # Wavelet Denoising 去噪
    data_denoise = denoise_wavelet(data_noise, method='VisuShrink', mode='soft', wavelet_levels=1, wavelet='sym8',
                                   rescale_sigma='True')

    coeffs = pywt.wavedec(data_denoise, wavelet='sym8', mode='sym', level=3)
    y = pywt.waverec(coeffs, wavelet='sym8', mode='sym')

    show_np_array(data_noise)
    show_np_array(data_denoise)
    show_np_array(y)


if __name__ == '__main__':
    file = pd.read_csv("./gearbox/b30hz0.csv", skiprows=20, header=None)
    data = []
    for i in range(4):
        data.append(file[i].ravel())
    data = np.array(data)
    
    signal2matrix(data[i])
    pass
