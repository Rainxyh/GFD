# 利用 滑动窗口 将多列长的序列信号生成多个 GAF 图片
# Using a sliding window to generate GAF images from a long sequence of signals
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from pyts.image import GramianAngularField


def rowsignal2img(datapath, savepath):
    if not os.path.exists(savepath):
        filename_list = [name for name in os.listdir(datapath) if name.find('.csv') != -1]  # 文件名
        for i, name in enumerate(filename_list):
            filepath = "{}/{}".format(datapath, name)  # 要处理的文件路径
            img_sz = 224  # 生成的 GAF 图片的大小 (the size of each GAF image)
            # 如果 滑动窗口的大小 等于 滑动步长 则滑动窗口之间没有重叠
            window_sz = 256  # 滑动窗口的大小，需要满足 window_sz > img_sz
            step = 256  # 滑动窗口的步长 (step of slide window)
            assert window_sz >= img_sz, "window_sz < img_sz（滑动窗口大小 小于 GAF 图片尺寸）。"
            method = 'summation'  # GAF 图片的类型，可选 'summation'（默认）和 'difference'

            # 以下是 GAF 生成的代码
            print("GAF 生成方法：%s，图片大小：%d * %d" % (method, img_sz, img_sz))
            img_path = "{}/images/{}".format(savepath, name)  # 可视化图片保存的文件夹
            data_path = "{}/textdata/{}".format(savepath, name)  # 数据文件保存的文件夹
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
                if img_num % 5 == 0:
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

def make_path_label_file(datapath, savepath):
    txt_path = os.path.join(savepath, "img_path_label_list.txt").replace("\\", "/")
    if not os.path.exists(txt_path):
        all_file = []
        for i, cls_name in enumerate(filename_list):
            file_dir = os.path.join(savepath, "images", cls_name).replace("\\", "/")
            for img_name in os.listdir(file_dir):
                if img_name.endswith(".png"):
                    img_path = os.path.join(file_dir, img_name)
                    all_file.append([img_path, str(i)])  # 图片路径和标签

        file_str = ""
        for img_path, cls_name in all_file:
            file_str += img_path + "\t" + cls_name + "\n"
        with open(txt_path, "w") as fw:
            fw.write(file_str)
    else:
        print("已存在图片路径标签列表，未进行操作")

if __name__ == "__main__":
    datapath = "{}/gearbox0".format(os.getcwd()).replace("\\", "/")
    savepath = "{}/GAF".format(datapath).replace("\\", "/")  # GAF 图片保存的路径
    rowsignal2img(datapath=datapath, savepath=savepath)
    make_path_label_file(datapath=datapath, savepath=savepath)
