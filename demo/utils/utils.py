from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt


from dataset_preprocess import preprocess


class MyDataset(Dataset):
    def __init__(self, data_path, model=0):
        self.dataset = []
        train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess(d_path=data_path,
                                                                        length=864,
                                                                        number=1000,
                                                                        normal=False,
                                                                        rate=[0.5, 0.25, 0.25],
                                                                        enhancement=False,
                                                                        enhancement_step=28)
        self.model = model
        self.train_data = np.concatenate((train_X, train_Y), axis=1)
        self.valid_data = np.concatenate((valid_X, valid_Y), axis=1)
        self.test_data = np.concatenate((test_X, test_Y), axis=1)
        self.dataset.append(self.train_data)
        self.dataset.append(self.valid_data)
        self.dataset.append(self.test_data)
        self.dividing = len(train_X[0])

    def __getitem__(self, index):
        # print(lenself.dataset[0][self.model][index][:self.dividing])
        x, label = self.dataset[self.model][index][:self.dividing], self.dataset[self.model][index][self.dividing:]
        return x, label

    def __len__(self):
        return len(self.dataset[self.model])


def validate(net, data_loader, set_name, classes_name):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'tesst
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        images, labels = data
        images = Variable(images)
        labels = Variable(labels)

        outputs = net(images)
        outputs.detach_()

        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i].numpy()
            pre_i = predicted[i].numpy()
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
                                                                conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


def show_confMat(confusion_mat, classes, set_name, out_dir):

    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    plt.close()


def normalize_invert(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


if __name__ == "__main__":
    path = r'../data/Gearbox'
    train_data = MyDataset(data_path=path, model=0)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.shape, labels.shape)