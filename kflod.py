import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import seaborn
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from model import ModelInterface
from data import DatasetInterface
from argparse import ArgumentParser

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


def plot_classification_report(trues, preds, names, valid_labels,
                               title='Classification Report Plot',
                               out_path=None, save=False):
    report = classification_report(trues, preds, labels=valid_labels, target_names=names, output_dict=True)

    f, ax = plt.subplots(figsize=(14, 10))
    ax = seaborn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    ax.set_title(title)
    plt.show()
    if save and out_path is not None:
        plt.savefig(out_path)


def plot_confusion_matrix(trues, preds, names,
                          title="Confusion Matrix Visualization",
                          out_path=None, save=False):
    cm = confusion_matrix(trues, preds)
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    plt.figure(figsize=(10, 7))
    ax = seaborn.heatmap(df_cm, annot=True)
    ax.set_title(title)
    plt.show()
    if save and out_path is not None:
        plt.savefig(out_path)  # 'confusion matrix of sr.png'


def vis_metrix(trues, preds, names,
               valid_labels,
               explore_type='f1-score',
               title="Bar Plot", out_path=None, save=False):
    report = classification_report(trues, preds, labels=valid_labels, target_names=names, output_dict=True)
    f1_list = [report[name][explore_type] for name in target_names]
    f, ax = plt.subplots(figsize=(10, 7))
    ax = seaborn.barplot(x=target_names, y=f1_list, ax=ax)
    title = f"{explore_type} {title}"
    ax.set_title(title)
    plt.show()
    if save and out_path is not None:
        plt.savefig(out_path)


def vis_melt_metrix(result_holder, names,
               valid_labels,
               explore_type='f1-score',
               title=None):
    report_drone = classification_report(result_holder['trues'],
                                    result_holder['drone'],
                                    labels=valid_labels,
                                    target_names=names,
                                    output_dict=True)
    report_sr = classification_report(result_holder['trues'],
                                    result_holder['sr'],
                                    labels=valid_labels,
                                    target_names=names,
                                    output_dict=True)

    mcs_drone = [report_drone[name][explore_type] for name in target_names]
    mcs_sr = [report_sr[name][explore_type] for name in target_names]

    metrics_holder = {'class': target_names, 'Drone': mcs_drone, 'SR HSI': mcs_sr}

    df = pd.DataFrame(metrics_holder)
    df = pd.melt(df, id_vars="class", var_name="source", value_name=explore_type)
    fig, ax = plt.subplots(figsize=(10, 7))
    seaborn.barplot(x='class', y=explore_type, hue='source', data=df, ax=ax)
    if title is None:
        title = f"{explore_type} Bar Plot"
    else:
        title = f"{explore_type} {title}"
    ax.set_title(title)
    plt.show()


def test_model(loader, model):
    labels_list = []
    results_list = []
    correct_num = 0
    total = 0
    for data in loader:
        input, labels = data
        input = input.to(device)
        output = model(input).to(device)
        label_digit = labels.argmax(axis=1)
        predict_digit = output.argmax(axis=1).cpu()
        correct_num += sum(label_digit==predict_digit).item()
        total += len(labels)
        labels_list.append(label_digit.cpu().numpy())
        results_list.append(predict_digit.cpu().numpy())
    print(f"Acc: {correct_num / total}")
    return labels_list, results_list


def kfold_test(kfold, data_module, para_names, model_name):  # 读取data、model接口生成相应的对象 使用test_model_fold获得结果并记录进列表
    # log_root = f'{os.getcwd()}/{kfold}fold_log/{model_name}'
    log_root = f'{os.getcwd()}/tensorboard_logs/{model_name}'
    # model_paths = [f'{log_root}/{para_names}/version_{i}/checkpoints/best*'.replace("\\", "/") for i in range(kfold)]
    model_paths = [glob.glob(os.path.join(f'{log_root}/{para_names}/version_{i}/checkpoints/best*.ckpt'))[0] for i in range(kfold)]

    labels_list_total = []
    results_list_total = []
    for fold_idx in range(kfold):
        data_module.setup()
        loader = data_module.val_dataloader()
        print("running: version_", fold_idx)
        model = ModelInterface.load_from_checkpoint(model_paths[fold_idx]).to(device)  # 加载出第kfold的模型
        model = model.eval()
        labels_list, results_list = test_model(loader, model)
        labels_list_total.extend(labels_list)
        results_list_total.extend(results_list)
    return labels_list_total, results_list_total


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = ArgumentParser()
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--sample_length', default=8192, type=int)
parser.add_argument('--class_num', default=10, type=int)

# Basic Training Control
parser.add_argument('--power', default=5, type=int)

# Dataset
dataset_name_list = ['gearbox', 'gearbox_tfdata', 'gearbox_imgdata']
dataset_name = dataset_name_list[1]
parser.add_argument('--dataset', default=dataset_name, type=str)
parser.add_argument('--data_dir', default=r'{}/data/gearbox'.format(os.getcwd()).replace("\\", "/"), type=str)
parser.add_argument('--ref_dir', default=r'{}/data/ref'.format(os.getcwd()).replace("\\", "/"), type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--train_size', default=0.8, type=float)
parser.add_argument('--num_workers', default=16 if torch.cuda.is_available() else 0, type=int)
parser.add_argument('--standard_scalar', default=True, type=bool)
parser.add_argument('--train_mode', default=True, type=bool)
parser.add_argument('--sample_number', default=50, type=int)
parser.add_argument('--sensor_number', default=4, type=int)
parser.add_argument('--GAFpath', default=r'{}/data/{}/GAF'.format(os.getcwd(), dataset_name).replace("\\", "/"), type=str)
parser.add_argument('--transforms', default=data_transforms, type=dict)

if __name__ == '__main__':
    model = ModelInterface.load_from_checkpoint(
        '/home/rain/PythonProject/GFD/tensorboard_logs/mutilconvgru_net/pow5_hdim16_nlay4_ksz9,7,5,3_ncla10_slen8192/version_0/checkpoints/best-val_acc=0.980.ckpt').to(
        device)  # 加载出第kfold的模型


    args = parser.parse_args()
    pl.seed_everything(args.seed)  # 随机种子参数
    dataset_module = DatasetInterface(**vars(args))  # 数据集模型
    inputs, labels = dataset_module.train_set.__getitem__(0)
    print(inputs.shape, inputs.dtype)
    model(inputs)
    exit()

    result_holder = {}
    para_names = 'pow5_hdim16_nlay4_ksz9,7,5,3_ncla10_slen8192'
    model_name = 'mutilconvgru_net'


    labels_list, results_list = kfold_test(5, dataset_module, para_names, model_name)
    result_holder['trues'] = labels_list
    result_holder['Drone'] = results_list
    print(labels_list.shape)

    # label_dict = mio.load('data/ref/9cls_crop_dict.pkl')
    # rev_dict = {v : k for k, v in label_dict.items()}
    # valid_labels = list(set(labels_list) | set(results_list))
    # target_names = [rev_dict[i] for i in valid_labels]
    # # 基本评估
    # plot_classification_report(labels_list, results_list, target_names, valid_labels, # label_dict,
    #                            title=f"results/{type_str} Crop Classification Results",
    #                            out_path=f'{test_type}_result_report.png')
    # # 混淆矩阵
    # plot_confusion_matrix(labels_list, results_list, target_names,
    #                       title=f"{type_str} Crop Classification Results",
    #                       out_path=f'results/{test_type}_confusion_matrix.png')
    # # 条形图
    # vis_melt_metrix(result_holder, target_names,
    #             valid_labels,
    #             explore_type='f1-score',
    #             title=None,
    #             figsize=(10, 7))
    # # 文本报告
    # txt_report = classification_report(labels_list, drone_results_list)
    # print(txt_report)
