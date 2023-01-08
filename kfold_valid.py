import csv
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import seaborn
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from model import ModelInterface
from data import DatasetInterface
from argparse import ArgumentParser


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
GPU = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

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

model_name_list = ['mutilconvgru_net', 'cnn_net', 'gru_net', 'onedcnn_net', 'cnngru_net']

def plot_classification_report(result_holder, names, valid_labels,
                               title='Classification Report Plot',
                               out_path=None, save=False):
    report = classification_report(result_holder['label'], result_holder['result'], labels=valid_labels, target_names=names, output_dict=True)

    f, ax = plt.subplots(figsize=(14, 10))
    ax = seaborn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    ax.set_title(title)
    plt.show()
    if save and out_path is not None:
        plt.savefig(out_path)


def plot_confusion_matrix(result_holder, names,
                          title="Confusion Matrix Visualization",
                          out_path=None, save=False):
    cm = confusion_matrix(result_holder['label'], result_holder['result'])
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    plt.figure(figsize=(10, 7))
    ax = seaborn.heatmap(df_cm, annot=True)
    ax.set_title(title)
    plt.show()
    if save and out_path is not None:
        plt.savefig(out_path)  # 'confusion matrix of sr.png'


def vis_metrix(result_holder, names,
               valid_labels,
               explore_type='f1-score',
               title="Bar Plot", out_path=None, save=False):
    report = classification_report(result_holder['label'], result_holder['result'], labels=valid_labels, target_names=names, output_dict=True)
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
                    title=None,
                    out_path=None,
                    save=False):
    metrics_holder = {'class': target_names}

    report = {}
    mcs = {}
    for name in model_name_list:
        report[name] = classification_report(result_holder[name]['label'],
                                        result_holder[name]['result'],
                                        labels=valid_labels,
                                        target_names=names,
                                        output_dict=True)

        mcs[name] = [report[name][target][explore_type] for target in target_names]
        metrics_holder[name] = mcs[name]

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
    if save and out_path is not None:
        plt.savefig(out_path)


def test_model(loader, model, model_name):
    labels_list = []
    results_list = []
    correct_num = 0
    total = 0

    namelist = []
    filelist = os.listdir(r'{}/data/gearset'.format(os.getcwd()).replace("\\", "/"))
    for filename in filelist:
        if filename[-3:] == 'csv':
            namelist.append(filename)


    for data in loader:
        input, labels = data
        input = input.to(device)
        outputs = model(input).to(device)
        with open(f'{os.getcwd()}/results/{model_name}/outputs.tsv', 'a+', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerows(outputs.cpu().detach().numpy().tolist())

        label_digit = labels.argmax(axis=1)

        with open(f'{os.getcwd()}/results/{model_name}/labels.tsv', 'a+', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerows([namelist[label][:-4]] for label in label_digit.cpu().detach().numpy().tolist())

        predict_digit = outputs.argmax(axis=1).cpu()
        # print(label_digit)
        # print(predict_digit)
        correct_num += sum(label_digit==predict_digit).item()
        total += len(labels)
        labels_list.extend(label_digit.tolist())
        results_list.extend(predict_digit.tolist())
    print(f"Acc: {correct_num / total}")
    return labels_list, results_list


def kfold_test(kfold, data_module, para_names, model_name):  # 读取data、model接口生成相应的对象 使用test_model_fold获得结果并记录进列表
    log_root = f'{os.getcwd()}/{kfold}fold_logs/{model_name}'
    # log_root = f'{os.getcwd()}/tensorboard_logs/{model_name}'
    model_paths = [glob.glob(os.path.join(f'{log_root}/version_{i}/checkpoints/best*.ckpt'))[0] for i in range(kfold)]

    labels_list_total = []
    results_list_total = []
    print("running: ", model_name)
    for fold_idx in range(kfold):
        data_module.setup()
        loader = data_module.val_dataloader()
        print("running: ver_", fold_idx)
        model = ModelInterface.load_from_checkpoint(model_paths[fold_idx]).to(device)  # 加载出第kfold的模型
        model = model.eval()
        labels_list, results_list = test_model(loader, model, model_name)
        labels_list_total.extend(labels_list)
        results_list_total.extend(results_list)
    return labels_list_total, results_list_total


parser = ArgumentParser()
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--sample_length', default=4096, type=int)
parser.add_argument('--class_num', default=10, type=int)
parser.add_argument('--power', default=5, type=int)

# Dataset
dataset_name_list = ['gearbox_data', 'gearbox_imgdata']
dataset_name = dataset_name_list[0]
parser.add_argument('--dataset', default=dataset_name, type=str)
parser.add_argument('--data_dir', default=r'{}/data/gearset'.format(os.getcwd()).replace("\\", "/"), type=str)
parser.add_argument('--ref_dir', default=r'{}/data/ref'.format(os.getcwd()).replace("\\", "/"), type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--train_size', default=0.8, type=float)
parser.add_argument('--num_workers', default=16 if torch.cuda.is_available() else 0, type=int)
parser.add_argument('--standard_scalar', default=True, type=bool)
parser.add_argument('--train_mode', default=True, type=bool)
parser.add_argument('--sample_number', default=100, type=int)
parser.add_argument('--sensor_number', default=8, type=int)
parser.add_argument('--wavelet_packet', default=False, type=bool)
args = parser.parse_args()
pl.seed_everything(args.seed)  # 随机种子参数

if __name__ == '__main__':
    root_path = f'{os.getcwd()}/results'
    path = os.path.join(root_path, "result_holder.pkl")
    save_resutlt = True

    if not os.path.exists(path):
        kfold = 2

        result_holder = {}
        for model_name in model_name_list:
            result_holder[model_name] = {}

            if model_name=='mutilconvgru_net' or model_name=='cnn_net':
                args.wavelet_packet = True
            else :
                args.wavelet_packet = False
            dataset_module = DatasetInterface(**vars(args))  # 数据集模型

            para_names = ''
            labels_list, results_list = kfold_test(kfold, dataset_module, para_names, model_name)
            result_holder[model_name]['label'] = labels_list
            result_holder[model_name]['result'] = results_list

            file_save = open(path, 'wb')
            pickle.dump(result_holder, file_save)
            file_save.close()
    else:
        file_read = open(path, 'rb')
        result_holder = pickle.load(file_read)
        file_read.close()


    for model_name in model_name_list:
        # 获取类别
        label_path = os.path.join(args.ref_dir, "file_label_dict.pkl")
        with open(label_path, 'rb') as fo:
            label_dict = pickle.load(fo)
        rev_dict = {v: k for k, v in label_dict.items()}
        valid_labels = list(set(result_holder[model_name]['label']) | set(result_holder[model_name]['result']))
        target_names = [rev_dict[i][0]+"-"+rev_dict[i][-8:-4] for i in valid_labels]

        save_path = f'{root_path}/{model_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 基本评估
        plot_classification_report(result_holder[model_name], target_names, valid_labels,  # label_dict,
                                   title=f"results/{model_name} Crop Classification Results",
                                   out_path=f'{save_path}/result_report.png', save=save_resutlt)
        # 混淆矩阵
        plot_confusion_matrix(result_holder[model_name], target_names,
                              title=f"{model_name} Crop Classification Results",
                              out_path=f'{save_path}/confusion_matrix.png', save=save_resutlt)

        # 文本报告
        txt_report = classification_report(result_holder[model_name]['label'], result_holder[model_name]['result'])
        print(txt_report)

    # 条形图
    vis_melt_metrix(result_holder, target_names,
                    valid_labels,
                    explore_type='f1-score',
                    title=f"Bar Chart Results",
                    out_path=f'{root_path}/bar_chart.png', save=save_resutlt)
