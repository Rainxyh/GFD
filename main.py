""" main.py 职责
    定义parser，添加parse项。（注意如果你的模型或数据集文件的__init__函数中有需要外部控制的变量，如一个random_arg，你可以直接在main.py的Parser中添加这样一项，如parser.add_argument('--random_arg', default='test', type=str)，两个Interface类会自动传导这些参数到你的模型或数据集类中。）
    选好需要的callback函数们，如自动存档，Early Stop，LR Scheduler等。
    实例化ModelInterface, DatasetInterface, Trainer。

    This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The ModelInterface and 
    DatasetInterface can be seen as transparent to all your args.    

    项目结构
    root-
        |-data
            |-__init__.py
            |-data_interface.py
            |-xxxdataset1.py
            |-xxxdataset2.py
            |-...
        |-model
            |-__init__.py
            |-model_interface.py
            |-xxxmodel1.py
            |-xxxmodel2.py
            |-...
        |-main.py
        |-utils.py
"""

import importlib
import os
# import pl_bolts.callbacks as plbc
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from torchvision import models, transforms
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.loggers import TensorBoardLogger
from model import ModelInterface
from data import DatasetInterface
from utils import load_model_path_by_args
from datetime import datetime

time_str = datetime.strftime(datetime.now(), '%y-%m-%d_%H:%M:%S')
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

def load_callbacks():
    callbacks = []
    # callbacks.append(plbc.PrintTableMetricsCallback())

    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_acc',
    #     mode='max',
    #     patience=30,
    #     min_delta=0.001
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=False
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)  # 随机种子参数
    load_path = load_model_path_by_args(args)  # 模型地址
    dataset_module = DatasetInterface(**vars(args))  # 数据集模型

    if load_path is None:
        model = ModelInterface(**vars(args))
    else:
        model = ModelInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    key_args = ''
    # If you want to change the logger's saving folder, tensorboard命令中路径名称需要与创建logger文件夹路径名称相同
    if args.model_name == 'mutilconvgru_net':
        key_args = 'pow{}_nlay{}_ksz{}_slen{}_ep{}'.format(
            args.power, args.num_layers, args.kernel_size, args.sample_length, args.max_epochs)
    if args.model_name == 'onedcnn_net':
        key_args = 'nlay{}_ocha{}_ksz{}_hdim{}'.format(args.layer_num, args.out_channels, args.kernel_sizes, args.fc_hidden_dim)
    if args.model_name == 'gru_net':
        key_args = 'idim{}_hdim{}_odim{}_fcd{}_nlay{}'.format(args.input_dim, args.hidden_num, args.output_dim, args.fc_hidden_dim, args.num_layers)

    # save_dir = 'tensorboard_logs/{}/{}'.format(args.model_name, key_args)
    save_dir = f'{args.kfold}fold_logs/{args.model_name}/{key_args}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = TensorBoardLogger(save_dir=save_dir, name=None)
    args.logger = logger
    args.callbacks = load_callbacks()

    # 自动添加所有Trainer会用到的命令行参数
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dataset_module)


if __name__ == '__main__':
    parser = ArgumentParser()

    # KFold Support
    parser.add_argument('--kfold', default=2, type=int)
    parser.add_argument('--fold_num', default=0, type=int)

    # Basic Training Control
    power = 5  # 3-5
    parser.add_argument('--power', default=power, type=int)
    num_layers = 3  # 2-4 kernel对应[(5,5), (3,3)] [(7, 7), (5, 5), (3, 3)] [(9, 9) (7, 7) (5, 5) (3, 3)] [(11, 11)..]
    parser.add_argument('--num_layers', default=num_layers, type=int)
    sensor_number = 8

    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--sample_length', default=4096, type=int)
    parser.add_argument('--class_num', default=10, type=int)
    parser.add_argument('--sensor_number', default=sensor_number, type=int)

    # Training Info
    parser.add_argument('--criterion', default='ce', type=str)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--computational_graph', default=False, type=bool)


    model_name_list = ['mutilconvgru_net', 'cnn_net', 'gru_net', 'onedcnn_net', 'cnngru_net', 'convgru_net',]
    model_name = model_name_list[0]
    # Model Hyperparameters
    if model_name == 'mutilconvgru_net':
        # Multi-ConvGRU
        parser.add_argument('--model_name', default='mutilconvgru_net', type=str)
        parser.add_argument('--input_size', default=(pow(2, power), pow(2, power)), type=tuple)
        parser.add_argument('--input_dim', default=sensor_number, type=int)
        parser.add_argument('--hidden_dim', default=sensor_number*2, type=int)
        parser.add_argument('--kernel_size', default=[tuple((2*(i+1)+1, 2*(i+1)+1)) for i in reversed(range(num_layers))], type=list)
        parser.add_argument('--dtype', default=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor, type=torch.dtype)
        parser.add_argument('--batch_first', default=True, type=bool)
        parser.add_argument('--bias', default=True, type=bool)
        parser.add_argument('--return_all_layers', default=False, type=bool)

    if model_name == 'cnn_net':
        # CNN(direct contact 2D matrix)
        parser.add_argument('--model_name', default='cnn_net', type=str)

    if model_name == 'onedcnn_net':
        # 1D-CNN
        parser.add_argument('--model_name', default='onedcnn_net', type=str)
        parser.add_argument('--layer_num', default=3, type=int)
        parser.add_argument('--in_channel', default=8, type=int)  # 输入特征维度
        parser.add_argument('--out_channels', default=[16, 32, 64, 64, 64], type=list)  # 输出特征维度
        parser.add_argument('--kernel_sizes', default=[64, 16, 3, 3, 3], type=list)
        parser.add_argument('--strides', default=[32, 8, 1, 1, 1], type=list)
        parser.add_argument('--conv_paddings', default=[0, 0, 'same', 'same', 'valid'], type=list)
        parser.add_argument('--batchnorm_num_features', default=[16, 32, 64, 64, 64], type=list)
        parser.add_argument('--pool_kernel_sizes', default=[2, 2, 2, 2, 2], type=list)
        parser.add_argument('--fc_hidden_dim', default=100, type=int)

    if model_name == 'gru_net':
        # GRU
        parser.add_argument('--model_name', default='gru_net', type=str)
        parser.add_argument('--input_dim', default=128, type=int)
        parser.add_argument('--hidden_num', default=128, type=int)
        parser.add_argument('--output_dim', default=10, type=int)
        parser.add_argument('--fc_hidden_dim', default=64, type=int)

    if model_name == 'bigru_net':
        # Bi-GRU
        parser.add_argument('--model_name', default='bigru_net', type=str)
        parser.add_argument('--input_dim', default=128, type=int)
        parser.add_argument('--hidden_num', default=32, type=int)
        parser.add_argument('--output_dim', default=10, type=int)
        parser.add_argument('--fc_hidden_dim', default=64, type=int)

    if model_name == 'cnngru_net':
        # CNN-GRU
        parser.add_argument('--model_name', default='cnngru_net', type=str)

    if model_name == 'cnnbigru_net':
        # CNN-BiGRU
        parser.add_argument('--model_name', default='cnnbigru_net', type=str)

    if model_name == 'standard_net':
        # CNN-GAF
        parser.add_argument('--model_name', default='standard_net', type=str)


    # Optimizer
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], default='step', type=str)
    parser.add_argument('--lr_decay_steps', default=50, type=int)
    parser.add_argument('--lr_decay_rate', default=0.3, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')  # action=‘store_true’，只要运行时该变量有传参就将该变量设为True
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_version_num', default=None, type=int)  # 版本号

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
    parser.add_argument('--GAFpath', default=r'{}/data/{}/GAF'.format(os.getcwd(), dataset_name).replace("\\", "/"), type=str)
    parser.add_argument('--transforms', default=data_transforms, type=dict)
    parser.add_argument('--wavelet_packet', default=True if (model_name == 'mutilconvgru_net' or model_name == 'cnn_net') else False, type=bool)
    # parser.add_argument('--', default=, type=)

    # Data Augmentation
    parser.add_argument('--aug_prob', default=0.5, type=float)  # 数据增强概率

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Deprecated, old version 混合式，既使用Trainer相关参数，又使用一些自定义参数，如各种模型超参
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100,
                        gpus=[2] if torch.cuda.is_available() else 0,
                        # accelerator='gpu|tpu|hpu',
                        # devices='cpu',
                        # precision=16,
                        # auto_weights_summaryselect_gpus=True,
                        weights_summary='full',  # ‘top’顶层模块 'full’所有模块 None不打印
                        # val_check_interval=1,  # 进行Validation测试的周期 use (float) to check within a training epoch use (int) to check every n steps (batches)
                        # limit_train_batches=1,  # 使用训练数据的百分比
                        fast_dev_run=False,
                        auto_lr_find=False,  # 当且仅当执行trainer.tune(model)代码时工作
                        # auto_scale_batch_size="binsearch",  # 搜索到的最大batch size后将会自动覆盖trainer的hparams.batch_size
                        # accumulate_grad_batches={5: 3, 10: 20},  # 从第5个epoch开始，累加3个step的梯度；从第10个epoch之后，累加20个step的梯度
                        num_sanity_val_steps=1,
                        )

    args = parser.parse_args()

    main(args)
    