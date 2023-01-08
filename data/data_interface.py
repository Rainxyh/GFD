"""
    class DatasetInterface(pl.LightningDataModule):类, 用作所有数据集文件的接口
    __init__()函数中import相应Dataset类，setup()进行实例化，并老老实实加入所需要的的train_dataloader, val_dataloader, test_dataloader函数。这些函数往往都是相似的，可以用几个输入args控制不同的部分。

    为了实现自动加入新model和dataset而不用更改Interface，model文件夹中的模型文件名应该使用snake case命名，如rdn_fuse.py，而文件中的主类则要使用对应的驼峰命名法，如RdnFuse。
"""

import inspect
import importlib
import os
import pickle
import pytorch_lightning as pl
from pathlib2 import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class DatasetInterface(pl.LightningDataModule):

    def __init__(self,
                 **kwargs):
        super().__init__()
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.data_module = None
        self.kwargs = kwargs
        self.dataset_name = kwargs['dataset']
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.load_data_module()

    def prepare_data(self) -> None:
        pass

    def load_data_module(self):
        name = self.dataset_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        self.data_module = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        # try:  # importlib.import_module('.'+name, package=__package__) 动态导入dataset文件 getattr取出对应class camel_name
        #     self.data_module = getattr(importlib.import_module('.'+name, package=__package__), camel_name)
        # except:
        #     raise ValueError(
        #         f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}!')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        # inspect.getargspec 获取函数参数的名称和默认值，返回一个命名的元组
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inside_args = self.kwargs.keys()
        all_args = {}
        for arg in class_args:
            if arg in inside_args:
                all_args[arg] = self.kwargs[arg]
        all_args.update(other_args)
        return self.data_module(**all_args)  # 用参数值初始化对应的data.Dataset类

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_set = self.instancialize(train_mode=True)
            self.valid_set = self.instancialize(train_mode=False)
        # Assign test dataset for use in dataloader(s)
        else:
            self.test_set = self.instancialize(train_mode=False)

        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
    
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):  # 加权随机采样器
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.train_set)*20)
    #     return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == '__main__':
    getattr(importlib.import_module('gearbox_data', package=__package__), 'GearboxData')
    exit()

    parser = ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--dataset', default='gearbox_data', type=str)
    parser.add_argument('--data_dir', default='{}/gearbox'.format(os.getcwd()), type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--standard_scalar', default=True, type=bool)
    parser.add_argument('--class_num', default=2, type=int)
    parser.add_argument('--train_mode', default=True, type=bool)
    parser.add_argument('--sample_number', default=1000, type=int)
    parser.add_argument('--sample_length', default=864, type=int)

    args = parser.parse_args()
    dataset_module = DatasetInterface(**vars(args))  # 数据集模型
    dataset_module.setup(stage='fit')
    train_loader = dataset_module.train_dataloader()
    for i in train_loader:
        print(i[0].shape, i[1].shape)
        exit()

