"""
    class ModelInterface(pl.LightningModule):类，作为模型的中间接口。
    __init__()函数中import相应模型类，然后老老实实加入configure_optimizers, training_step, validation_step等函数，
    用一个接口类控制所有模型。不同部分使用输入参数控制。

    为了实现自动加入新model和dataset而不用更改Interface，model文件夹中的模型文件名应该使用snake case命名，如rdn_fuse.py，
    而文件中的主类则要使用对应的驼峰命名法，如RdnFuse。
"""
import inspect
import numpy as np
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.nn import functional as F


class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.network_module = None
        self.criterion = None
        self.save_hyperparameters()  # 储存init中输入的所有超参。后续访问可以由self.hparams.argX方式进行。同时，超参表也会被存到文件中。
        self.load_network_module()
        self.configure_optimizers()
        self.configure_loss()

    def load_network_module(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.network_module = getattr(importlib.import_module('.'+name, package=__package__), camel_name)
        except :
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.instancialize()

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        # inspect.getargspec 获取函数参数的名称和默认值，返回一个命名的元组
        class_args = inspect.getfullargspec(self.network_module.__init__).args[1:]
        inside_args = self.hparams.keys()  # 内部参数
        all_args = {}
        for arg in class_args:
            if arg in inside_args:
                all_args[arg] = getattr(self.hparams, arg)
        all_args.update(other_args)
        self.network_module = self.network_module(**all_args)
    # def instancialize(self, Model, **other_args):
    #     """ Instancialize a model using the corresponding parameters
    #         from self.hparams dictionary. You can also input any args
    #         to overwrite the corresponding value in self.hparams.
    #     """
    #     class_args = inspect.getargspec(Model.__init__).args[1:]
    #     inkeys = self.hparams.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = getattr(self.hparams, arg)
    #     args1.update(other_args)
    #     return Model(**args1)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        criterion = self.hparams.criterion.lower()
        if criterion == 'mse':
            self.criterion = F.mse_loss
        elif criterion == 'l1':
            self.criterion = F.l1_loss
        elif criterion == 'bce':
            self.criterion = F.binary_cross_entropy
        elif criterion == 'ce':
            self.criterion = F.cross_entropy
        else:
            raise ValueError("Invalid Criterion Type!")

    def forward(self, X):
        return self.network_module(X)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch  # 对应于Dataset中的__getitem__()
        outputs = self.forward(inputs)
        outputs = outputs.squeeze(1)
        labels = torch.argmax(labels.squeeze(1), dim=-1).view(-1)
        loss = self.criterion(outputs, labels)

        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        outputs = outputs.squeeze(1)
        labels = torch.argmax(labels.squeeze(1), dim=-1).view(-1)
        loss = self.criterion(outputs, labels)

        _, predicted = torch.max(outputs, dim=1)  # value index
        acc = predicted.eq(labels.view_as(predicted)).squeeze().sum().item() / (len(predicted) * 1.0)

        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}


    def training_epoch_end(self, outputs):
        # 添加网络参数直方图Adding Histograms iterating through all parameters
        # for name, params in self.network_module.named_parameters():
        #     self.logger.experiment.add_histogram(name, params, self.current_epoch)
        if self.current_epoch == 0:
            # if self.hparams.computational_graph:
            #     # 添加计算图Adding Computational graph
            #     self.logger.experiment.add_graph(self.network_module,
            #                                      torch.rand((1, self.hparams.sample_length // (self.hparams.input_size[0] * self.hparams.input_size[1]),
            #                                                  self.hparams.input_dim, self.hparams.input_size[0], self.hparams.input_size[1])).type(
            #                                          self.hparams.dtype))
            # # 计算模型参数总量
            total = sum(param.numel() for param in self.network_module.parameters()) // 10000  # 单位（万）
            self.logger.experiment.add_scalar('param_num', int(total), self.current_epoch)
        return


        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        # creating log dictionary
        tensorboard_logs = {'loss': avg_loss, "Accuracy": correct / total}

        epoch_dictionary = {
            # required
            'loss': avg_loss,

            # for logging purposes
            'log': tensorboard_logs}

        return epoch_dictionary

    def validation_epoch_end(self, outputs):
        print()

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


if __name__ == '__main__':
    getattr(importlib.import_module('.' + 'convgru_net', package=__package__), camel_name)
    exit()

    parser = ArgumentParser()
    parser.add_argument('--model_name', default='gru_net', type=str)
    parser.add_argument('--criterion', default='bce', type=str)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], default='step', type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    args = parser.parse_args()
    dataset_module = ModelInterface(**vars(args))  # 网络模型

    inputs = torch.Tensor(np.random.rand(864).reshape(1, 1, -1)).float()
    outputs = dataset_module.forward(inputs)
    print(outputs, outputs.shape)
