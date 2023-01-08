"""
	在data_interface 中建立一个class DatasetInterface(pl.LightningDataModule):
	用作所有数据集文件的接口。__init__()函数中import相应Dataset类，setup()进行实例化，并老老实实加入所需要的的train_dataloader, val_dataloader, test_dataloader函数。这些函数往往都是相似的，可以用几个输入args控制不同的部分。
"""

from .data_interface import DatasetInterface
