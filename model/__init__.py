"""
	在model_interface 中建立class ModelInterface(pl.LightningModule):类，作为模型的中间接口。
	__init__()函数中import相应模型类，然后老老实实加入configure_optimizers, training_step, validation_step等函数，
	用一个接口类控制所有模型。不同部分使用输入参数控制。
"""

from .model_interface import ModelInterface
