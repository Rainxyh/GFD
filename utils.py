import os
from pathlib2 import Path


def load_model_path(root=None, version=None, version_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        version_num: The version's number that you are going to load.
        best: Whether return the best model.
    Return:
        checkpoint file name: type=str
    """
    def sort_by_epoch(path):
        name = path.stem  #stem不带后缀文件名 name带后缀文件名 suffix后缀
        epoch = int(name.split('-')[1].split('=')[1])  # best-epoch={}-val_acc={}
        return epoch

    # 生成模型路径
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:  # root 与 version均为空
            return str(Path('lightning_logs', f'version_{version_num}', 'checkpoints'))

    if root is None and version is None and version_num is None:
        return None

    root = generate_root()
    if Path(root).is_file():  # 若为文件 则直接返回ckpt文件
        return root
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)  # 选取Epoch最大且Best的checkpoint
        res = str(files[0])
    else:  # 若不需要Best 直接返回最新的checkpoint
        res = str(Path(root) / 'last.ckpt')
    return res


def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, version_num=args.load_version_num)
