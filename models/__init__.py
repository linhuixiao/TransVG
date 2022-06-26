# TODO： 从当前目录结构的文件中导入一个函数
from .trans_vg import TransVG


def build_model(args):
    return TransVG(args)
