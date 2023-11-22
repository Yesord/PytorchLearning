import os
import os.path as osp
import sys
root_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))) # 获取项目根目录
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__))) # 获取当前目录的父目录
sys.path.append(root_dir) # 将项目根目录添加到系统路径中


print
# 以下代码用于验证PyTorch是否安装成功 cuda是否可用
import Test.PyTorch_Cuda_Verify as cuda_verify
device = cuda_verify.check_cuda()

######################################################



from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms

data_folder = parent_dir + '/datasets' # 数据集路径

fmnist = datasets.FashionMNIST(data_folder, download=False, train=True) # 下载fmnist训练集

tr_iamges = fmnist.data # 训练集图像
tr_targets = fmnist.targets  # 训练集标签


######################################################
# name: FMNISTDataset  
# function: 将数据集转换成Dataset类
# params: x - 图像数据
#         y - 图像标签
#         transform - 数据转换
# note: 数据集类必须包含__len__和__getitem__两个方法
#       __len__方法返回数据集的大小，__getitem__方法返回数据集中的数据
#       __init__方法用于初始化数据集
######################################################
class FMNISTDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y





# 测试代码
if __name__ == '__main__':
    print("\n########## Pytorch_img_classification/Project/train.py ##########\n")
    print(f"root_dir:{root_dir}") # print(f"{}") 是Python3.6中的新特性，称之为f-string，是一种方便的字符串格式化方法
    print(f"device: {device}")
    cuda_verify.check_cudnn()
    cuda_verify.check_torch_cuda_version()
    print("\n########## Pytorch_img_classification/Project/train.py ##########\n")