import os
import os.path as osp
import sys
root_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))) # 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__))) # 获取当前目录的父目录
sys.path.append(root_dir) # 将项目根目录添加到系统路径中

# 以下代码用于验证PyTorch是否安装成功 cuda是否可用
import Test.PyTorch_Cuda_Verify as cuda_verify
device = cuda_verify.check_cuda()

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import matplotlib.pyplot as plt



def get_model():
    model = nn.Sequential(
            nn.Linear(28*28, 1000), # 输入层
            nn.ReLU(), # 激活函数
            nn.Linear(1000, 10) # 输出层
        ).to(device) #创建一个神经网络模型，并将其移动到GPU上
    
model = get_model()

static_dict = torch.load(f'{script_dir}'+'\\weight\\BP_Pytorch.pth')

model.load_state_dict(static_dict) # 加载模型参数 保证模型参数一致即可，不需要保证模型结构一致

# 以下代码用于验证模型训练效果





