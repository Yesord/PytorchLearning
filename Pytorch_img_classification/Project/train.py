import os
import os.path as osp
import sys
root_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))) # 获取项目根目录
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__))) # 获取当前目录的父目录
sys.path.append(root_dir) # 将项目根目录添加到系统路径中



# 以下代码用于验证PyTorch是否安装成功 cuda是否可用
import Test.PyTorch_Cuda_Verify as cuda_verify
device = cuda_verify.check_cuda()

######################################################


import time
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
from torchvision import datasets, transforms
from torch.optim import SGD

data_folder = parent_dir + '/datasets' # 数据集路径
print(f"data_folder:{data_folder}")
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True) # 下载fmnist训练集

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
        x = x.float() / 255 # 归一化 0-255 -> 0-1 好处：1.加快训练速度 2.限制变量范围，防止梯度爆炸 
        x = x.view(-1, 28*28) # view()函数用于将一个多行的Tensor,拼接成一行
        self.x, self.y = x, y
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x.to(device), y.to(device)

##########################################################################################################################################
# name: get_data
# function: 获取数据
# params: None
# return: tr_loader - 训练集
##########################################################################################################################################

def get_data():
    # 数据转换
    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # 获取训练集
    tr_dataset = FMNISTDataset(tr_iamges, tr_targets)
    tr_loader = DataLoader(tr_dataset, batch_size=10000, shuffle=True)
    return tr_loader

##########################################################################################################################################
# name: get_model
# function: 获取模型
# params: None
# return: model - 模型
#         loss_fn - 损失函数
#         optimizer - 优化器
##########################################################################################################################################

def get_model():
    # 定义模型
    model = nn.Sequential(
        nn.Linear(28*28, 1000), # 输入层
        nn.ReLU(), # 激活函数
        nn.Linear(1000, 10) # 输出层
    ).to(device) #创建一个神经网络模型，并将其移动到GPU上
    loss_fn = nn.CrossEntropyLoss() # 定义损失函数 CrossEntropyLoss()交叉熵损失函数
    optimizer = SGD(model.parameters(), lr=1e-2) # 定义优化器
    return model, loss_fn, optimizer

##########################################################################################################################################
# name: train_batch
# function: 训练一个批次的数据 
# params: x - 图像数据
#         y - 图像标签 
#         model - 模型 
#         loss_fn - 损失函数
#         optimizer - 优化器
# return: batch_loss.item() - 返回损失值
##########################################################################################################################################

def train_batch(x, y, model, loss_fn, optimizer):
    model.train() # 将模型设置为训练模式
    prediction = model(x) # 前向传播
    batch_loss = loss_fn(prediction, y) # 计算损失
    batch_loss.backward() # 反向传播
    optimizer.step() # 更新参数
    optimizer.zero_grad() # 清空梯度
    return batch_loss.item() # 返回损失值


##########################################################################################################################################
# name: accuracy
# function: 计算准确率
# params: x - 图像数据
#         y - 图像标签
#         model - 模型
# return: is_correct.cpu().numpy().tolist() - 返回预测结果
##########################################################################################################################################

@torch.no_grad() # 该装饰器表示该函数不会计算梯度 
#装饰器的作用是在不改变原函数的情况下，为函数添加新的功能
def accuracy(x, y, model):
    model.eval() # 将模型设置为评估模式
    with torch.no_grad():
        # get the prediction matrix for a tensor of `x` images
        prediction = model(x)
    max_values ,argmaxes = prediction.max(-1)   # 返回每一行中最大值的那个元素，且返回其索引 -1表示最后一维
    is_correct = argmaxes == y  # 判断预测结果是否正确 argmaxes和y比较之后返回值给到is_correct
    return is_correct.cpu().numpy().tolist()
##########################################################################################################################################
# name: train
# function: 训练模型
# params: None
# return: None
##########################################################################################################################################

def train():
    trl_dl = get_data() # 获取训练集
    model, loss_fn, optimizer = get_model() # 获取模型

    loss_history, accuracy_history = [], [] # 定义损失列表和准确率列表
    print("\n************train start************\n")
    epoch_max = 10 # 定义最大迭代次数
    for epoch in range(epoch_max):
        print("epoch: ", epoch+1, "/", epoch_max, "\n")
        epoch_loss, epoch_accuracy = [], []
        for ix, batch in enumerate(iter(trl_dl)):# enumerate()函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            x, y = batch
            
            batch_loss = train_batch(x, y, model, loss_fn, optimizer) # 训练一个批次的数据返回损失值
            epoch_loss.append(batch_loss) # 将损失值添加到损失列表中
        epoch_loss = np.array(epoch_loss).mean() # 一轮迭代完成后计算损失 np.array()函数用于创建数组，mean()函数用于计算数组中元素的算术平均值
        for ix, batch in enumerate(iter(trl_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            epoch_accuracy.extend(is_correct) # extend()函数用于在列表末尾一次性追加另一个序列中的多个值
        epoch_accuracy = np.mean(epoch_accuracy) # 一轮迭代完成后计算准确率 np.mean()函数用于计算数组中元素的算术平均值
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)
    epochs = np.arange(1, epoch_max+1) # np.arange()函数用于创建一个数组，其中包含一个等差序列的所有值
    plt.figure(figsize=(12, 4)) # 设置画布大小
    plt.subplot(121) # 设置子图 1行2列第1个
    plt.title("Loss value over increasing epochs") # 设置标题
    plt.plot(epochs, loss_history, label='Training Loss')
    plt.legend() # 显示图例
    plt.subplot(122)
    plt.title("Accuracy value over increasing epochs")
    plt.plot(epochs, accuracy_history, label='Training Accuracy')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    #plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    # plt.gca()函数用于获取当前坐标轴，set_yticklabels()函数用于设置y轴刻度标签
    plt.legend()    
    plt.show() # 显示图像
    return model
# 测试代码
if __name__ == '__main__':
    print("\n########## Pytorch_img_classification/Project/train.py ##########\n")
    print(f"root_dir:{root_dir}") # print(f"{}") 是Python3.6中的新特性，称之为f-string，是一种方便的字符串格式化方法
    print(f"device: {device}")
    cuda_verify.check_cudnn()
    cuda_verify.check_torch_cuda_version()
    # 开始计时
    start = time.time()
    model = train()
    # 结束计时
    end = time.time()
    # 获取train.py的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

# 构造模型文件的路径
    directory = os.path.join(script_dir, 'weight')

# 如果目录不存在，创建它
    if not os.path.exists(directory):
        os.makedirs(directory)

# 保存模型参数
    torch.save(model.state_dict(), os.path.join(directory, 'FMNIST_Pytorch.pth')) # 保存模型参数是以当前路径为参考路径的
    print("\n########## Pytorch_img_classification/Project/train.py ##########\n")