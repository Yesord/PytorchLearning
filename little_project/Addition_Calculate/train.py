import os
from sys import path as sys_path

script_path = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本的绝对路径
parent_path = os.path.dirname(os.path.dirname(script_path)) # 获取当前脚本的上级目录
print(parent_path)
sys_path.append(parent_path) # 将上级目录加入到系统路径中
#sys_path.append(os.path.dirname("G:\Program Files (x86)\ComputerVision\Test"))

from torch.utils.data import Dataset, DataLoader # 导入数据集和数据加载器
import torch
import torch.nn as nn # 导入神经网络模块
from torch.optim import SGD # 导入随机梯度下降优化器
import time # 导入时间模块z
import Test.PyTorch_Cuda_Verify as cuda_verify #导入pytorch_cuda驗證模塊
import matplotlib.pyplot as plt # 导入绘图模块

cuda_verify.check_cuda() # 打印CUDA是否可用
cuda_verify.check_cudnn() # 打印cuDNN是否可用
cuda_verify.check_torch_cuda_version() # 打印PyTorch的CUDA版本

##################################################################################################

# 创建输入数据

x = [[1,2],[3,4],[5,6],[6,7],[8,9],[10,11],[10,12],[11,13],[12,13],[14,15],[16,17],[18,19]] #,[50,15],[70,46],[323,13],[131,515],[123123,2331],[134,4323]
y = [[3],[7],[11],[13],[17],[21],[22],[24],[25],[29],[33],[37]] #,[65],[116],[336],[646],[124454],[4457]

X = torch.tensor(x).float() # 将x转换成张量
Y = torch.tensor(y).float() # 将y转换成张量

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判断是否有cuda

X = X.to(device)  # 将x放到cuda上
Y = Y.to(device)  # 将y放到cuda上

##################################################################################################
# 创建神经网络的类
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # 创建神经网络的层就是直接调用一些nn提供的函数来实现的
        self.input_to_hidden_layer = nn.Linear(2, 10) # 输入层到隐藏层 nn.Linear(2,10)表示输入层有2个神经元，隐藏层有10个神经元
        self.hidden_to_output_layer = nn.Linear(10, 1) # 隐藏层到输出层
        self.hidden_layer_activation = nn.ReLU() # 激活函数

    def forward(self, x):
        # 前向传播 表达层与层之间的逻辑关系
        x = self.input_to_hidden_layer(x) # 输入层到隐藏层
        x = self.hidden_layer_activation(x) # 激活函数
        x = self.hidden_to_output_layer(x) # 隐藏层到输出层
        return x
    
# 创建数据集类
class MyDataset(Dataset):
    def __init__(self, x, y):#初始化数据集
        self.x = torch.tensor(x).float() # 将x转换成张量
        self.y = torch.tensor(y).float() # 将y转换成张量
        self.len = len(x) #数据集的长度
    def __getitem__(self, index): #获取数据集中的一条数据
        return self.x[index], self.y[index] #返回数据
    def __len__(self): #获取数据集的长度
        return self.len
    
##################################################################################################    
# 实例化神经网络


torch.manual_seed(0) # 设置随机数种子

mynet = MyNeuralNetwork().to(device) # 创建神经网络
print(mynet.input_to_hidden_layer.weight) # 查看输入层到隐藏层的权重

for par in mynet.parameters(): # 查看神经网络的参数
    print(par)

dataset = MyDataset(X, Y) # 实例化数据集 
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) # 实例化数据加载器 
# 从dataset加载 batch_size=2表示每次加载2条数据，shuffle=True表示打乱数据
for data in dataloader: # 从dataloader中循环取出数据
    print(data) # 打印数据

##################################################################################################
# 创建损失函数和优化器

loss_func = nn.MSELoss() # 均方误差损失函数 

optimizer = SGD(mynet.parameters(), lr=0.0015) # 随机梯度下降优化器 lr 学习率

# optimizer.zero_grad() # 梯度清零
# loss_value = loss_func(mynet(X), Y) # 计算损失
# loss_value.backward() # 反向传播
# optimizer.step() # 更新权重

##################################################################################################

loss_history = [] # 创建一个空列表，用于保存损失值
train_epoch = 10000 # 训练次数

start = time.time() # 记录开始时间
for epoch in range(train_epoch):
    for data in dataloader: # 从dataloader中循环取出数据
        x, y = data
        optimizer.zero_grad() # 梯度清零
        loss_value = loss_func(mynet(x), y) # 计算损失
        loss_value.backward() # 反向传播
        optimizer.step() # 更新权重
        loss_history.append(loss_value.item()) # 将损失值添加到列表中
    if epoch % 100 == 0 and epoch < train_epoch - 1: 
            plt.ion() # 打开交互模式
            plt.plot(loss_history) # 绘制损失曲线
            plt.title('Loss over the increasing number of epoch') # 设置标题
            plt.xlabel('Epoch') # 设置x轴标签
            plt.ylabel('Loss') # 设置y轴标签
            plt.pause(0.1) # 暂停0.1秒
    elif epoch == train_epoch - 1:
            plt.ioff() # 关闭交互模式
            plt.show() # 显示图像
end = time.time() # 记录结束时间
print("训练时间：", end - start) # 打印训练时间

val_x = [[13,19]] # 1*2的list
val_x = torch.tensor(val_x).float().to(device) # 将val_x转换成张量
prediction = mynet(val_x) # 预测val_x的值
prediction_int = round(prediction.item()) # 将预测值转换成整数
print("{} + {} = ".format(val_x[0][0],val_x[0][1]) + str(prediction_int)) # 打印预测值

# 获取train.py的路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构造模型文件的路径
directory = os.path.join(script_dir, 'weight')

# 如果目录不存在，创建它
if not os.path.exists(directory):
    os.makedirs(directory)

# 保存模型参数
torch.save(mynet.state_dict(), os.path.join(directory, 'BP_Pytorch.pth')) # 保存模型参数是以当前路径为参考路径的