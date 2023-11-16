from sys import path as sys_path

sys_path.append(r"G:\Program Files (x86)\ComputerVision") # 将上级目录加入到系统路径中


from Test import PyTorch_Cuda_Verify as Test
import time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
# 输入和输出数据 使用 np.array 创建任意维度和大小的数组，然后进行各种数学运算。
x = np.array([[1,1]])
y = np.array([[0]])

##################################################################################################
# name: feed_forward
# function: 前向传播函数
# parameter: inputs：输入数据，outputs：输出数据，weights：权重
# return: mean_squared_error：均方误差
# note: 该函数用于计算前向传播的结果和均方误差
# author: xuruolun
# version: v1.0
# date: 2023-11-15
##################################################################################################

# 前向传播函数
def feed_forward(inputs, outputs, weights):       
    pre_hidden = np.dot(inputs,weights[0])+ weights[1]  # 隐藏层前的计算 np.dot 矩阵乘法
    hidden = 1/(1+np.exp(-pre_hidden))  # 隐藏层的激活函数，这里使用的是sigmoid函数 np.exp 指数函数
    pred_out = np.dot(hidden, weights[2]) + weights[3]  # 输出层的计算
    mean_squared_error = np.mean(np.square(pred_out - outputs))  # 计算均方误差 np.mean 求平均值 np.square 求平方
    return mean_squared_error # 返回均方误差


##################################################################################################
# name: update_weights
# function: 更新权重函数
# parameter: inputs：输入数据，outputs：输出数据，weights：权重，lr：学习率
# return: update_weights：更新后的权重，original_loss：原始损失
# note: lr 学习率：
#                 当lr很小时会导致梯度消失，当lr很大时会导致梯度爆炸。
#                 当lr很小时会导致训练速度很慢，当lr很大时会导致训练不稳定。
# author: xuruolun
# version: v1.0
# date: 2023-11-15
##################################################################################################
def update_weights(inputs, outputs, weights, lr): 
    original_weights = deepcopy(weights)  # 复制原始权重
    temp_weights = deepcopy(weights)  # 创建临时权重
    update_weights = deepcopy(weights)  # 创建用于更新的权重
    original_loss = feed_forward(inputs, outputs, original_weights)  # 计算原始损失
    for i, layer in enumerate(original_weights):
        for index, weight in np.ndenumerate(layer):# np.ndenumerate()函数用于获取数组中每个元素的索引及其对应的值
            temp_weights = deepcopy(weights) # 创建临时权重
            temp_weights[i][index] += 0.0001  # 对权重进行微小的改变 temp_weights[i][index]表示第i层第index个权重
            _loss_plus = feed_forward(inputs, outputs, temp_weights)  # 计算改变后的损失 
            grad = (_loss_plus - original_loss)/0.0001  # 计算梯度 离散化求导
            update_weights[i][index] -= lr*grad  # 更新权重
    return update_weights, original_loss # 返回更新后的权重和原始损失

# 初始化权重
W = [
    np.array([[-0.0053, 0.3793],
              [-0.5820, -0.5204],
              [-0.2723, 0.1896]], dtype=np.float32).T, 
    np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32), 
    np.array([[ 0.1528, -0.1745, -0.1135]], dtype=np.float32).T, # np.array().T函数用于转置
    np.array([-0.5516], dtype=np.float32) # dtype=np.float32表示数组的数据类型为float32
]

# 随机初始化权重
W_random = [
    np.random.randn(2, 3).astype(np.float32), # np.random.randn()函数用于创建随机数组 
    np.random.randn(3).astype(np.float32),  # .astype()函数用于改变数组的数据类型
    np.random.randn(3, 1).astype(np.float32), 
    np.random.randn(1).astype(np.float32)
]

losses = [] #创建一个空的列表，存储每个训练周期（epoch）的损失值

Test.check_cuda() # 检查CUDA是否可用
Test.check_torch_cuda_version() # 检查PyTorch的CUDA版本

print("start training...")
# 开始训练
for epoch in range(1000):
    W_random, loss = update_weights(x,y,W_random,0.01)  # 更新权重并获取损失 
    losses.append(loss)  # 记录损失
    if epoch % 10 == 0: # 每100个周期打印一次损失
        # 绘制损失曲线，实现动态显示
        plt.ion()  # 打开交互模式
        plt.plot(losses) # plt.plot()函数用于绘制折线图
        plt.title('Loss over increasing number of epochs') # plt.title()函数用于设置图表的标题
        plt.xlabel('Epochs') # plt.xlabel()函数用于设置x轴的标签
        plt.ylabel('Loss') # plt.ylabel()函数用于设置y轴的标签
        plt.pause(0.1) # plt.pause()函数用于暂停程序
        print("epoch: ", epoch, "loss: ", loss) # 打印epoch和loss
        