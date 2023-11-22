import torch
import time
from sys import path as sys_path
sys_path.append(r"G:\Program Files (x86)\ComputerVision") # 将上级目录加入到系统路径中

import Test.PyTorch_Cuda_Verify as cuda_verify #导入pytorch_cuda驗證模塊

##################################################################################################
# 验证PyTorch和CUDA是否可用
cuda_verify.check_cuda() # 打印CUDA是否可用
cuda_verify.check_cudnn() # 打印cuDNN是否可用
cuda_verify.check_torch_cuda_version() # 打印PyTorch的CUDA版本
print("Pytorch_Cuda_Verify.py is finished.\n") # 打印Pytorch_Cuda_Verify.py is finished.

##################################################################################################

# 输入和输出数据 使用 torch.tensor 创建任意维度和大小的数组，然后进行各种数学运算。
x = torch.tensor([[1,2.0]]) # 创建一个1*2的张量 

y = torch.tensor([[False],[2]])  # 创建一个2*1的张量

##################################################################################################
#打印张量的形状和数据类型
print('x.shape:',x.shape)  # 打印x的形状
print('y.shape:',y.shape)  # 打印y的形状
print('x.dtype:',x.dtype)  # 打印x的数据类型
print('y.dtype:',y.dtype)  # 打印y的数据类型

#如果数据全是整数，那么输出的数据类型就是int64，如果是含浮点数，那么输出的数据类型就是float32
#张量内的数据类型必须是一致的，否则会强制转换成最通用的数据类型
print('x:',x)  # 打印x 
print('y:',y)  # 打印y
##################################################################################################

# tensor初始化

torch.zeros((2,3)) # 创建一个2*3的全0张量
torch.ones((2,3)) # 创建一个2*3的全1张量
torch.rand((2,3)) # 创建一个2*3的随机张量
torch.randn((2,3)) # 创建一个2*3的正态分布张量
torch.arange(0,10,2) # 创建一个0-10步长为2的张量
torch.linspace(0,10,6) # 创建一个0-10等分为6份的张量

torch.randint(low=0,high=10,size=(2,3)) 
# 创建一个2*3的随机整数张量 生成的随机数范围为[low,high)

torch.normal(mean=torch.zeros(2,3),std=torch.ones(2,3)) 
# 创建一个2*3的正态分布张量，均值为0，标准差为1

