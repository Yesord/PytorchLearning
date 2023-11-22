import torch

# # 张量的运算

x = torch.tensor([[1,2,3,4],[5,6,7,8],[9,0,1,4]]) # 创建一个1*2的张量
y= x.T # 转置
print('x=\n',x)  # 打印x
print('x**2=\n',x**2)  # 将x的每个元素平方
print('y+1=\n',y+1)  # 将x的每个元素加1


