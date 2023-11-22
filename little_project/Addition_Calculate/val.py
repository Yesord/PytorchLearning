import torch
import torch.nn as nn
from torchsummary import summary

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

model = MyNeuralNetwork().to('cuda') # 创建神经网络

static_dict = torch.load('BP_Pytorch.pth')

model.load_state_dict(static_dict) # 加载模型参数 保证模型参数一致即可，不需要保证模型结构一致
#类创建的神经网络和nn.Squential（序贯）创建的神经网络的参数保存和加载方式不同，二者不能混用

summary(model, input_data=(1,2)) # 打印模型结构

print("请输入两个数，我来帮你计算它们的和：")
x1 = float(input('请输入第一个数：'))
x2 = float(input('请输入第二个数：'))

val_x = [[x1,x2]]
val_x = torch.tensor(val_x).float().to('cuda')
prediction = model(val_x) # 预测val_x的值
prediction_int = round(prediction.item()) # 将预测值转换成整数
print("{} + {} = ".format(val_x[0][0],val_x[0][1]) + str(prediction.item())) # 打印预测值

