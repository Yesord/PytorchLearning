
import os, os.path as osp
import sys
import cv2 as cv, matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


from torchvision import datasets
import torch

script_dir = osp.dirname(osp.abspath(__file__)) # 获取当前脚本文件所在目录
project_dir = osp.dirname(script_dir) # 获取项目所在目录 
computer_vision_dir = osp.dirname(project_dir) # 获取computer_vision目录 
sys.path.append(computer_vision_dir) # 将computer_vision目录添加到系统路径中

data_folder = project_dir + '/datasets' # 数据集路径

fmnist = datasets.FashionMNIST(data_folder, download=True, train=True) # 下载fmnist训练集

tr_images = fmnist.data # 训练集图像
tr_targets = fmnist.targets  # 训练集标签

unique_values = tr_targets.unique(sorted=True) # 获取标签中的唯一值
print(f'tr_images & tr_targets:\n\tX - {tr_images.shape}\n\tY - {tr_targets.shape}\n\tY - Unique Values: {unique_values}')
print(f'Task:\n\t{len(unique_values)}-Classification')
print(f'UNIQUE CLASSES:\n\t{fmnist.classes}')

R, C = len(tr_targets.unique(sorted=True)), 10 # 行数，列数
fig, ax = plt.subplots(R, C, figsize=(C, R)) # 创建子图
for label_class, plot_row in enumerate(ax):
    label_x_raws = np.where(tr_targets == label_class)[0] # 获取标签为label_class的图像索引
    for plot_cell in plot_row:
        plot_cell.grid(False) # 关闭网格
        plot_cell.axis('off') # 关闭坐标轴
        ix = np.random.choice(label_x_raws) # 随机选择一个索引
        x, y= tr_images[ix], tr_targets[ix]
        plot_cell.imshow(x, cmap='gray')
    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.show()









