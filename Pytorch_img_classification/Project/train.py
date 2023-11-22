import os, os.path as osp
import sys
import cv2 as cv, matplotlib.pyplot as plt
import numpy as np


def show_images(images, titles=None, num_cols=None, scale=3, normalize=False):
    """ 一个窗口中绘制多张图像:
    Args: 
        images: 可以为一张图像(不要放在列表中)，也可以为一个图像列表
        titles: 图像对应标题、
        num_cols: 每行最多显示多少张图像
        scale: 用于调整图窗大小
        normalize: 显示灰度图时是否进行灰度归一化
    """

    # 加了下面2行后可以显示中文标题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 单张图片显示
    if not isinstance(images, list):
        if not isinstance(scale, tuple):
            scale = (scale, scale * 1.5)

        plt.figure(figsize=(scale[1], scale[0]))
        img = images
        if len(img.shape) == 3:
            # opencv库中函数生成的图像为BGR通道，需要转换一下
            B, G, R = cv.split(img)
            img = cv.merge([R, G, B])
            plt.imshow(img)
        elif len(img.shape) == 2:
            # pyplot显示灰度需要加一个参数
            if normalize:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise TypeError("Invalid shape " +
                            str(img.shape) + " of image data")
        if titles is not None:
            plt.title(titles, y=-0.15)
        plt.axis('off')
        plt.show()
        return

    # 多张图片显示
    if not isinstance(scale, tuple):
        scale = (scale, scale)

    num_imgs = len(images)
    if num_cols is None:
        num_cols = int(np.ceil((np.sqrt(num_imgs))))
    num_rows = (num_imgs - 1) // num_cols + 1

    idx = list(range(num_imgs))
    _, figs = plt.subplots(num_rows, num_cols,
                           figsize=(scale[1] * num_cols, scale[0] * num_rows))
    for f, i, img in zip(figs.flat, idx, images):
        if len(img.shape) == 3:
            # opencv库中函数生成的图像为BGR通道，需要转换一下
            B, G, R = cv.split(img) # 将BGR通道分离
            img = cv.merge([R, G, B]) # 将BGR通道转换成RGB通道
            f.imshow(img)
        elif len(img.shape) == 2:
            # pyplot显示灰度需要加一个参数
            if normalize:
                f.imshow(img, cmap='gray')
            else:
                f.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise TypeError("Invalid shape " +
                            str(img.shape) + " of image data")
        if titles is not None:
            f.set_title(titles[i], y=-0.15)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # 将不显示图像的fig移除，不然会显示多余的窗口
    if len(figs.shape) == 1:
        figs = figs.reshape(-1, figs.shape[0])
    for i in range(num_rows * num_cols - num_imgs):
        figs[num_rows - 1, num_imgs % num_cols + i].remove()
    plt.show()




script_dir = osp.dirname(osp.abspath(__file__)) # 获取当前脚本文件所在目录
project_dir = osp.dirname(script_dir) # 获取项目所在目录 
computer_vision_dir = osp.dirname(project_dir) # 获取computer_vision目录 

sys.path.append(computer_vision_dir) # 将computer_vision目录添加到系统路径中
img1_path = osp.join(project_dir, 'img', '13314845299835436.jpeg') # 构造图片路径
img2_path = osp.join(project_dir, 'img', '20230405-DSC_1654.JPG') # 构造图片路径
img3_path = osp.join(project_dir, 'img', '102655341_p0_master1200.jpg') # 构造图片路径

imgs_path = [img1_path, img2_path, img3_path]


img = []
img_cat = []
img_gray = []
img_gray_smaller = []
img_list = []
title_list = ['原图', '灰度图', '缩放后的灰度图',
              '原图', '灰度图', '缩放后的灰度图',
              '原图', '灰度图', '缩放后的灰度图']

for img_path in imgs_path:
    img.append(cv.imread(img_path)) # 读取图片

for index in range (len(img)):
    img_cat.append(img[index][10:3000, 10:3000]) # 裁剪坐标为[y0:y1, x0:x1]
    img_gray.append(cv.cvtColor(img_cat[index], cv.COLOR_BGR2GRAY)) # 将图片转换成灰度图
    img_gray_smaller.append(cv.resize(img_gray[index], (25, 25))) # 将图片缩放成100*100



for index in range (len(img)):
    img_list.append(img_cat[index])
    img_list.append(img_gray[index])
    img_list.append(img_gray_smaller[index])

#print(img1_gray_smaller) # 打印灰度图像素值


show_images(img_list, titles=title_list , num_cols=3, scale=2.5, normalize=True)


print(os.getcwd()) # 获取当前工作目录路径
print(script_dir) # 获取当前脚本文件所在目录
print(project_dir) # 获取项目所在目录
print(computer_vision_dir) # 获取computer_vision目录