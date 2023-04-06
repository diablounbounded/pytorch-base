import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.png'))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('img', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        #cvColor 方法用于将图像从一种颜色空间转换为另一种颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        #将图片中原有的所有像素值 reshape
        #shape[0]高度
        #shape[1]宽度
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
#        flipCode = random.choice([-1, 0, 1, 2])
#        if flipCode != 2:
#            image = self.augment(image, flipCode)
 #           label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        return len(self.imgs_path)


