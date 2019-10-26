# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:26:09 2019

@author: Julius Zhu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
def img_reverse(img):
    """
    1.直接相减时np的广播机制
    2.暂时还不知道怎么用python查看图片的灰度级，或者查看是几比特图像，只能写死为8bit图像。
    """
    new_img = np.array(img.shape, np.uint8)
    new_img = 255 - img
    return new_img
    
def unknow_transform(img):
    """
    不知到这个变换叫什么名字，是从别人博客上看到的
    根据公式 new_x = x**2/255,变换得到
    根据公式可以看出这个变换有拉伸对比度的作用，如果将1/255提出来就是指数变化，
    除以255只是为了将像素值约束在255以内（个人理解）
    """
    def plot():
        x = np.arange(0, 256, 0.01)
        y = x**2/255
        plt.plot(x, y, 'r', linewidth=1)
        plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
        plt.title(u'指数变换函数')
        plt.xlim(0, 255), plt.ylim(0, 255)
        plt.show()
    plot()
    h, w = img.shape[:2]
    new_img = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            new_img[i][j] = img[i][j]**2/255
            
    return new_img

def logarithm_transform(img, constant=45):
    """
    1.常数c的作用是将像素值拉升到0-255范围之间（接近），并不是随意设置的
    2.对数变换对整体像素偏低的图像增强效果明显，用书上的话说：
    将输入范围较窄的底灰度值映射为输出范围较宽的灰度值，或将范围较宽的高灰度值映射为输出较窄的灰度值，
    我们使用这种类型的变换来扩展图像中的暗像素值，同时压缩更高灰度的值。
    在傅里叶频谱中有重要作用。
    """
    
    def log_plot(c):
        x = np.arange(0, 256, 0.01)
        y = c * np.log(1 + x)
        plt.plot(x, y, 'r', linewidth=1)
        plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
        plt.title(u'对数变换函数')
        plt.xlim(0, 255), plt.ylim(0, 255)
        plt.show()
        
    log_plot(constant)
    new_img = np.zeros(img.shape,np.uint8)

    new_img = np.uint8(constant*np.log(1+img)+0.5)
    return new_img

def power_transform(img, constant_c, constant_r):
    """
    幂律变换又称为伽马变换（gamma）
    1.调节r值改变函数形状，调节c值为了将像素映射到0-255
    2.
    当γ>1时，会拉伸图像中灰度级较高的区域，压缩灰度级较低的部分。(过曝)
    当γ<1时，会拉伸图像中灰度级较低的区域，压缩灰度级较高的部分。（过暗）
    当γ=1时，该灰度变换是线性的，此时通过线性方式改变原图像。
    总体而言就是增强对比度。
    处理图像对比度偏低的有良好效果
    """
    def gamma_plot(c, v):
        x = np.arange(0, 256, 0.01)
        y = c*x**v
        plt.plot(x, y, 'r', linewidth=1)
        plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
        plt.title(u'伽马变换函数')
        plt.xlim([0, 255]), plt.ylim([0, 255])
        plt.show()
        
    gamma_plot(constant_c, constant_r)
    
    table = np.zeros(256, np.float)
    for i in range(256):
        table[i] = constant_c*i**constant_r
    new_img = np.uint8(cv2.LUT(img, table)+0.5)
    return new_img

def contrastStretch(img, coord_1, coord_2):
    x1 = coord_1[0]
    x2 = coord_2[0]
    k1 = coord_1[1]/coord_1[0]
    k2 = (coord_2[1]-coord_1[1])/(coord_2[0]-coord_1[0])
    b2 = coord_1[1] - k2*coord_1[0]
    k3 = (255 - coord_2[1])/(255 - coord_2[0])
    b3 = coord_2[1] - k3*coord_2[0]
    def show_plot():
        xx = np.arange(0, 255, 0.01)
        yy = []
        y = 0
        for x in xx:
            if x <= x1:
                y = k1*x
            elif x > x1 and x < x2:
                y = k2*x + b2
            else:
                y = k3*x + b3
            yy.append(y)
        plt.plot(xx, yy, 'r', linewidth=1)
        plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
        plt.title(u'分段函数函数')
        plt.xlim([0, 255]), plt.ylim([0, 255])
        plt.show()
    
    show_plot()
    h, w = img.shape[:2]
    new_img = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i][j] <= x1:
                new_img[i][j] = np.uint8(k1*img[i][j])
            elif img[i][j] > x1 and img[i][j] < x2:
                new_img[i][j] = np.uint8(k2*img[i][j] + b2)
            else:
                new_img[i][j] = np.uint8(k3*img[i][j] + b3)
    
    return new_img
    
    
def main():
    
    
    path = r'../ImageData\ch02\bubbles.tif'
    img =cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    unknow_transform(img)
    
    path = r'../ImageData\ch03\spectrum.tif'
    img =cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logarithm_transform(img, 45)
    
    path = r'../ImageData\ch02\bubbles.tif'
    img =cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    power_transform(img, 0.000015, 3.0)
    
    
    

if __name__=='__main__':
    path = r'../ImageData\ch03\breast.tif'
    path = 'logarithm_thansform.jpg'
    img =cv2.imread(path)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = contrastStretch(img, (96, 32), (160, 224))
    cv2.imwrite('contrastStretch.jpg', new_img)