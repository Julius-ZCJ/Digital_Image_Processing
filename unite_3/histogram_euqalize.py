# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:31:03 2019

@author: Julius Zhu
"""

import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import time

def histogram_equalization(img):
    h, w = img.shape[:2]
    pdf = defaultdict(lambda:0)
    new_pdf = defaultdict(lambda:0)
    for i in range(h):
        for j in range(w):
            pdf[img[i][j]] = pdf[img[i][j]] + 1
    def plot(pdf, max_y = 0.1):
        x = np.arange(0, 256, 1)
        y = [pdf[xi]/(h*w) for xi in x] 
        # print(y)
        plt.plot(x, y, 'r', linewidth=1)
        plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
        plt.title(u'{}直方图'.format('pdf'))
        plt.xlim(0, 255), plt.ylim(0, max_y)
        plt.show()
    plot(pdf)
    dis_prob = []
    param = 255/(h*w)
    prob = 0
    S_k = []
    for i in range(256):
        prob = prob + pdf[i] 
        dis_prob.append(prob)
        new_grey_level = int(param*prob+0.5)
        S_k.append(new_grey_level)
        new_pdf[new_grey_level] = new_pdf[new_grey_level] + pdf[i]
    plot(new_pdf)
    plot(dis_prob, 1)
    
    new_img = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            new_img[i][j] = S_k[img[i][j]]
    
    return new_img

def histogram_specification(img, spec_pdf):
    h, w = img.shape[:2]
    pdf = defaultdict(lambda:0)
    new_pdf = defaultdict(lambda:0)
    for i in range(h):
        for j in range(w):
            pdf[img[i][j]] = pdf[img[i][j]] + 1
    dis_prob = []
    param = 255/(h*w)
    prob = 0
    S_k = []
    for i in range(256):
        prob = prob + pdf[i] 
        dis_prob.append(prob)
        new_grey_level = int(param*prob+0.5)
        S_k.append(new_grey_level)
        new_pdf[new_grey_level] = new_pdf[new_grey_level] + pdf[i] # 直方图变换后的概率密度
    G_k = []
    prob = 0
    for i in range(256):
        prob = prob+spec_pdf[i]
        G_k.append(int(255*prob+0.5))  # 根据指定的概率密度函数求出各个灰度级
    
    grey_level = []
    start = 0
    #print('S_k', S_k)
    #print('G_k', G_k)
    for i in range(256):
        a = S_k[i]
        min_grey = 255
        jj = 0
        flag = True
        for j in range(start, 256):
            b = abs(G_k[j] - a)
            if b < min_grey:
                min_grey=b
                jj = j
                if min_grey == 0:
                    grey_level.append(jj)
                    flag = False
                    break
     
        if flag:
            grey_level.append(jj)
    #print('grey_level', grey_level)
            
    new_img = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            new_img[i][j] = grey_level[img[i][j]]
            
    return new_img

def main():
    path = r'..\ImageData\ch03\DIP3E_CH03_Original_Images\Fig0316(4)(bottom_left).tif'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = time.time()
    new_img = histogram_equalization(img)
    print(time.time() - t1)
    cv2.imwrite('./results/histogram_equalization_2.jpg', new_img)
        
if __name__=='__main__':
    path = r'..\ImageData\ch03\DIP3E_CH03_Original_Images\Fig0316(4)(bottom_left).tif'
    path1 = r'..\ImageData\ch03\DIP3E_CH03_Original_Images\Fig0316(1)(top_left).tif'
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape[:2]
    pdf = defaultdict(lambda:0)
    new_pdf = defaultdict(lambda:0)
    for i in range(h):
        for j in range(w):
            pdf[img1[i][j]] = pdf[img1[i][j]] + 1
    #print(pdf)
    pdf_list = []
    for i in range(256):
        a = pdf[i]/(h*w)
        pdf_list.append(a)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = time.time()
    new_img = histogram_specification(img, pdf_list)
    print(time.time() - t1)
    cv2.imwrite('./results/histogram_specification.jpg', new_img)
    