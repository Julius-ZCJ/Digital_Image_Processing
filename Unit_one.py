# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:01:15 2019

@author: Julius Zhu
"""

from PIL import Image
import numpy as np
import os
import time
import cv2

class ImageInsert():
    
    def __init__(self):
        pass
        
    @classmethod
    def imread(cls, img_path):
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            return img
        else:
            # print('{} is not exist, please input correct path'.format(self.impath))
            raise NameError('{} is not exist, please input correct path'.format(img_path))
        
    @staticmethod
    def nearest_insert(img, new_shape):
        h, w, c= img.shape
        new_h, new_w = new_shape
        new_shape = [i for i in new_shape]
        new_shape.append(c)
        new_img = np.zeros(new_shape, dtype=np.uint8)  # the type is very imporant
        h_rate = h/new_h
        w_rate = w/new_w
        for i in range(new_h):
            for j in range(new_w):
                img_h = int(i*h_rate)
                img_w = int(j*w_rate)
                new_img[i][j] = img[img_h][img_w]

        return new_img
    
    @staticmethod
    def double_lines_insert(img, new_shape):
        """
        1. 使用矩阵得到：
        value = (x1, x2)*[(a,b),(c,d)]*(y1,y2)T
        2. 使用分步计算：
        value1 = 
        """
        h,w,c = img.shape
        new_h, new_w = new_shape
        new_shape = [i for i in new_shape]
        new_shape.append(c)
        new_img = np.zeros(new_shape, dtype=np.uint8)
        h_rate = h/new_h
        w_rate = w/new_w
        for i in range(new_h):
            for j in range(new_w):
                for k in range(3):
                    img_h = (i + 0.5) * h_rate - 0.5# 使用几何中心对称，这点很重要
                    img_w = (j + 0.5) * w_rate - 0.5
                    
                    int_img_w_0 = int(np.float(img_w))
                    int_img_h_0 = int(np.float(img_h))
                    int_img_w_1 = min(int_img_w_0 + 1, w - 1)
                    int_img_h_1 = min(int_img_h_0 + 1, h - 1)
                    
                    x_vectory = np.array([int_img_w_1-img_w, img_w-int_img_w_0])
                    value_vectory = np.array([[img[int_img_h_0][int_img_w_0][k], img[int_img_h_1][int_img_w_0][k]],
                                              [img[int_img_h_0][int_img_w_1][k], img[int_img_h_1][int_img_w_1][k]]])
                    y_vectory = np.array([img_h-int_img_h_0, int_img_h_1-img_h])
                    value = np.dot(np.dot(x_vectory, value_vectory), y_vectory.T)
                    
                    
#                    value_1 = (int_img_w_1-img_w)*img[int_img_h_0][int_img_w_0][k]+(img_w-int_img_w_0)*img[int_img_h_0][int_img_w_1][k]
#                    value_2 = (int_img_w_1-img_w)*img[int_img_h_1][int_img_w_0][k]+(img_w-int_img_w_0)*img[int_img_h_1][int_img_w_1][k]
#                    value = (int_img_h_1-img_h)*value_2+(img_h-int_img_h_0)*value_1
                    new_img[i][j][k] = value
                
        new_img = np.array(new_img, dtype=np.uint8)
        return new_img
    
    @staticmethod
    def double_three_insert(img, new_shape):
        
        h,w,c = img.shape
        new_h, new_w = new_shape
        new_shape = [i for i in new_shape]
        new_shape.append(c)
        new_img = np.zeros(new_shape, dtype=np.uint8)
        h_rate = h/new_h
        w_rate = w/new_w
        for i in range(new_h):
            for j in range(new_w):
                img_h = i * h_rate 
                img_w = j * w_rate 
                weight_h = ImageInsert._get_weight(img_h)
                weight_w = ImageInsert._get_weight(img_w)
                img_h_int = int(img_h)
                img_w_int = int(img_w)
                value = np.zeros((3,), np.uint8)
                for ii in range(4):
                    for jj in range(4):
                        h_min = min(img_h_int+ii-1, h-1)
                        w_min = min(img_w_int+jj-1, w-1)
                        value = value + img[h_min][w_min]*weight_h[ii]*weight_w[jj]
                        
                new_img[i][j] = value
        return new_img
                
    @staticmethod                
    def _get_weight(x, a=0.5):
        X = int(x)
        x1 = 1 + (x-X)
        x2 = x - X
        x3 = 1 - (x - X)
        x4 = 2 - (x - X)
        w_x = []
        
        w_x_1 = a*abs(x1 * x1 * x1) - 5 * a*x1 * x1 + 8 * a*abs(x1) - 4 * a;
        w_x_2 = (a + 2)*abs(x2 * x2 * x2) - (a + 3)*x2 * x2 + 1;
        w_x_3 = (a + 2)*abs(x3 * x3 * x3) - (a + 3)*x3 * x3 + 1;
        w_x_4 = a*abs(x4 * x4 * x4) - 5 * a*x4 * x4 + 8 * a*abs(x4) - 4 * a;
        w_x = [w_x_1, w_x_2, w_x_3, w_x_4]
        
        return w_x
    
    
if __name__=='__main__':
    path = './ImageData/test.jpg'
    new_shape = (1440, 2560)
    img = ImageInsert.imread(path)
    t1 = time.time()
    new_img = ImageInsert.double_three_insert(img, new_shape)
    new_img = Image.fromarray(new_img, 'RGB')
    print(time.time()-t1)
    # new_img.show()   
    new_img.save('./ImageData/double_three_new.jpg')
    
#    img = cv2.imread(path)
#    t2 = time.time()
#    img = cv2.resize(img, (2560, 1440), cv2.INTER_LINEAR) # 宽，高
#    cv2.imwrite('./ImageData/cv_INTER_LINEAR.jpg', img)
#    print(img.shape, time.time()-t2)
    
    
    
    
    
    
    