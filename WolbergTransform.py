# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:21:09 2019

@author: Julius Zhu
"""

import numpy as np
import cv2
import time

class Operator():
    @staticmethod
    def scale_operator(multiple):
        operator = np.zeros((3,3))
        operator[0][0] = multiple
        operator[1][1] = multiple
        operator[2][2] = 1
        return operator
    
    @staticmethod
    def rotate_operator(angle):
        operator = np.zeros((3,3))
        operator[0][0] = np.cos((angle*np.pi)/180)
        operator[1][1] = np.cos((angle*np.pi)/180)
        operator[2][2] = 1
        operator[0][1] = np.sin((angle*np.pi)/180)
        operator[1][0] = -np.sin((angle*np.pi)/180)
        return operator
    
    @staticmethod
    def bias_operator(bias_x, bias_y):
        operator = np.zeros((3,3))
        operator[0][0] = 1
        operator[1][1] = 1
        operator[2][2] = 1
        operator[2][0] = bias_y
        operator[2][1] = bias_x
        return operator
        
    @staticmethod
    def vertical_bias_operator(bias):
        operator = np.zeros((3,3))
        operator[0][0] = 1
        operator[1][1] = 1
        operator[2][2] = 1
        operator[1][0] = bias
        return operator
    
    @staticmethod
    def aclinic_bias_operator(bias):
        operator = np.zeros((3,3))
        operator[0][0] = 1
        operator[1][1] = 1
        operator[2][2] = 1
        operator[0][1] = bias
        return operator
    

def wolberg_transform(img, operator):
    
    h, w = img.shape[:2]
    print(h,w)
    new_img = np.zeros(img.shape)
    for x in range(w):
        for y in range(h):
            trans = np.dot(np.array([y, x, 1]), operator)
            try:
                new_img[int(trans[0])][int(trans[1])] = img[y][x]
            except:
                pass
    new_img = np.array(new_img, np.uint8)
    print(new_img, type(new_img))
    return new_img

if __name__=='__main__':
    path = r'ImageData\insert\test.jpg'
    operator = Operator.vertical_bias_operator(0.2)
    img = cv2.imread(path)
    new_img = wolberg_transform(img, operator)
    cv2.imwrite(r'ImageData\insert\vertical_bias_operator.jpg',new_img)















    