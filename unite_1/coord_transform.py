# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:57:05 2019

@author: Julius Zhu
"""

import numpy as np
import cv2

def calculate_perams(perams_array, coord_array):
   result_array = np.dot(np.linalg.inv(perams_array), coord_array.T)
   return result_array

def get_array(org_coords, src_coords, flag='x'):
    coord_array = []
    perams_array = np.zeros((4, 4))
    for i, coord in enumerate(org_coords):
        perams_array[i][0] = coord[0]
        perams_array[i][1] = coord[1]
        perams_array[i][2] = coord[0]*coord[1]
        perams_array[i][3] = 1
        if flag == 'x':
            coord_array.append(src_coords[i][0])
        elif flag == 'y':
            coord_array.append(src_coords[i][1])
        else:
            print('flag value is incorrect!')
    return perams_array, np.array(coord_array)

def transform(img, org_coords, src_coords):
    """
    1.尝试使用反向映射，但是计算化简有点困难，以后再说
    2.可以先将坐标系平移到图像中心，在进行变化，在平移回来。
    """
    h, w = img.shape[:2]
    print(h, w)
    rate_w = max((src_coords[1][0]-src_coords[0][0]), (src_coords[2][0]-src_coords[3][0]))/(org_coords[1][0]-org_coords[0][0])
    rate_h = max((src_coords[2][1]-src_coords[1][1]), (src_coords[3][1]-src_coords[0][1]))/(org_coords[2][1]-org_coords[1][1])
    print(rate_w, rate_h, (int(h*rate_h), int(w*rate_w), 3))
    new_img = np.zeros((int(h*rate_h), int(w*rate_w), 3), np.uint8)
    
    y_perams_array, y_coord_array = get_array(org_coords, src_coords, flag='y')
    x_perams_array, x_coord_array = get_array(org_coords, src_coords, flag='x')
    
    c1_4 = calculate_perams(x_perams_array, x_coord_array)
    c5_8 = calculate_perams(y_perams_array, y_coord_array)
    print(c1_4, c5_8)
    print(new_img.shape)
    for i in range(w):
        for j in range(h):
            coord_array = np.array([i, j, i*j, 1])
            new_w = int(np.inner(coord_array, c1_4.T)) # 计算内积
            new_h = int(np.inner(coord_array, c5_8.T))
            try:
                new_img[new_h][new_w] = img[j][i]
            except Exception as e:
                print(i, j, new_h, new_w, e)
            
    return new_img

if __name__=='__main__':
    path = r'ImageData\insert\test.jpg'
    img = cv2.imread(path)
    # org_coords, src_coords 对应的坐标时（x,y）,但是到图片中是（y, x）
    org_coords = [[0, 0], [1279, 0], [1279, 719], [0, 719]]
    src_coords = [[100, 0], [1579, 0], [2179, 719], [0, 719]]
    new_img = transform(img, org_coords, src_coords)
    cv2.imwrite(r'ImageData\insert\coord_transform.jpg',new_img)
    
    
    
    
    