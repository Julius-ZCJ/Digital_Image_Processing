# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:17:41 2019

@author: Julius Zhu
"""

import cv2
import numpy as np
import grep_level_transform

def publish_process(img, k):
    """
    出版和印刷通常所作的操作：高提升滤波
    """
    h, w = img.shape[0:2]
    img_filter = cv2.GaussianBlur(img, (5, 5), 3)
    img = np.float16(img)
    img_filter = np.float16(img_filter)
    img_mask = img - img_filter
    img_result = img + k*img_mask
    for i in range(h):
        for j in range(w):
            if img_result[i][j]>=256:
                img_result[i][j]=255
            else:
                continue
    img_result = np.uint8(img_result)
    print(img_mask)

    cv2.imwrite('0.jpg', np.uint8(img))
    cv2.imwrite('1.jpg', np.uint8(img_filter))
    cv2.imwrite('2.jpg', np.uint8(img_mask))
    cv2.imwrite('3.jpg', img_result)
    #return img_result
    
def img_process(img):
    """
    
    """
    gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
    img_b = cv2.convertScaleAbs(gray_lap)
    img_c = img+img_b  # 拉普拉斯算子处理后的图加原图，书上c图
    
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    img_d = cv2.addWeighted(absX,0.5,absY,0.5,0) # sobel算子处理后的图,书上的e图
    
    img_e = cv2.blur(img_d, (5, 5))
    
    img_f = img_c*img_e
    
    img_g = img+img_f
    
    img_h = grep_level_transform.power_transform(img_g)
    
    cv2.imwrite('./a.jpg', img)
    cv2.imwrite('./b.jpg', img_b)
    cv2.imwrite('./c.jpg', img_c)
    cv2.imwrite('./d.jpg', img_d)
    cv2.imwrite('./e.jpg', img_e)
    cv2.imwrite('./f.jpg', img_f)
    cv2.imwrite('./g.jpg', img_g)
    cv2.imwrite('./h.jpg', img_h)
    
    

def main():
    path = r'..\ImageData\ch03\DIP3E_CH03_Original_Images\Fig0340(a)(dipxe_text).tif'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    publish_process(img, 1)

if __name__=='__main__':
    path = r'..\ImageData\ch03\bone.tif'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float64(img)
    img_b = cv2.Laplacian(img,-1,ksize = 3)
    #img_b = cv2.convertScaleAbs(gray_lap)
    img_c = img+img_b  # 拉普拉斯算子处理后的图加原图，书上c图
    
    img_blur = cv2.blur(img_c, (5, 5))
    x = cv2.Sobel(img_blur,-1,1,0)
    y = cv2.Sobel(img_blur,-1,0,1) # cv2.CV_16S
    # absX = cv2.convertScaleAbs(x)   # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    img_e = cv2.addWeighted(x,0.5,y,0.5,0) # sobel算子处理后的图,书上的e图
    
    img_f = img_c*img_e
    
    img_g = img+img_f
    
    #img_h = grep_level_transform.power_transform(img_g, 1, 0.5)
    cv2.imwrite('./a.jpg', img)
    cv2.imwrite('./b.jpg', img_b)
    cv2.imwrite('./c.jpg', img_c)
    cv2.imwrite('./d.jpg', img_d)
    cv2.imwrite('./e.jpg', img_e)
    cv2.imwrite('./f.jpg', img_f)
    cv2.imwrite('./g.jpg', img_g)
   # cv2.imwrite('./h.jpg', img_h)
        
    # img_process(img)
    # cv2.imwrite('./results/publish_process.jpg',result)