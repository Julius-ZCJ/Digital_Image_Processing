# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:54:01 2019

@author: Julius Zhu
"""

import cv2
import numpy as np

    
    
    
class SpaceFilter():

    
    # 自定义kernal滤波
    mean_kernal = np.ones((3,3))/9
    
    @staticmethod
    def self_filter(img, kernal):
        """
        cv2.filter2D:
        函数原型： filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst
        src参数表示待处理的输入图像。
        ddepth参数表示目标图像深度，输入值为-1时，目标图像和原图像深度保持一致
        
        """
        
        img = cv2.filter2D(img, -1, kernal)
        return img
    
    @staticmethod
    def mean_blur(image):      #均值模糊  去随机噪声有很好的去燥效果
        """
        线性平滑滤波
        低通滤波（均值模糊）函数原型：blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst
        src参数表示待处理的输入图像。
        ksize参数表示模糊内核大小。比如(1,15)表示生成的模糊内核是一个1*15的矩阵。
        dst参数表示输出与src相同大小和类型的图像。
        anchor参数、borderType参数可忽略
        """
        dst = cv2.blur(image, (1, 9))    #（1, 9）是垂直方向模糊，（9， 1）还水平方向模糊
        cv2.namedWindow('blur_demo', cv2.WINDOW_NORMAL)
        cv2.imshow("blur_demo", dst)
        
    def gaussian_blur(image):        #去除高斯噪声
        """
        线性平滑滤波
        高斯模糊GaussianBlur函数原型：GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
        src参数表示待处理的输入图像。
        ksize参数表示高斯滤波器模板大小。 
        ksize.width和ksize.height可以不同，但它们都必须是正数和奇数。或者，它们可以是零，即（0, 0），然后从σ计算出来。
        sigmaX参数表示 X方向上的高斯内核标准差。
        sigmaY参数表示 Y方向上的高斯内核标准差。 如果sigmaY为零，则设置为等于sigmaX，
        如果两个sigma均为零，则分别从ksize.width和ksize.height计算得到。
        补：若ksize不为(0, 0)，则按照ksize计算，后面的sigmaX没有意义。若ksize为(0, 0)，则根据后面的sigmaX计算ksize
        """

        dst = cv2.GaussianBlur(image, (15, 15), 0)  # 高斯模糊
        cv2.namedWindow("Gaussian", cv2.WINDOW_NORMAL)
        cv2.imshow("Gaussian", dst)
       
    @staticmethod
    def median_blur(image):    # 中值模糊  对椒盐噪声有很好的去燥效果
        """
        非线性平滑滤波
        中值滤波（中值模糊）函数原型：medianBlur(src, ksize[, dst]) -> dst
        src参数表示待处理的输入图像。
        ksize参数表示滤波窗口尺寸，必须是奇数并且大于1。比如这里是5，中值滤波器就会使用5×5的范围来计算，
        即对像素的中心值及其5×5邻域组成了一个数值集，对其进行处理计算，当前像素被其中值替换掉。
        dst参数表示输出与src相同大小和类型的图像。
        """
        dst = cv2.medianBlur(image, 5)
        cv2.namedWindow('median_blur_demo', cv2.WINDOW_NORMAL)
        cv2.imshow("median_blur_demo", dst)
       
        
    #进行边缘保留滤波通常用到两个方法：高斯双边滤波和均值迁移滤波。
    @staticmethod
    def bi_demo(image):   #双边滤波
        """
        非线性滤波
        双边滤波函数原型：bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst
        src参数表示待处理的输入图像。
        d参数表示在过滤期间使用的每个像素邻域的直径。如果输入d非0，则sigmaSpace由d计算得出，
        如果sigmaColor没输入，则sigmaColor由sigmaSpace计算得出。
        sigmaColor参数表示色彩空间的标准方差，一般尽可能大。较大的参数值意味着像素邻域内较远的颜色会混合在一起，
        从而产生更大面积的半相等颜色。
        sigmaSpace参数表示坐标空间的标准方差(像素单位)，一般尽可能小。参数值越大意味着只要它们的颜色足够接近，
        越远的像素都会相互影响。当d > 0时，它指定邻域大小而不考虑sigmaSpace。 否则，d与sigmaSpace成正比
        """
        dst = cv2.bilateralFilter(image, 0, 100, 15)
        cv2.namedWindow("bi_demo", cv2.WINDOW_NORMAL)
        cv2.imshow("bi_demo", dst)
        
    @staticmethod
    def shift_demo(image):   #均值迁移
        """
        均值漂移pyrMeanShiftFiltering函数:pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) -> dst
        src参数表示输入图像，8位，三通道图像。
        sp参数表示漂移物理空间半径大小。
        sr参数表示漂移色彩空间半径大小。
        dst参数表示和源图象相同大小、相同格式的输出图象。
        maxLevel参数表示金字塔的最大层数。
        termcrit参数表示漂移迭代终止条件。
        """
        dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
        cv2.namedWindow("shift_demo", cv2.WINDOW_NORMAL)
        cv2.imshow("shift_demo", dst)
        
    @staticmethod
    def sobel_filter(image):
        sobel_kernal = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
        SpaceFilter.self_filter(image, sobel_kernal)
        
    @staticmethod
    def sobel(img):
        """
        dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
        函数返回其处理结果。
        前四个是必须的参数：
        第一个参数是需要处理的图像；
        第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
        dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
        可选：
        ksize是Sobel算子的大小，必须为1、3、5、7。
        scale是缩放导数的比例常数，默认情况下没有伸缩系数；
        delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
        borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
        
        dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
        其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值

        """
        x = cv2.Sobel(img,cv2.CV_16S,1,0)
        y = cv2.Sobel(img,cv2.CV_16S,0,1)
        absX = cv2.convertScaleAbs(x)   # 转回uint8
        absY = cv2.convertScaleAbs(y)
         
        dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    
        return dst
    
    def laplacian(img):
        """
        dst = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])
        前两个是必须的参数：
    
        第一个参数是需要处理的图像；
        第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；

        其后是可选的参数：
        dst不用解释了；
        ksize是算子的大小，必须为1、3、5、7。默认为1。
        scale是缩放导数的比例常数，默认情况下没有伸缩系数；
        delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
        borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。

        """
        gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3) # cv2.CV_16S 16位有符号的数据类型
        dst = cv2.convertScaleAbs(gray_lap) # 16位有符号的数据类型转为8位
         
        cv2.imshow('laplacian',dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        
        
        
        
        
        
        
        
        
        
        
        
        
        