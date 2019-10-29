

Unit_one.py 三种插值方式：
1. 最近邻插值： 
   
    new_shape = (new_x, new_y)
   根据变化后的坐标i，j来通过比例对应到原图像（origin_x, origin_y）的像素值：
   map_x = i * origin_x/new_x
   map_y = j * origin_y/new_y
   
   new_img = img(map_x,map_y)
   
2. 双线性插值：
    在x ， y轴上都使用一次插值，如图：

​     ![](F:\project\Digital_Image_Processing\introduce\double_insert.png)

假如我们想得到未知函数 f 在点 P = (x, y) 的值，假设我们已知函数 f 在 Q11 = (x1, y1)、Q12 = (x1, y2), Q21 = (x2, y1) 以及 Q22 = (x2, y2) 四个点的值。最常见的情况，f就是一个像素点的像素值。首先在 x 方向进行线性插值，得到


![](F:\project\Digital_Image_Processing\introduce\fountion1.png)

​                   		  ![](F:\project\Digital_Image_Processing\introduce\fountion2.png)

 然后在 y 方向进行线性插值，得到 

​				![](F:\project\Digital_Image_Processing\introduce\fountion3.png)

 综合起来就是双线性插值最后的结果： 

​				![](F:\project\Digital_Image_Processing\introduce\fountion4.png)

​				![](F:\project\Digital_Image_Processing\introduce\fountion5.png)

[参考这里](https://blog.csdn.net/xbinworld/article/details/65660665)



3.双三次内插：

​	参考[这里](https://blog.csdn.net/qq_29058565/article/details/52769497)

双三次内插其实就是取最近的16个像素点，在通过 BiCubic基函数 得到x，y方向的权值和对应像素点相乘在相加的结果。



![](F:\project\Digital_Image_Processing\introduce\info.png)

图像来源于[这篇](https://blog.csdn.net/qq_39683287/article/details/80288872)博客



学习中思考：

1. 我在实现插值算法之后，用opencv中的算法结果进行对比，发现在相同的算法下，不论是速度还是效果都和opencv有很大的差距。思考良久始终不知何解（想要了解opencv是怎么优化和实现算法的）。

2. 在双线性插值算法中最好使用 几何中心的对齐 ，就是在计算对应点时用： 

   **SrcX=(dstX+0.5)\* (srcWidth/dstWidth) -0.5** 来进行计算。

3. 算法优化，在计算时将小数通过放大取整，最后在缩小相应的倍数，通常是2的倍数，这样在缩小时就可以通过移位的操作实现。我用python实现，发现效果并不是特别明显。

4. 思考：既然可以通过4个点和16个点的操作进行插值，是不是也可以通过最近的9个点来计算呢，这样相比于双三次内插来说可以减少计算量。

   我的想法是，通过正太分布确定各个像素点的权重，在通过相加来求出这点的像素值。（有时间再来实现看看效果）



wolbergTransform.py

​	参考数字图像处理51页。

​	这里是5种放射变换，在其中还存在一些问题待解决：

		1. 将坐标系平移到图形中心的操作，正确的操作应该是：平移--->变换---->平移，由于时间有限待以后学习中在加入。
  		2. 在进行放射变换时，会出现某些像素点没有赋值的情况，可以通过反向映射解决。
  		3. 边缘锯齿的问题，这个或许可以通过平滑滤波来解决（待处理）。



coord_transform.py

​	数字图像处理52页

​	图像配准，实际就是这个公式：

​		x = c_1 * v+c_2 * w+ c_3 * v * w +c_4

​		y = c_5 * v +c_6 * w+ c_7 * v * w + c_8

图像配准必须至少要知道4个对应的点来求8个参数，当然知道的对应点越多匹配越准。

这里面也会出现有某些像素点没有赋值的情况，通过滤波或许可以解决。

