'''
本节讲解内容如下：
1.卷积运算
2.卷积运算的参数stride和padding
'''


import torch
import torch.nn.functional as F

#定义输入input和卷积核kernel
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

kernel=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])

print(input.shape)
print(kernel.shape)
#将input和kernel的形状改为(1,1,5,5)和(1,1,3,3)因为卷积输入需要4个输入
input=input.reshape(1,1,5,5)
kernel=kernel.reshape(1,1,3,3)
print(input.shape)
print(kernel.shape)


#卷积运算，stride部分
#stride=1 步长为1,意思是每次移动1个像素
output=F.conv2d(input,kernel,stride=1)
print(output)
#stride=2 步长为2,意思是每次移动2个像素
output2=F.conv2d(input,kernel,stride=2)
print(output2)

#卷积运算，padding部分
output3=F.conv2d(input,kernel,stride=1,padding=1)#padding=1 在图像的边缘填充1圈
print(output3)

