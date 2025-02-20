'''
这个程序定义了一个卷积神经网络模型，用于图像分类任务。
被引用：p27_fulltrainning.py
'''

import torch
import torch.nn as nn

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()    
        self.model=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),#输入通道数为3，输出通道数为32，卷积核大小为5，步长为1，填充为2
            nn.MaxPool2d(kernel_size=2),#池化层，池化核大小为2
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),#输入通道数为32，输出通道数为32，卷积核大小为5，步长为1，填充为2
            nn.MaxPool2d(kernel_size=2),#池化层，池化核大小为2
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),#输入通道数为32，输出通道数为64，卷积核大小为5，步长为1，填充为2
            nn.MaxPool2d(kernel_size=2),#池化层，池化核大小为2
            nn.Flatten(),#展平
            nn.Linear(1024,64),#线性层，输入大小为1024，输出大小为64
            nn.Linear(64,10)#线性层，输入大小为64，输出大小为10
        )
        
    def forward(self,x):
        x=self.model(x)
        return x

#测试模型的输出正确与否
if __name__=='__main__':
    tudui=Tudui()
    input=torch.ones((64,3,32,32))#64batchsize，3通道，32*32像素
    output=tudui(input)
    print(output)

