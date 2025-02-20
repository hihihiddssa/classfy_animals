'''
本节内容：
1. 线性层
    1.1 展平
    1.2 线性层

'''
import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

datasets=torchvision.datasets.CIFAR10('./CIFAR10datasets',train=True,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(datasets,batch_size=64,shuffle=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.linear1=nn.Linear(196608,10)#线性层输入196608，输出10

    def forward(self,input):
        output=self.linear1(input)#经过线性层后，输出结果
        return output

tudui=Tudui()
for data in dataloader:
    imgs,targets=data
    print(imgs.shape)#结果：torch.Size([64, 3, 32, 32])batchsize=64,通道数=3，高度=32，宽度=32
    
    # 展平
    output=torch.flatten(imgs)#展成一行
    # output=torch.reshape(imgs,(1,1,1,-1))#各参数含义：1：通道数，1：高度，1：宽度，-1：自动计算shape
    # print(output.shape)#结果：torch.Size([1, 1, 1, 196608])batchsize=1,通道数=1，高度=1，宽度=196608
    print(output.shape)#结果：torch.Size([196608])
    
    # 展平后，输入tudui模型，经过线性层后，输出结果
    output=tudui(output)#经过tudui模型后，输出结果
    print(output.shape)


