'''
本节讲解内容如下：
1.神经网络的骨架，nn.Module
2.神经网络的结构，__init__和forward方法
'''
#container:指的是神经网络的骨架
  #nn.Module:神经网络的结构
from torch import nn
import torch

#定义一个简单的神经网络模型
class MyModel(nn.Module):
    def __init__(self):#初始化方法
        super().__init__()#调用父类的初始化方法
   
    def forward(self,x):#forward方法定义了前向传播的过程
        x=x+1
        return x
    '''
    私有方法：__init__和__forward__。使用方法举例：
    model=MyModel()#实例化
    model.forward(x)#调用forward方法
    model(x)#调用forward方法
  
    '''
model=MyModel()#创建模型
x=torch.tensor(1.0)#创建输入
y=model(x)#前向传播
print(y)











        

