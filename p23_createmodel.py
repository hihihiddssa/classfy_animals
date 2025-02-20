import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model=Sequential(
            Conv2d(3,32,5,padding=2),#参数含义-->3：输入通道数，32：输出通道数，5：卷积核大小，padding=(kernel_size-1)/2：填充
            MaxPool2d(2),#使用2*2的池化层
            Conv2d(32,32,5,padding=2),#32：输入通道数，32：输出通道数，5：卷积核大小，padding=(kernel_size-1)/2：填充
            MaxPool2d(2),#使用2*2的池化层
            Conv2d(32,64,5,padding=2),#32：输入通道数，64：输出通道数，5：卷积核大小，padding=(kernel_size-1)/2：填充
            MaxPool2d(2),#使用2*2的池化层
            Flatten(),#展平，展平后特征数为64*4*4=1024
            Linear(1024,64),#1024：输入特征数，64：输出特征数
            Linear(64,10)#64：输入特征数，10：输出特征数
        )

    def forward(self,x):
        x=self.model(x)
        return x

tudui=Tudui()
print(tudui)
input=torch.ones((64,3,32,32))#64：batch_size，3：通道数，32：高度，32：宽度
output=tudui(input)
print(output.shape)

#另一种可视化方式
writer=SummaryWriter("logs_p23")
writer.add_graph(tudui,input)#add_graph()函数用于记录模型结构
writer.close()

