'''
本节讲解内容：
1.池化层 参数含义：kernel_size：池化窗口大小 ceil_mode：是否向上取整
'''


import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#########################输入数据####################################
dataset = torchvision.datasets.CIFAR10(root='./CIFAR10datasets',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=4)
#########################定义网络模型#############################
class MyNet(nn.Module):
    def __init__(self):
        #继承父类
        super(MyNet,self).__init__()
        #定义池化层
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=True)#kernel_size表示池化窗口大小 ceil_mode表示是否保留边缘/向上取整
    def forward(self,input):
        #前向传播池化
        output = self.maxpool1(input)
        #返回输出
        return output
##########################实例化#############################
myNet = MyNet()
summarywriter = SummaryWriter('logs_p19')
step = 0
#输入数据
for data in dataloader:
    imgs,targets = data#解包
    output = myNet(imgs)
    summarywriter.add_images('input',imgs,step)
    summarywriter.add_images('output',output,step)
    step += 1

summarywriter.close()

