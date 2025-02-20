'''
本节讲解内容如下：
1.卷积运算 
2.卷积运算的参数stride和padding 
3.卷积运算的输入和输出形状
4.卷积运算的输入和输出写入tensorboard
'''

#conv2d指的是二维卷积运算/conv3d指的是三维卷积运算
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#加载数据集并且使用DataLoader进行数据加载
dataset = torchvision.datasets.CIFAR10(root='./CIFAR10datasets', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
'''
batch_size：每次读取的样本数量
shuffle：是否打乱数据集
num_workers：读取数据时使用的线程数
'''
#print(dataset[0])


#定义网络模型
class MyNet(nn.Module):
    #定义初始化函数
    def __init__(self):
        super(MyNet, self).__init__()#继承父类
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)#定义卷积层
        
        '''
        in_channels：输入通道数
        out_channels：输出通道数
        kernel_size：卷积核大小
        stride：步长
        padding：填充
        '''
    #定义前向传播
    def forward(self, x):
        x = self.conv1(x)#输入x经过卷积层conv1后输出x
        return x
    
myNet = MyNet()#实例化模型

writer = SummaryWriter('logs_p18')
step = 0
for data in dataloader:
    imgs, targets = data#data来自于dataloader，包含数据和标签
    output = myNet.forward(imgs)#将数据输入网络模型中
    #打印输入和输出的形状
    # print('imgs.shape:',imgs.shape)
    # print('output.shape:',output.shape)
    #imgs.shape: torch.Size([64, 3, 32, 32])
    #output.shape: torch.Size([64, 6, 30, 30])
    '''
    由于torchvision.datasets.CIFAR10数据集中的图像是3通道的，而卷积层输出的通道数为6，
    所以需要将输出形状转换为[64, 3, 30, 30]，这样才能与输入形状相同，便于写入tensorboard
    
    '''
    output = torch.reshape(output, (-1, 3, 30, 30))#将输出形状转换为[64, 3, 30, 30]
    '''
    -1表示自动计算batch_size
    3表示通道数
    30和30表示图像的高度和宽度
    '''
    #将输入和输出写入tensorboard
    writer.add_images('input',imgs, step)
    writer.add_images('output',output, step)
    step += 1




