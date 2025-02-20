import torch
import torch.nn as nn
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',train=False,download=True,transform=torchvision.transforms.ToTensor())
data_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=64,shuffle=True)#shuffle=True表示打乱数据

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.relu1=nn.ReLU(inplace=False) #inplace=False表示不直接在原input上进行操作，返回一个新的tensor/inplace=True表示直接在原input上进行操作，节省内存/
        #ReLU函数的作用是：如果输入的值小于0，则输出0；如果输入的值大于0，则输出输入的值。
        self.sigmod1=nn.Sigmoid()
        #Sigmoid函数的作用是：将输入的值映射到0到1之间。
    
    def forward(self,input):
        output=self.sigmod1(input)
        return output

summerywriter=SummaryWriter('logs_p20_sigmod')
step=0

mynet=Mynet()
for data in data_loader:
    imgs,targets=data#解包
    summerywriter.add_images('input',imgs,step)
    outputs=mynet(imgs)
    step+=1
    summerywriter.add_images('output',outputs,step)

summerywriter.close()


