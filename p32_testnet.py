from PIL import Image
import torch
import torchvision
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
            nn.Dropout(p=0.5),#Dropout层，丢弃概率为0.5
            nn.Linear(64,10)#线性层，输入大小为64，输出大小为10
        )
        
    def forward(self,x):
        x=self.model(x)
        return x


img_path='./imgs/dog.png'
img=Image.open(img_path)

transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])

tudui=Tudui()
tudui=torch.load('tudui_10.pth')


img=transform(img)
img=torch.reshape(img,(1,3,32,32))
output=tudui(img)
print(output)
output=torch.argmax(output,dim=1)
print(output)
