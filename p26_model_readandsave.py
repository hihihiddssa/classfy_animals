import torchvision
import torch
import torch.nn as nn

vgg16=torchvision.models.vgg16(pretrained=False)


#两种模型保存方法
#方法一：保存模型结构和参数
torch.save(vgg16,'vgg16_method1.pth')

#方法二：保存模型参数（官方推荐）
torch.save(vgg16.state_dict(),'vgg16_method2.pth')#相当于使用字典保存模型参数

#方式一二分别打印会发现，方法一保存的模型结构和参数，方法二保存的模型参数（字典）
model1=torch.load('vgg16_method1.pth')
print('model1:',model1)
model2=torch.load('vgg16_method2.pth')
print('model2:',model2)#各种卷积层、池化层、线性层等没有了，只有参数


#陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()    
        self.conv1=nn.Conv2d(3,64,kernel_size=3)#输入通道数为3，输出通道数为64，卷积核大小为3
    def forward(self,input):
        output=self.conv1(input)
        return output

#按照方式一保存那么就按方式一读取
#方式一的话不需要创建模型实例，直接保存
tudui=Tudui()
torch.save(tudui,'p26_tudui_method1.pth')
model=torch.load('p26_tudui_method1.pth')
print('model:',model)


#按照方式二保存那么就按方式二读取
#方式二的话需要创建模型实例，然后加载参数
tudui=Tudui()
torch.save(tudui.state_dict(),'p26_tudui_method2.pth')
model2=torch.load('p26_tudui_method2.pth')
print('model2:',model2)
