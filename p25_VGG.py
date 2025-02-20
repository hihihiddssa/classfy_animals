import torchvision
from torch import nn
# train_data=torchvision.datasets.ImageNet(root='./ImageNet',split='train',download=True
#                                          ,transform=torchvision.transforms.ToTensor())
# vgg16_false=torchvision.models.vgg16(pretrained=False)
# print(vgg16_false)
vgg16_true=torchvision.models.vgg16(pretrained=True)#最新调用方法vgg16_true=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

train_data=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',train=True,download=True,transform=torchvision.transforms.ToTensor())

'''
由于CIFAR10数据集10类
VGG16输出层为1000类
所以需要修改输出层
'''
#方法一：直接在最后加线性层
# vgg16_true.add_module('add_linear',nn.Linear(1000,10))#在vgg16_true模型中添加名为add_linear的线性层，输入特征数为1000，输出特征数为10
#方法二：在vgg里面添加其他层，可以参考以下代码
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
print('vgg16_true:',vgg16_true)


#使用vgg16_false演示修改
vgg16_false=torchvision.models.vgg16(pretrained=False)
#修改原有模型的线性层：[4096,1000]-->[4096,10]
vgg16_false.classifier[6]=nn.Linear(4096,10)
print('vgg16_false:',vgg16_false)
