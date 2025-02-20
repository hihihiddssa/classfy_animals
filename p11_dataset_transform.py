#前言
#使用自己的数据集批量处理数据集

import PIL.ImageShow
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


#定义数据集的转换compose
dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
#下载数据集且进行转换（compose）
train_set=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',transform=dataset_transform,download=True)

print(train_set[0])
# img,target=train_set[0]
# print(img)
# print(target)
# print(train_set.classes[target])
# PIL.ImageShow.show(img)

#将数据集中的图像写入到tensorboard
writer=SummaryWriter('logs_p11')
for i in range(10):
    img,target=train_set[i]
    writer.add_image('train_set',img,i)
writer.close()
