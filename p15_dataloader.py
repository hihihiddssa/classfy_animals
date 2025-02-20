'''
本节讲解内容如下：
1.数据加载器，DataLoader的用法，如：test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
2.数据加载器的参数,batch_size一批次大小,shuffle打乱数据,num_workers进程数,drop_last是否丢弃最后一个批次
'''
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#下载数据集且进行转换（compose）
test_data=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',train=False,download=True,transform=torchvision.transforms.ToTensor())

#dataloader
# 这里传入参数dataset=test_data，batch_size=4，shuffle=True，num_workers=0，drop_last=False
# 返回参数：
# 1.imgs：图像数据，形状为 (batch_size, channels, height, width)
# 2.targets：标签数据，形状为 (batch_size)
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
#drop_last=True表示当数据集大小不足一个batch_size时，将最后一个batch_size的数据丢弃
#shuffle=True表示打乱数据集
#num_workers=0表示使用0个进程来加载数据.windows系统不支持多进程加载数据，所以num_workers只能为0

img,target=test_data[0]
print(img)
print(target)

#使用tensorboard将数据集中的图像写入到tensorboard
writer=SummaryWriter('logs_p15')
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data#imgs是图像数据，targets是标签数据
        writer.add_images('Epoch:{}'.format(epoch),imgs,step)#imags就一次取一个batch_size的图像数据
        step=step+1
writer.close()



#使用dataloader进行批量处理
# for data in test_loader:
#     imgs,targets=data
#     print(imgs.shape)
#     print(targets)

