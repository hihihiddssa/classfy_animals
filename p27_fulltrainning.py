import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from p27_model import Tudui
from torch.utils.tensorboard import SummaryWriter

#训练数据集
train_data=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',train=True,download=True,transform=torchvision.transforms.ToTensor())
#测试数据集
test_data=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',train=False,download=True,transform=torchvision.transforms.ToTensor())

#打印数据集长度测试
# train_data_size=len(train_data)
# test_data_size=len(test_data)
# print('训练数据集的长度为：{}'.format(train_data_size))#.format()方法用于格式化字符串,替换{}中的内容
# print('测试数据集的长度为：{}'.format(test_data_size))

#dataLoader加载数据集
train_loader=DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True)



#创建网络模型
tudui=Tudui()

#损失函数
loss_fn=nn.CrossEntropyLoss()#交叉熵损失函数,适合分类问题

#优化器(随机梯度下降)
learning_rate=0.01#学习率 或者写成1e-2
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate)#parameters()方法返回模型中所有可训练参数

#设置训练网络的参数
total_train_step=0#记录训练次数
total_test_step=0#记录测试次数
epoch=10#训练轮数

#添加tensorboard
writer=SummaryWriter('./logs_p27_train')


#训练步骤开始
for i in range(epoch):
    print('第{}轮训练开始'.format(i+1))

    #训练步骤
    for data in train_loader:
        imgs,targets=data#解包
        outputs=tudui(imgs)#前向传播
        loss=loss_fn(outputs,targets)#计算损失：查看推理结果和真实值的差距
        #优化器优化模型
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#更新参数
        
        total_train_step+=1
        if total_train_step%100==0:
            print('训练次数：{}，loss：{}'.format(total_train_step,loss.item()))#item()方法将tensor类型转换为一个标量
            writer.add_scalar('train_loss',loss.item(),total_train_step)#将损失写入tensorboard
            #add_scalar()方法的参数：
            #1.tag:标签，用于区分不同的数据
            #2.scalar_value:要记录的标量值
            #3.global_step:当前训练的步数
    #测试步骤开始
    total_test_loss=0#记录整体测试集上的损失
    with torch.no_grad():#关闭梯度计算
        for data in test_loader:
            imgs,targets=data#解包
            outputs=tudui(imgs)#前向传播
            loss=loss_fn(outputs,targets)#当前损失不是整体损失，而是单个样本的损失
            total_test_loss+=loss#累加损失
    print('整体测试集上的loss：{}'.format(total_test_loss))
    writer.add_scalar('test_loss',total_test_loss,total_test_step)#将测试损失写入tensorboard
    total_test_step+=1

    #每轮训练后保存模型
    torch.save(tudui,'tudui_{}.pth'.format(i))#保存模型
    print('模型第{}轮已保存'.format(i))


#关闭writer
writer.close()


'''
分类问题一般使用正确率
假设：
2*input
|
model(二分类)
|
output=
[0.1,0.2]
[0.3,0.4]
0     1
|
argmax(output)#返回最大值的索引
|
preds=[1][1]
inputstargets=[0][1]
|
preds==targets #比较推理结果和真实值是否相等，返回一个布尔值列表
|
[False,True].sum()=0+1=1 #布尔值列表求和，True=1,False=0
|
correct_ratio=1.0/2=0.5 #正确率
'''