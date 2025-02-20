'''
两种方式将模型和数据转移到GPU上
第一种：

首先找到网络模型 tudui=tudui.to('cuda')
然后找到数据 imgs=imgs.to('cuda')
找到损失函数 loss_fn=loss_fn.to('cuda')
找到优化器 optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate).to('cuda')

这些都可以使用.to('cuda')方法
再返回就行




'''
import torch
import torch.nn as nn
#定义网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()    
        self.model=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),#输入通道数为3，输出通道数为32，卷积核大小为5，步长为1，填充为2
            nn.MaxPool2d(kernel_size=2),#池化层，池化核大小为2
            nn.Dropout(0.25),  # 添加 Dropout 层，丢弃率为 25%
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),#输入通道数为32，输出通道数为32，卷积核大小为5，步长为1，填充为2
            nn.MaxPool2d(kernel_size=2),#池化层，池化核大小为2
            nn.Dropout(0.25),  # 添加 Dropout 层
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),#输入通道数为32，输出通道数为64，卷积核大小为5，步长为1，填充为2
            nn.MaxPool2d(kernel_size=2),#池化层，池化核大小为2
            nn.Flatten(),#展平
            nn.Linear(1024,64),#线性层，输入大小为1024，输出大小为64
            nn.Dropout(0.5),  # 添加 Dropout 层，丢弃率为 50%
            nn.Linear(64,10)#线性层，输入大小为64，输出大小为10
        )
        
    def forward(self,x):
        x=self.model(x)
        return x

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#训练数据集
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪
    torchvision.transforms.ToTensor()
])

train_data=torchvision.datasets.CIFAR10(root='./CIFAR10datasets',train=True,download=True,transform=transform)
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
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate, weight_decay=1e-4)#parameters()方法返回模型中所有可训练参数，weight_decay=1e-4：权重衰减，防止过拟合

# 假设您有一个模型类 Tudui
tudui = Tudui()

# 判断 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#这就是第二种方式，第一种方式在上面
if torch.cuda.is_available():
    print('GPU可用')
    tudui = tudui.to(device)  # 将模型转移到 GPU 上#第二种方式
    loss_fn = loss_fn.to(device)  # 将损失函数转移到 GPU 上#第二种方式
else:
    print('GPU不可用')



#设置训练网络的参数
total_train_step=0#记录训练次数
total_test_step=0#记录测试次数
epoch=30#训练轮数


#添加tensorboard
writer=SummaryWriter('./logs_p27_train')


#训练步骤开始   
for i in range(epoch):
    print('第{}轮训练开始'.format(i+1))
    #训练步骤
    for data in train_loader:  
        imgs,targets=data#解包
        if torch.cuda.is_available():
            imgs=imgs.to('cuda')#将数据转移到GPU上#第一种方式
            targets=targets.to('cuda')#将数据转移到GPU上#第一种方式
        outputs=tudui(imgs)#前向传播
        loss=loss_fn(outputs,targets)#计算损失：查看推理结果和真实值的差距
    
        #优化器优化模型
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#更新参数
        #记录训练次数
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
    total_correct = 0#记录正确预测的样本数
    total_samples = 0#记录测试样本总数

    with torch.no_grad():#关闭梯度计算
        for data in test_loader:
            imgs, targets = data#解包
            if torch.cuda.is_available():
                imgs=imgs.to('cuda')#将数据转移到GPU上
                targets=targets.to('cuda')#将数据转移到GPU上
            outputs = tudui(imgs)#前向传播
            loss = loss_fn(outputs, targets)#计算损失
            total_test_loss += loss.item()

            #计算错误率
            preds=outputs.argmax(1)#返回最大值的索引
            total_correct += (preds == targets).sum().item()#累加正确预测的样本数
                #preds==targets：比较推理结果和真实值是否相等，返回一个布尔值列表
                #(preds == targets).sum()：布尔值列表求和，True=1,False=0
            total_samples += targets.size(0)#累加测试样本总数

    error_rate = 1 - (total_correct / total_samples)
    print('整体测试集上的loss：{}，错误率：{}'.format(total_test_loss, error_rate))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('error_rate', error_rate, total_test_step)
    total_test_step += 1

    #每轮训练后保存模型
    if i%5==0:
        torch.save(tudui,'tudui_{}.pth'.format(i))#保存模型
        print('模型第{}轮已保存'.format(i))


#关闭writer
writer.close()

