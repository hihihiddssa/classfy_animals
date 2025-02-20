import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim


'''
损失函数意思：衡量预测值与真实值之间的差距，差距越小，模型越好
优化器意思：根据损失函数计算的梯度更新模型参数，帮助模型找到最优参数
'''

# 加载数据集
dataset = torchvision.datasets.CIFAR10(root='./CIFAR10datasets', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model=Sequential(
            Conv2d(3,32,5,padding=2),#参数含义-->3：输入通道数，32：输出通道数，5：卷积核大小，padding=(kernel_size-1)/2：填充
            MaxPool2d(2),#使用2*2的池化层
            Conv2d(32,32,5,padding=2),#32：输入通道数，32：输出通道数，5：卷积核大小，padding=(kernel_size-1)/2：填充
            MaxPool2d(2),#使用2*2的池化层
            Conv2d(32,64,5,padding=2),#32：输入通道数，64：输出通道数，5：卷积核大小，padding=(kernel_size-1)/2：填充
            MaxPool2d(2),#使用2*2的池化层
            Flatten(),#展平，展平后特征数为64*4*4=1024
            Linear(1024,64),#1024：输入特征数，64：输出特征数
            Linear(64,10)#64：输入特征数，10：输出特征数
        )

    def forward(self,x):
        x=self.model(x)
        return x
    
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 分类问题，使用交叉熵损失函数
loss_cross=nn.CrossEntropyLoss()

# 构造模型并移到 GPU
tudui=Tudui().to(device)

# 构造优化器：根据孙叔函数计算的梯度更新模型参数，帮助模型找到最优参数
optimizer=torch.optim.SGD(tudui.parameters(),lr=0.01)#各参数含义：tudui.parameters()：模型参数，lr：学习率 后面的参数是可选参数，可以设置动量momentum等，可以参考别人的

for epoch in range(20):#训练20个周期
    running_loss=0.0#初始化损失
    for data in dataloader:
        imgs,targets=data#imgs是输入数据，targets是目标数据
        imgs=imgs.to(device)#将输入数据移到 GPU
        targets=targets.to(device)#将目标数据移到 GPU
        
        outputs=tudui(imgs)#将输入数据传入模型中，得到输出
        #print('outputs:',outputs)#Eg:outputs: tensor([[-0.0050,  0.0836, -0.0969,  0.1032,  0.0086,  0.0521, -0.0888,  0.0996,0.0271,  0.1046]], grad_fn=<AddmmBackward0>)
        #print('targets:',targets)#Eg:targets: tensor([3])

        outputs=torch.reshape(outputs,(1,10))#将输出展平为batch_size=1，类别数为10
        result_cross=loss_cross(outputs,targets)#计算交叉熵损失
        #print('result_cross:',result_cross)

        #套路：梯度清零，否则梯度会累加
        optimizer.zero_grad()
        #套路：反向传播   
        result_cross.backward()#自动计算梯度grad，之后使用优化器更新模型参数会用到
        #套路：优化器更新模型参数
        optimizer.step()#更新模型参数
        
        running_loss+=result_cross.item()#累加损失
    print(f'Epoch {epoch+1}, running_loss: {running_loss}')














