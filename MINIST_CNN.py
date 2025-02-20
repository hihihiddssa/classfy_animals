import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保结果可重现
torch.manual_seed(1)

# 定义CNN模型
class CNN(nn.Module):  
    def __init__(self):
        super(CNN, self).__init__()#继承父类
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            # 输入通道数=1（灰度图像），输出通道数=16，卷积核大小=5x5，步长=2，填充=2
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            # ReLU激活函数
            nn.ReLU(),
            # 最大池化层，核大小=2x2
            nn.MaxPool2d(2)
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            # 输入通道数=16，输出通道数=32，卷积核大小=5x5，步长=1，填充=2
            nn.Conv2d(16, 32, 5, 1, 2),
            # ReLU激活函数
            nn.ReLU()
        )
        # 全连接输出层
        # 32 * 7 * 7 是根据前面的卷积和池化操作计算得出的
        # 10是输出类别的数量（数字0-9）
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # 前向传播函数，定义数据如何通过网络
        
        x = self.conv1(x)  # 通过第一个卷积层（包含卷积、ReLU激活和最大池化）
        x = self.conv2(x)  # 通过第二个卷积层（包含卷积和ReLU激活）
        
        # 重塑张量，将多维特征图展平为二维张量
        x = x.view(x.size(0), -1)
        # 解释：
        # x.size(0) 是批量大小（batch size）
        # -1 让PyTorch自动计算这个维度，以保持元素总数不变
        # 例如，如果 x 之前的形状是 (64, 32, 7, 7)，那么这个操作后
        # x 的新形状将是 (64, 1568)，其中 1568 = 32 * 7 * 7
        
        # 这个操作的目的是：
        # 1. 保持批量大小不变
        # 2. 将每个样本的多维特征"展平"成一个向量
        # 3. 为全连接层准备合适的输入格式
        
        output = self.out(x)  # 通过全连接层，产生最终输出
        # 全连接层需要二维输入，这就是为什么我们需要先展平数据
        
        return output  # 返回模型的输出

# 设置超参数
batch_size = 64  # 每批处理的样本数
learning_rate = 0.01  # 学习率
epochs = 10  # 训练轮数
# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化：(像素值 - 均值) / 标准差
])

# 加载MNIST训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 加载MNIST测试集


test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型、损失函数和优化器
model = CNN().to(device)  # 将模型移动到指定设备（GPU或CPU）
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 创建 TensorBoard SummaryWriter 实例
writer = SummaryWriter('runs/mnist_experiment')

# 修改后的训练函数
def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将数据移动到指定设备
        optimizer.zero_grad()  # 清零梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        # 记录训练损失
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        
        if batch_idx % 100 == 0:  # 每100批次打印一次训练状态
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 修改后的测试函数
def test(model, device, test_loader, writer, epoch):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # 前向传播
            test_loss += criterion(output, target).item()  # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测的数量

    test_loss /= len(test_loader.dataset)  # 计算平均损失
    accuracy = 100. * correct / len(test_loader.dataset)  # 计算准确率
    
    # 记录测试损失和准确率
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 训练和测试模型
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, writer)  # 训练一个epoch
    test(model, device, test_loader, writer, epoch)  # 在测试集上评估模型

# 关闭 SummaryWriter
writer.close()

print("训练完成！")
