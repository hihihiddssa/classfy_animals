import torch
import torch.nn as nn

#loss函数：衡量模型预测值与实际值之间的差距
inputs = torch.tensor([1.0,2.0,3.0],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)
#由于L1Loss和MSELoss要求输入格式为[batch_size,类别数]，所以需要reshape
inputs = torch.reshape(inputs,(1,3))
targets = torch.reshape(targets,(1,3))

#L1Loss是均方差损失函数
loss=nn.L1Loss()
result_l1 = loss(inputs,targets)
print(result_l1)
#MSELoss是均方差损失函数
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs,targets)
print(result_mse)


#求交叉熵损失cross_entropy_loss
x = torch.tensor([0.1,0.2,0.3])#预测值
y = torch.tensor([1])#命中类别 1指的就是0，1，2中的1，也就是第二个类别
x = torch.reshape(x,(1,3))#由于交叉熵损失函数输入要求格式为[batch_size,类别数]，所以需要reshape
loss_fn = nn.CrossEntropyLoss()#交叉熵损失函数
cross_entropy_loss = loss_fn(x,y)
print(cross_entropy_loss)

