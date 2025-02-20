import torch
import torch.nn as nn


input=torch.tensor([[1,-0.5],
                    [-1,3]])

#output=torch.reshape(input,(-1,1,2,2))#自己计算batch_size，1通道，2行2列
#print(output.shape)

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.relu1=nn.ReLU(inplace=False)
        #inplace=False表示不直接在原input上进行操作，返回一个新的tensor/inplace=True表示直接在原input上进行操作，节省内存/
        #ReLU函数的作用是：如果输入的值小于0，则输出0；如果输入的值大于0，则输出输入的值。
    def forward(self,input):
        output=self.relu1(input)
        return output

mynet=Mynet()
output=mynet(input)
print(output)

