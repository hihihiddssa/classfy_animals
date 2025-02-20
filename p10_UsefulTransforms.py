#前言
#transform模块提供了一些常用的图像变换方法，可以通过transforms.Compose()将多个transforms组合在一起

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
#创建一个SummaryWriter对象，指定日志文件写入的目录
writer =SummaryWriter('logs')
#读取图片
img=Image.open(r'dataset/train/ants_image/24335309_c5ea483bb8.jpg')
print(img)


#1.ToTensor（）输入是PIL图像或numpy.ndarray，输出是tensor类型
trans_totensor =transforms.ToTensor()#使用transforms.ToTensor()将PIL图像或numpy.ndarray转换为tensor类型
img_tensor=trans_totensor(img)
#添加到tensorboard
writer.add_image('Tensor_img',img_tensor,1)

#2.Normalize()标准化
#mean和std参数分别指定图像的均值和标准差 针对每个通道分别设置的
#标准化的公式是：output = (input - mean) / std
print(img_tensor[0][0][0])#0层的第0行第0列的像素值
trans_nomarlize=transforms.Normalize(mean=[0.8,0.8,0.8],std=[0.5,0.5,0.5])#使用transforms.Normalize()对图像进行标准化
img_norm=trans_nomarlize(img_tensor)
print(img_norm[0][0][0])
#添加到tensorboard
writer.add_image('Normalize_img',img_norm,5)

#3.Resize()调整图像大小:输入时PIL图像，输出也是PIL图像
trans_resize=transforms.Resize((512,512))#使用transforms.Resize()调整图像大小为512*512
#img (PIL)->resize->img_resize (PIL)
img_resize=trans_resize(img)
#img_resize (PIL)->ToTensor->img_resize (tensor)
img_resize=trans_totensor(img_resize)
#添加到tensorboard
writer.add_image('Resize_img',img_resize,0)

#4.composes()组合多个transforms
trans_resize_compose=transforms.Resize((512,256))
#Compose()将多个transforms组合在一起
trans_composes=transforms.Compose([trans_resize_compose,trans_totensor])
img_compose=trans_composes(img)
#添加到tensorboard
writer.add_image('Compose_img',img_compose,1)

#5.RandomCrop()随机裁剪
trans_random_crop = transforms.RandomCrop((256,100))#使用transforms.RandomCrop()随机裁剪图像
trans_composes2=transforms.Compose([trans_random_crop,trans_totensor])
for i in range(5):
    img_crop =trans_composes2(img)
    writer.add_image('RandomCrop_img',img_crop,i)

#tips
#不知道数据类型的时候，可以使用print(type())函数查看数据类型

#关闭SummaryWriter对象
writer.close()