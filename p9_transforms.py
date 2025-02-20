#前言
#主题：transforms模块
#像一个工具箱，里面有很多工具，我们可以通过这些工具来对数据进行处理比如对图像进行旋转，缩放，裁剪，对图像进行归一化，灰度化，标准化

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

#绝对路径 D:\AMY_PROJECTS\torch_learning\dataset\train\ants_image\6743948_2b8c096dda.jpg
#相对路径 dataset\train\ants_image\6743948_2b8c096dda.jpg
img_path=r'dataset\train\ants_image\6743948_2b8c096dda.jpg'
img=Image.open(img_path)



#1.toTensor()将PIL图像或numpy.ndarray转换为tensor类型
#将图片转换为tensor类型图片
tool=transforms.ToTensor()
tensor_img=tool(img)
#print(tensor_img)

writer=SummaryWriter('logs')
writer.add_image('Tensor_img',tensor_img,2)
writer.close()