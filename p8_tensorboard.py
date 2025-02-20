#前言
#主题：tensorboard 可视化工具
#tensorboard是一个由TensorFlow提供的可视化工具，用于展示训练过程中的损失、准确率、梯度等信息，以及模型结构、图像、音频、文本等数据。PyTorch也提供了与tensorboard兼容的接口，可以使用tensorboard来监控PyTorch模型的训练过程。
#终端运行tensorboard --logdir=logs打开tensorboard
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter类，用于将数据写入TensorBoard
from PIL import Image  # 导入PIL库中的Image类，用于处理图像
import numpy as np  # 导入NumPy库，用于处理数组

#如何使用tensorboard：

#1. 创建一个SummaryWriter对象，指定日志文件写入的目录
# 创建一个SummaryWriter对象，指定日志文件写入的目录为'logs'
writer = SummaryWriter('logs')
#2. 使用add_scalar()方法将标量数据写入日志文件
# 指定图像文件的路径
image_path = 'dataset/train/ants_image/69639610_95e0de17aa.jpg'
# 打开图像文件
img_PIL = Image.open(image_path)
# 将图像转换为NumPy数组
img_array = np.array(img_PIL)
# 打印图像数组的形状
print(img_array.shape)
# 将图像写入到日志文件中
# 第一个参数为标题'test'
# 第二个参数为图像数组img_array
# 第三个参数为索引3
# RGB格式影响：dataformats参数指定图像数组的格式为'HWC'（高度、宽度、通道）
writer.add_image('test', img_array, 3, dataformats='HWC')
#3.关闭SummaryWriter对象
writer.close()