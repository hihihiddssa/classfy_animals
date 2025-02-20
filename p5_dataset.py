#datasets用于加载数据集，Dataset是一个抽象类，为了使用Dataset，需要继承它并实现两个方法：__len__和__getitem__。
from torch.utils.data import Dataset
from PIL import Image
import os
# 自定义数据集类，继承自torch.utils.data.Dataset
class MyDataset(Dataset):
    # 初始化方法（定义好传入的数据是什么，这些数据拿来构建了哪些新的数据）
    def __init__(self, root_dir, label_dir):
        # 根目录
        self.root_dir = root_dir
        # 标签目录
        self.label_dir = label_dir
        # 构建完整路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获取目录下所有图片的文件名列表
        self.img_path = os.listdir(self.path)

# 私有方法_init_ 之后就可以直接使用 MyDataset.root_dir 和 MyDataset.label_dir 来访问根目录和标签目录
# 公有方法则需要先实例化对象，然后调用方法
    # 在这里可以把传入的数据按照索引进行处理
    def __getitem__(self, index):
       #引入img_path后，可以根据索引获取图片文件名
        # 根据索引获取图片文件名
        img_name = self.img_path[index]#（上面构建的img_path,是列出了所有图片的文件名）
       #根据索引获取图片和标签
        # 构建图片文件的完整路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 打开图片文件
        img = Image.open(img_item_path)
        # 标签即为目录名
        label = self.label_dir
        
        # 返回图片和标签
        return img, label

    # 获取数据集的长度
    def __len__(self):
        return len(self.img_path)

# 数据集目录
root_dir = r'dataset\train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'

#实例化两个数据集
ants_dataset = MyDataset(root_dir, ants_label_dir)
bees_dataset = MyDataset(root_dir, bees_label_dir)

# 将两个数据集合并
train_dataset = ants_dataset + bees_dataset





#写入标签
# import os
#
# root_dir = 'dataset/train'
# target_dir = 'ants_image'
# img_path = os.listdir(os.path.join(root_dir, target_dir))
# label = target_dir.split('_')[0]
# out_dir = 'ants_label'
# for i in img_path:
#     file_name = i.split('.jpg')[0]
#     with open(os.path.join(root_dir, out_dir,"{}.txt".format(file_name)),'w') as f:
#         f.write(label)