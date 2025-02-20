# 🐕 小狗图片分类检测项目

## 📁 程序文件详解

### 核心功能模块

#### 数据处理相关
- **p5_dataset.py**
  - 自定义数据集类 `MyDataset`
  - 继承自 `torch.utils.data.Dataset`
  - 实现了 `__len__` 和 `__getitem__` 方法
  - 用于加载和处理图片数据

- **p8_tensorboard.py**
  - TensorBoard 可视化工具的使用示例
  - 展示如何记录训练过程中的数据
  - 包含图像数据的可视化方法

- **p9_transforms.py** 
  - 数据预处理工具示例
  - 展示 transforms 模块的基本使用
  - 包含图像转换、归一化等操作

- **p10_UsefulTransforms.py**
  - 常用图像变换方法的集合
  - 包含 ToTensor、Normalize、Resize 等转换
  - 展示如何组合多个 transforms

#### 模型相关
- **p16_model.py**
  - 神经网络基础结构示例
  - 展示 `nn.Module` 的基本用法
  - 包含 `__init__` 和 `forward` 方法的实现

- **p23_createmodel.py**
  - CNN 模型创建示例
  - 包含卷积层、池化层、全连接层
  - 实现了完整的图像分类网络结构

- **p27_model.py**
  - 主要的模型定义文件
  - 实现了用于图像分类的 CNN 网络
  - 包含详细的网络层配置

#### 训练相关
- **p27_fulltrainning.py**
  - 完整的模型训练流程
  - 包含数据加载、模型训练、损失计算等
  - 使用 CIFAR10 数据集进行训练

- **p27_fulltrainning_correctratio.py**
  - 带有准确率计算的训练流程
  - 实现了测试集评估
  - 记录训练过程中的各项指标

- **p30_GPU_1.py**
  - GPU 训练支持实现
  - 展示如何将模型和数据迁移到 GPU
  - 包含 GPU 训练的完整流程

### 工具和辅助模块

- **p11_dataset_transform.py**
  - 数据集批量处理示例
  - 展示如何对数据集应用转换
  - 包含数据可视化方法

- **p15_dataloader.py**
  - DataLoader 使用示例
  - 展示数据加载器的配置方法
  - 包含批处理、打乱等功能

- **p24_lossfunc_1.py** 和 **p24_lossfunc_2netwoek.py**
  - 损失函数实现和使用示例
  - 包含多种损失函数的对比
  - 展示如何在网络中使用损失函数

- **p25_VGG.py**
  - VGG 模型使用示例
  - 展示如何使用预训练模型
  - 包含模型修改和微调方法

- **p26_model_readandsave.py**
  - 模型保存和加载示例
  - 展示不同的模型存储方法
  - 包含模型参数的保存和读取

### 测试和验证
- **p29_details.py**
  - 模型测试和验证流程
  - 包含测试集评估方法
  - 展示如何计算模型性能指标

- **p32_testnet.py**
  - 网络测试实现
  - 包含完整的测试流程
  - 展示如何评估模型效果

## 🔧 使用说明

1. 首先运行 `p5_dataset.py` 准备数据集
2. 使用 `p27_model.py` 定义网络结构
3. 通过 `p27_fulltrainning.py` 开始训练
4. 使用 `p32_testnet.py` 测试模型效果
5. 可以通过 TensorBoard 查看训练过程

## 📈 可视化

运行 TensorBoard:

```bash
tensorboard --logdir=logs
```

## 💡 注意事项

- 确保已安装所有必要的依赖
- GPU 训练需要 CUDA 支持
- 建议使用 Python 3.7 或更高版本
