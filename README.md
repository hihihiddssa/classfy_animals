# 🐶 Animal Image Classification / 动物图片分类

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Network-green.svg)](https://en.wikipedia.org/wiki/Artificial_neural_network)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is for learning neural networks. It implements an animal image classification system, with a completed pipeline for classifying dog images.  
本项目主要用于学习神经网络，实现了动物图片分类流程，目前已完成对小狗图片的分类检测。

---

## 🌟 Features / 功能亮点

- 🐶 Dog image classification / 小狗图片识别与分类
- 🧠 Neural network model training and inference / 神经网络模型训练与推理
- 📈 Training and evaluation process logging / 训练与评估过程记录
- 🔄 Easily extendable for other animal species / 易于扩展到其他动物类别
- ⚡ 100% Python implementation / 完全基于 Python 实现

---

## 🚀 Quick Start / 快速开始

1. **Clone the repository / 克隆代码库**
    ```bash
    git clone https://github.com/hihihiddssa/classfy_animals.git
    cd classfy_animals
    ```

2. **Install dependencies / 安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your dataset / 准备数据集**
    - Place your animal images in the `data/` folder  
      将动物图片放入 `data/` 目录

4. **Train the model / 训练模型**
    ```bash
    python train.py
    ```

5. **Classify an image / 测试分类效果**
    ```bash
    python classify.py --image_path ./data/dog1.jpg
    ```

---

## 🔧 Project Structure / 项目结构

```
classfy_animals/
├── train.py                # Training script / 训练脚本
├── classify.py             # Classification script / 分类脚本
├── requirements.txt        # Dependencies list / 依赖列表
├── README.md               # Project documentation / 项目说明
├── data/                   # Dataset directory / 数据集目录
├── model/                  # Trained models / 训练好的模型
├── utils/                  # Utility functions / 工具函数
└── ...
```

---

## 📦 Dependencies / 依赖说明

- Python 3.7+
- TensorFlow or PyTorch (depending on your implementation)
- Numpy
- Pandas
- OpenCV
- Others, see `requirements.txt`

---

## 💡 Contribution / 贡献方式

Feel free to submit [Issues](https://github.com/hihihiddssa/classfy_animals/issues) or [Pull Requests](https://github.com/hihihiddssa/classfy_animals/pulls) to improve the project.  
欢迎提交 Issue 或 Pull Request 改进本项目功能！

---

## 📄 License / 许可证

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  
本项目采用 MIT 许可证，详情见 LICENSE 文件。

---

## ✨ Author / 作者

- [hihihiddssa](https://github.com/hihihiddssa)

# 🐕 Dog Image Classification Project / 小狗图片分类检测项目

---

## 📁 File Descriptions / 程序文件详解

### Core Modules / 核心功能模块

#### Data Processing / 数据处理相关
- **p5_dataset.py**
  - Custom dataset class `MyDataset`
  - Inherits from `torch.utils.data.Dataset`
  - Implements `__len__` and `__getitem__` methods
  - Used for loading and processing image data
  - 自定义数据集类 `MyDataset`，继承自 `torch.utils.data.Dataset`，实现了 `__len__` 和 `__getitem__` 方法，用于加载和处理图片数据

- **p8_tensorboard.py**
  - Example of using TensorBoard visualization tools
  - Shows how to log training data
  - Includes methods for visualizing images
  - TensorBoard 可视化工具使用示例，展示如何记录训练过程数据，包含图像数据的可视化方法

- **p9_transforms.py**
  - Data preprocessing tool example
  - Shows basic usage of the transforms module
  - Includes image transformation and normalization operations
  - 数据预处理工具示例，展示 transforms 模块的基本使用，包含图像转换、归一化等操作

- **p10_UsefulTransforms.py**
  - Collection of common image transformation methods
  - Includes ToTensor, Normalize, Resize, etc.
  - Shows how to compose multiple transforms
  - 常用图像变换方法集合，包含 ToTensor、Normalize、Resize 等转换，展示如何组合多个 transforms

#### Model Related / 模型相关
- **p16_model.py**
  - Example of basic neural network structure
  - Shows usage of `nn.Module`
  - Implements `__init__` and `forward` methods
  - 神经网络基础结构示例，展示 `nn.Module` 的基本用法，包含 `__init__` 和 `forward` 方法实现

- **p23_createmodel.py**
  - Example of creating a CNN model
  - Includes convolution, pooling, and fully connected layers
  - Implements a complete image classification network
  - CNN 模型创建示例，包含卷积层、池化层、全连接层，实现完整图像分类网络结构

- **p27_model.py**
  - Main model definition file
  - Implements CNN for image classification
  - Includes detailed network layer configuration
  - 主要模型定义文件，实现用于图像分类的 CNN 网络，包含详细的网络层配置

#### Training Related / 训练相关
- **p27_fulltrainning.py**
  - Complete model training process
  - Includes data loading, training, loss calculation
  - Uses CIFAR10 dataset for training
  - 完整模型训练流程，包含数据加载、模型训练、损失计算等，使用 CIFAR10 数据集进行训练

- **p27_fulltrainning_correctratio.py**
  - Training process with accuracy calculation
  - Implements test set evaluation
  - Logs various training metrics
  - 带有准确率计算的训练流程，实现测试集评估，记录训练过程各项指标

- **p30_GPU_1.py**
  - GPU training support
  - Shows how to move model and data to GPU
  - Complete GPU training workflow
  - GPU 训练支持实现，展示如何将模型和数据迁移到 GPU，包含 GPU 训练完整流程

### Tools & Utilities / 工具和辅助模块

- **p11_dataset_transform.py**
  - Batch processing example for datasets
  - Shows how to apply transforms to datasets
  - Includes data visualization methods
  - 数据集批量处理示例，展示如何对数据集应用转换，包含数据可视化方法

- **p15_dataloader.py**
  - DataLoader usage example
  - Shows configuration of data loader
  - Includes batching and shuffling
  - DataLoader 使用示例，展示数据加载器配置方法，包含批处理、打乱等功能

- **p24_lossfunc_1.py** & **p24_lossfunc_2netwoek.py**
  - Loss function implementation and usage example
  - Includes comparison of multiple loss functions
  - Shows how to use loss functions in networks
  - 损失函数实现和使用示例，包含多种损失函数对比，展示如何在网络中使用损失函数

- **p25_VGG.py**
  - VGG model usage example
  - Shows how to use pre-trained models
  - Includes model modification and fine-tuning
  - VGG 模型使用示例，展示如何使用预训练模型，包含模型修改和微调方法

- **p26_model_readandsave.py**
  - Example of saving and loading models
  - Shows different model storage methods
  - Includes saving and reading model parameters
  - 模型保存和加载示例，展示不同模型存储方法，包含模型参数保存和读取

### Testing & Validation / 测试和验证

- **p29_details.py**
  - Model testing and validation process
  - Includes test set evaluation methods
  - Shows how to calculate model performance metrics
  - 模型测试和验证流程，包含测试集评估方法，展示如何计算模型性能指标

- **p32_testnet.py**
  - Network testing implementation
  - Complete testing workflow
  - Shows how to evaluate model performance
  - 网络测试实现，包含完整测试流程，展示如何评估模型效果

---

## 🔧 Usage Instructions / 使用说明

1. First run `p5_dataset.py` to prepare the dataset  
   首先运行 `p5_dataset.py` 准备数据集
2. Define the network structure with `p27_model.py`  
   使用 `p27_model.py` 定义网络结构
3. Start training with `p27_fulltrainning.py`  
   通过 `p27_fulltrainning.py` 开始训练
4. Test model performance with `p32_testnet.py`  
   使用 `p32_testnet.py` 测试模型效果
5. View the training process using TensorBoard  
   可以通过 TensorBoard 查看训练过程

---

## 📈 Visualization / 可视化

Run TensorBoard:

```bash
tensorboard --logdir=logs
```
运行 TensorBoard:

```bash
tensorboard --logdir=logs
```

---

## 💡 注意事项

- 确保已安装所有必要的依赖
- GPU 训练需要 CUDA 支持
- 建议使用 Python 3.7 或更高版本
