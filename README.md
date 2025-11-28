# YOLO11 训练框架

这是一个完整的YOLO11目标检测模型训练框架，支持自定义配置、类别强化训练、自动类别识别等功能。

## 功能特性

- ✅ 支持所有YOLO11模型变体 (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
- ✅ 完整的训练参数配置
- ✅ 数据增强参数自定义
- ✅ 类别特定强化训练
- ✅ 自动识别数据集类别
- ✅ 多GPU训练支持
- ✅ 混合精度训练
- ✅ 训练进度监控和结果保存

## 文件结构

```
train_YOLO_model/
├── train_yolo11.py          # 主训练程序
├── train_config.yaml        # 训练配置文件
├── example_dataset.yaml     # 示例数据集配置
└── README.md               # 使用说明
```

## 安装依赖

```bash
# 安装PyTorch (根据您的CUDA版本选择)
pip install torch torchvision torchaudio

# 安装ultralytics
pip install ultralytics

# 可选: 安装其他依赖
pip install pyyaml numpy opencv-python
```

## 快速开始

### 1. 准备数据集

确保您的数据集按照YOLO格式组织：

```
your_dataset/
├── train/          # 训集图片
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/            # 验证集图片
│   ├── val1.jpg
│   ├── val2.jpg
│   └── ...
├── train/labels/   # 训练集标签
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── val/labels/     # 验证集标签
    ├── val1.txt
    ├── val2.txt
    └── ...
```

### 2. 创建数据集配置文件

参考 `example_dataset.yaml` 创建您的数据集配置文件：

```yaml
# dataset.yaml
path: /path/to/your/dataset
train: train
val: val
nc: 2
names:
  0: person
  1: motorcycle
```

### 3. 配置训练参数

编辑 `train_config.yaml` 文件：

```yaml
# 模型配置
model:
  name: "yolo11s"  # 选择模型大小

# 训练参数
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  device: "0"  # GPU设备

# 数据集配置
dataset:
  dataset_yaml: "/path/to/your/dataset.yaml"
```

### 4. 开始训练

```bash
# 基础训练
python train_yolo11.py

# 指定配置文件
python train_yolo11.py --config /path/to/your/config.yaml

# 使用特定GPU
python train_yolo11.py --device 1

# 恢复训练
python train_yolo11.py --resume
```

## 详细配置说明

### 模型配置

支持的模型类型：
- `yolo11n`: 最小最快，适合边缘设备
- `yolo11s`: 小型，平衡速度和精度
- `yolo11m`: 中型，推荐大多数场景
- `yolo11l`: 大型，更高精度
- `yolo11x`: 超大型，最高精度

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| epochs | 训练轮次 | 100 |
| batch_size | 批次大小 | 16 |
| image_size | 输入图像尺寸 | 640 |
| learning_rate | 初始学习率 | 0.01 |
| device | 训练设备 | "0" |
| workers | 数据加载线程数 | 8 |

### 数据增强参数

| 参数 | 说明 | 范围 |
|------|------|------|
| hsv_h | 色调增强 | 0.0-1.0 |
| hsv_s | 饱和度增强 | 0.0-1.0 |
| hsv_v | 明度增强 | 0.0-1.0 |
| degrees | 旋转角度 | 0.0-45.0 |
| translate | 平移 | 0.0-1.0 |
| scale | 缩放 | 0.0-2.0 |
| flipud | 上下翻转概率 | 0.0-1.0 |
| fliplr | 左右翻转概率 | 0.0-1.0 |

### 类别强化训练

可以为特定类别设置强化参数：

```yaml
class_specific:
  0:  # 类别ID (从0开始)
    loss_weight: 2.0        # 损失权重倍数
    augmentation_scale: 1.5 # 数据增强强度
    min_samples_weight: 1.2 # 最小样本权重
    lr_multiplier: 1.0      # 学习率倍数
```

## 训练监控

训练过程中会自动保存：

- **最佳模型**: `runs/detect/experiment_name/weights/best.pt`
- **最新模型**: `runs/detect/experiment_name/weights/last.pt`
- **训练日志**: `runs/detect/experiment_name/results.csv`
- **训练配置**: `runs/detect/experiment_name/train_config.yaml`
- **TensorBoard日志**: `runs/detect/experiment_name`

## 结果评估

训练完成后会自动进行评估，输出：

- mAP@0.5
- mAP@0.5:0.95
- 各类别的精确率和召回率

## 高级功能

### 多GPU训练

```yaml
advanced:
  multi_gpu:
    enabled: true
    sync_bn: true
```

### 混合精度训练

```yaml
advanced:
  mixed_precision:
    enabled: true
```

### 早停

```yaml
advanced:
  early_stopping:
    enabled: true
    patience: 50
    min_delta: 0.001
```

## 常见问题

### 1. CUDA内存不足

- 减少batch_size
- 减小image_size
- 使用更小的模型 (如yolo11n)

### 2. 训练速度慢

- 增加workers数量
- 启用混合精度训练
- 使用SSD缓存数据

### 3. 模型不收敛

- 调整学习率
- 增加训练轮次
- 检查数据质量
- 调整数据增强参数

### 4. 类别不平衡

- 使用类别特定参数
- 调整损失权重
- 使用重采样技术

## 示例命令

```bash
# 使用yolo11n模型，快速训练
python train_yolo11.py --config configs/quick_train.yaml

# 使用yolo11x模型，完整训练
python train_yolo11.py --config configs/full_train.yaml

# 在多GPU上训练
python train_yolo11.py --device 0,1,2,3

# 恢复中断的训练
python train_yolo11.py --resume
```

## 技术支持

如果遇到问题，请检查：

1. 依赖库是否正确安装
2. 配置文件格式是否正确
3. 数据集路径是否存在
4. 设备是否可用 (GPU/CPU)

更多信息请参考 [Ultralytics官方文档](https://docs.ultralytics.com/)。