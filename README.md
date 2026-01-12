# YOLO11/12 训练框架

一个完整的 YOLO 目标检测模型训练框架，支持 YOLO11/12 标准检测和 YOLO12-OBB 旋转框检测，配备完善的数据集处理工具链。

## 功能特性

### 通用YOLO训练框架 (train_yolo.py)
- ✅ 支持所有 YOLO11/12 模型变体 (n/s/m/l/x)
- ✅ 负样本训练支持（自动处理空标签文件）
- ✅ 自动识别数据集类别
- ✅ 系统资源监控（内存、GPU、CPU）
- ✅ 优雅中断处理机制（Ctrl+C 安全退出）
- ✅ 完整的训练参数配置
- ✅ 数据增强参数自定义
- ✅ 类别特定强化训练
- ✅ 多GPU训练支持
- ✅ 磁盘空间检查

### YOLO12-OBB训练框架 (train_yolo12_obb.py)
- ✅ 支持旋转框检测（Oriented Bounding Box）
- ✅ YOLO12n/s/m/l/x 全系列模型
- ✅ CUDA OOM 智能处理
- ✅ GPU 进程自动清理
- ✅ 8类交通事故检测优化

### 数据集处理工具链
- 📊 数据集分析和可视化
- 🔄 数据集格式转换（XML ↔ YOLO/OBB）
- 🧹 数据集清理和验证
- 📹 视频帧提取
- 🌙 红外图像处理

## 文件结构

```
train_YOLO_model/
├── train_yolo.py                      # 通用YOLO11/12训练框架
├── train_yolo12_obb.py                # YOLO12-OBB训练框架
├── train_yolov13.py                   # YOLOv13训练脚本
├── requirements.txt                   # Python依赖列表
├── config/                            # 训练配置文件目录
│   ├── train_config_model8.5.yaml
│   ├── train_yolo12x_obb_7class.yaml
│   ├── train_yolo12x_obb_8class.yaml
│   └── train_yolov13_obb_8class.yaml
├── tool/                              # 数据集处理工具目录
│   ├── analyze_dataset_labels.py      # 数据集标签分析
│   ├── analyze_unlabeled_images.py    # 无标签图像分析
│   ├── clean_extra_labels.py          # 清理多余标签
│   ├── clean_unlabeled_val_images.py  # 清理无标签验证图像
│   ├── comprehensive_dataset_cleaner.py # 综合数据集清理
│   ├── data_processor.py              # 数据处理器
│   ├── infrared_image_processor.py    # 红外图像处理
│   ├── modify_and_reorder_labels.py   # 标签修改和重排序
│   ├── supplement_dataset.py          # 数据集补充
│   ├── video_frame_extractor.py       # 视频帧提取
│   ├── visualize_dataset.py           # 数据集可视化
│   ├── visualize_dataset_v2.py        # 数据集可视化v2
│   ├── visualize_obb_dataset.py       # OBB数据集可视化
│   ├── xml_to_yolo_obb.py             # XML转YOLO-OBB格式
│   ├── xzobb_to_xml.py                # XZOBB转XML格式
│   ├── yolo_standard_dataset_processor.py # 标准YOLO数据集处理
│   └── yolo_to_xml.py                 # YOLO转XML格式
└── README.md                          # 使用说明
```

## 安装依赖

### 方法1: 使用 requirements.txt
```bash
pip install -r requirements.txt
```

### 方法2: 手动安装
```bash
# 安装PyTorch (根据您的CUDA版本选择)
pip install torch torchvision torchaudio

# 安装ultralytics
pip install ultralytics

# 安装其他依赖
pip install pyyaml numpy opencv-python psutil polars
```

## 快速开始

### 1. 准备数据集

确保您的数据集按照YOLO格式组织：

```
your_dataset/
├── images/
│   ├── train/           # 训练集图片
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/             # 验证集图片
│       ├── val1.jpg
│       ├── val2.jpg
│       └── ...
└── labels/
    ├── train/           # 训练集标签
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/             # 验证集标签
        ├── val1.txt
        ├── val2.txt
        └── ...
```

**负样本支持**: 如果某些图像没有目标检测对象，训练框架会自动创建空的标签文件。

### 2. 创建数据集配置文件

创建 `dataset.yaml` 文件：

```yaml
# 标准检测格式
path: /path/to/your/dataset
train: images/train
val: images/val
nc: 2  # 类别数量
names:
  0: person
  1: motorcycle

# OBB检测格式（旋转框）
path: /path/to/your/dataset
train: images/train
val: images/val
nc: 8
names:
  0: car
  1: truck
  # ... 其他类别
```

### 3. 配置训练参数

在 `config/` 目录下创建配置文件，例如 `train_config.yaml`：

```yaml
# 模型配置
model:
  name: "yolo12x"  # 支持 yolo11n/s/m/l/x, yolo12n/s/m/l/x
  # pretrained: "path/to/pretrained/weights.pt"  # 可选

# 训练参数
training:
  epochs: 150
  batch_size: 16
  image_size: 640
  learning_rate: 0.01
  device: "0"  # "0" GPU0, "0,1,2" 多GPU, "cpu" CPU
  workers: 8

  # 学习率调度
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3.0

  # 损失权重
  box_loss_gain: 7.5
  cls_loss_gain: 0.5
  obj_loss_gain: 1.0

  # 其他参数
  patience: 50
  cache: "ram"  # "ram", "disk", 或 false
  optimizer: "SGD"  # "SGD", "Adam", "AdamW"
  val: true

# 数据增强
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0
  translate: 0.1
  scale: 0.5
  flipud: 0.5
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0

# 类别特定强化
class_specific:
  0:  # 类别ID
    loss_weight: 2.0
    augmentation_scale: 1.5

# 数据集配置
dataset:
  dataset_yaml: "/path/to/dataset.yaml"
  root_path: "/path/to/dataset"
```

### 4. 开始训练

#### 通用YOLO训练 (train_yolo.py)

```bash
# 使用默认配置
python train_yolo.py

# 指定配置文件
python train_yolo.py --config config/train_config_model8.5.yaml

# 指定GPU设备
python train_yolo.py --config config/train_config.yaml --device 0,1,2

# 恢复训练
python train_yolo.py --config config/train_config.yaml --resume
```

#### YOLO12-OBB训练 (train_yolo12_obb.py)

```bash
# 使用默认配置
python train_yolo12_obb.py

# 指定配置文件
python train_yolo12_obb.py --config config/train_yolo12x_obb_8class.yaml

# 指定GPU设备
python train_yolo12_obb.py --device 0,1

# 恢复训练
python train_yolo12_obb.py --config config/train_config.yaml --resume
```

## 支持的模型

### 通用YOLO框架
- **YOLO11**: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
- **YOLO12**: yolo12n, yolo12s, yolo12m, yolo12l, yolo12x

### OBB框架（仅YOLO12）
- yolo12n-obb, yolo12s-obb, yolo12m-obb, yolo12l-obb, yolo12x-obb

模型大小对比：
| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| n (nano) | 最少 | 最快 | 一般 | 边缘设备 |
| s (small) | 较少 | 快 | 良好 | 平衡速度和精度 |
| m (medium) | 中等 | 中等 | 较高 | 大多数场景 |
| l (large) | 较多 | 慢 | 高 | 高精度需求 |
| x (xlarge) | 最多 | 最慢 | 最高 | 最佳精度 |

## 训练监控与保护

### 资源监控
训练框架会实时监控：
- 💾 **内存使用率**: 超过95%时警告
- 🎮 **GPU内存**: 超过95%时预警
- 🖥️ **CPU使用率**: 超过90%时建议
- 💿 **磁盘空间**: 低于5GB时警告

### 优雅中断
- 按 `Ctrl+C` 触发优雅退出
- 自动保存当前训练状态
- Ultralytics 保留 last.pt 权重
- 进程安全清理

### 自动保存
- **最佳模型**: `runs/detect/experiment_name/weights/best.pt`
- **最新模型**: `runs/detect/experiment_name/weights/last.pt`
- **训练配置**: `runs/detect/experiment_name/train_config.yaml`
- **训练日志**: `runs/detect/experiment_name/results.csv`
- **训练图表**: `runs/detect/experiment_name/*.png`

## 数据集处理工具

### 1. 数据集分析
```bash
# 分析数据集标签分布
python tool/analyze_dataset_labels.py --dataset /path/to/dataset

# 分析无标签图像
python tool/analyze_unlabeled_images.py --dataset /path/to/dataset
```

### 2. 数据集可视化
```bash
# 可视化标准YOLO数据集
python tool/visualize_dataset.py --dataset /path/to/dataset --output ./vis_output

# 可视化OBB数据集
python tool/visualize_obb_dataset.py --dataset /path/to/dataset --output ./obb_vis
```

### 3. 数据集清理
```bash
# 清理多余标签
python tool/clean_extra_labels.py --dataset /path/to/dataset

# 清理无标签验证图像
python tool/clean_unlabeled_val_images.py --dataset /path/to/dataset

# 综合清理
python tool/comprehensive_dataset_cleaner.py --dataset /path/to/dataset
```

### 4. 格式转换
```bash
# XML转YOLO-OBB
python tool/xml_to_yolo_obb.py --xml_dir /path/to/xml --output_dir /path/to/output

# XZOBB转XML
python tool/xzobb_to_xml.py --input /path/to/xzobb --output /path/to/xml
```

### 5. 视频处理
```bash
# 提取视频帧
python tool/video_frame_extractor.py --video /path/to/video.mp4 --output_dir /path/to/frames --fps 1
```

### 6. 红外图像处理
```bash
# 处理红外图像
python tool/infrared_image_processor.py --input /path/to/infrared --output /path/to/output
```

## 训练参数详解

### 核心参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| epochs | 训练轮次 | 100 | 50-300 |
| batch_size | 批次大小 | 16 | 4-64 (取决于GPU) |
| image_size | 输入图像尺寸 | 640 | 320-1280 |
| learning_rate (lr0) | 初始学习率 | 0.01 | 0.001-0.01 |
| lrf | 最终学习率倍数 | 0.01 | 0.01-0.1 |
| device | 训练设备 | "0" | "0", "0,1", "cpu" |
| workers | 数据加载线程 | 8 | 4-16 |
| patience | 早停耐心值 | 50 | 20-100 |

### 数据增强参数

| 参数 | 说明 | 推荐值 | 适用场景 |
|------|------|--------|----------|
| hsv_h | 色调增强 | 0.015 | 彩色图像 |
| hsv_s | 饱和度增强 | 0.7 | 彩色图像 |
| hsv_v | 明度增强 | 0.4 | 所有场景 |
| degrees | 旋转角度 | 5.0-15.0 | 有方向目标 |
| translate | 平移范围 | 0.1-0.2 | 所有场景 |
| scale | 缩放范围 | 0.3-0.7 | 不同尺度目标 |
| flipud | 上下翻转 | 0.0-0.5 | 特定场景 |
| fliplr | 左右翻转 | 0.5 | 大多数场景 |
| mosaic | Mosaic增强 | 1.0 | 检测任务 |
| mixup | Mixup增强 | 0.0-0.2 | 小数据集 |

### 优化器选择

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| SGD | 稳定，泛化好 | 大多数场景 |
| Adam | 收敛快 | 小数据集 |
| AdamW | 防止过拟合 | 高精度需求 |

## 常见问题

### 1. CUDA内存不足 (OOM)

**解决方案:**
```yaml
# 减小batch_size
batch_size: 8  # 或更小

# 降低图像尺寸
image_size: 480  # 或更小

# 使用更小的模型
model:
  name: "yolo12s"  # 或 yolo12n

# 关闭缓存
cache: false
```

### 2. 训练速度慢

**优化建议:**
```yaml
# 增加workers
workers: 16

# 启用磁盘缓存
cache: "disk"

# 使用更小的模型或图像
image_size: 480

# 减少数据增强
degrees: 0.0
mosaic: 0.0
```

### 3. 模型不收敛

**调试步骤:**
1. 降低学习率: `learning_rate: 0.001`
2. 增加训练轮次: `epochs: 200`
3. 检查数据质量和标注
4. 调整损失权重
5. 减少数据增强强度

### 4. 类别不平衡

**解决方案:**
```yaml
class_specific:
  0:  # 少数类别
    loss_weight: 2.0
    min_samples_weight: 1.5
```

### 5. GPU进程残留

**清理命令:**
```bash
# 查看GPU进程
nvidia-smi

# 强制终止训练进程
pkill -9 -f yolo

# 清理Python进程
pkill -9 python
```

## 高级功能

### 多GPU训练
```yaml
training:
  device: "0,1,2,3"  # 使用4个GPU
```

### 混合精度训练
```yaml
training:
  amp: true  # 自动混合精度
```

### 多尺度训练
```yaml
training:
  multi_scale: true
  imgsz: [640, 640]  # base size
```

### 矩形训练
```yaml
training:
  rect: true  # 适用于统一比例的图像
```

### 早停配置
```yaml
training:
  patience: 50  # 50轮无改善则停止
```

## 训练结果评估

训练完成后会自动评估并输出：
- **mAP@0.5**: IoU=0.5时的平均精度
- **mAP@0.5:0.95**: IoU从0.5到0.95的平均精度
- **Precision**: 精确率
- **Recall**: 召回率
- **各类别指标**: 每个类别的详细指标

查看详细结果：
```bash
# 查看训练曲线
cat runs/detect/experiment_name/results.csv

# 使用TensorBoard
tensorboard --logdir runs/detect/experiment_name
```

## 技术支持

### 环境检查
```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查ultralytics
python -c "import ultralytics; print(ultralytics.__version__)"

# 检查GPU状态
nvidia-smi
```

### 日志级别
训练框架使用标准日志记录：
- INFO: 正常训练信息
- WARNING: 警告信息（资源不足等）
- ERROR: 错误信息

### 参考资料
- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [YOLO11 论文](https://arxiv.org/abs/...)
- [YOLO12 文档](https://github.com/ultralytics/ultralytics)

## 更新日志

### v2.0 (当前版本)
- 新增 YOLO12-OBB 训练支持
- 新增资源监控和优雅中断
- 新增负样本训练支持
- 完善 tool 工具链
- 改进配置文件结构

### v1.0
- 初始版本
- 支持 YOLO11 训练
- 基础数据集处理工具

## 许可证

本项目遵循 Apache 2.0 许可证。

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO实现
- PyTorch 团队 - 深度学习框架
