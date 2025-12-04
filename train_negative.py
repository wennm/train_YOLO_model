#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持负样本的YOLO11/12训练程序
专门用于火点检测，支持空标签文件（负样本）的训练
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import shutil
import random


def setup_interrupt_handlers():
    """设置中断处理器，确保DDP训练时能正确终止"""
    import signal
    import subprocess

    def emergency_cleanup():
        """紧急清理函数"""
        print("\n🚨 紧急终止训练进程...")
        cleanup_commands = [
            ['pkill', '-f', 'Ultralytics'],
            ['pkill', '-f', '_temp_'],
            ['pkill', '-f', 'yolo'],
            ['pkill', '-9', '-f', 'python']
        ]

        for cmd in cleanup_commands:
            try:
                subprocess.run(cmd, timeout=3, capture_output=True)
            except:
                pass
        print("✅ 清理完成")

    return emergency_cleanup

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    print("请先安装ultralytics库: pip install ultralytics")
    exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLONegativeTrainer:
    """支持负样本的YOLO11/12训练器类"""

    def __init__(self, config_path: str):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.validate_config()

        # 加载数据集配置
        self.dataset_config = self.load_dataset_config()
        self.num_classes = self.dataset_config.get('nc', 0)

        logger.info(f"检测到 {self.num_classes} 个类别: {self.dataset_config.get('names', [])}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载训练配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def load_dataset_config(self) -> Dict[str, Any]:
        """加载数据集配置文件"""
        dataset_yaml_path = self.config['dataset']['dataset_yaml']
        try:
            with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
                dataset_config = yaml.safe_load(f)
            logger.info(f"成功加载数据集配置: {dataset_yaml_path}")
            return dataset_config
        except Exception as e:
            logger.error(f"加载数据集配置文件失败: {e}")
            raise

    def validate_config(self):
        """验证配置文件的有效性"""
        required_keys = ['model', 'training', 'dataset']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置文件缺少必要字段: {key}")

        # 验证模型类型
        valid_models = [
            'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
            'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'
        ]
        model_name = self.config['model']['name']
        if model_name not in valid_models:
            raise ValueError(f"不支持的模型类型: {model_name}, 支持的类型: {valid_models}")

        logger.info("配置文件验证通过")

    def validate_dataset_structure(self):
        """验证数据集结构，处理负样本"""
        dataset_root = Path(self.config['dataset']['root_path'])

        train_images_dir = dataset_root / 'images' / 'train'
        train_labels_dir = dataset_root / 'labels' / 'train'
        val_images_dir = dataset_root / 'images' / 'val'
        val_labels_dir = dataset_root / 'labels' / 'val'

        # 检查目录是否存在
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            if not dir_path.exists():
                logger.warning(f"目录不存在: {dir_path}")

        # 统计负样本数量
        negative_train_count = self._count_negative_samples(train_labels_dir)
        negative_val_count = self._count_negative_samples(val_labels_dir)

        logger.info(f"训练集负样本数量: {negative_train_count}")
        logger.info(f"验证集负样本数量: {negative_val_count}")

        # 检查图像和标签对应关系
        self._check_image_label_consistency(train_images_dir, train_labels_dir, "训练集")
        self._check_image_label_consistency(val_images_dir, val_labels_dir, "验证集")

    def _count_negative_samples(self, labels_dir: Path) -> int:
        """统计负样本数量（空标签文件）"""
        if not labels_dir.exists():
            return 0

        negative_count = 0
        for label_file in labels_dir.glob('*.txt'):
            if label_file.stat().st_size == 0:
                negative_count += 1

        return negative_count

    def _check_image_label_consistency(self, images_dir: Path, labels_dir: Path, split_name: str):
        """检查图像和标签文件的一致性"""
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"{split_name}图像或标签目录不存在")
            return

        # 获取图像和标签文件集合
        image_files = {f.stem for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        label_files = {f.stem for f in labels_dir.glob('*.txt')}

        # 检查缺失的标签文件（负样本）
        missing_labels = image_files - label_files
        if missing_labels:
            logger.info(f"{split_name}中发现{len(missing_labels)}个图像没有对应标签文件（将作为负样本处理）")
            # 为这些图像创建空的标签文件
            for img_name in missing_labels:
                empty_label_path = labels_dir / f"{img_name}.txt"
                empty_label_path.touch()
                logger.debug(f"创建空标签文件: {empty_label_path}")

        # 检查多余的标签文件
        extra_labels = label_files - image_files
        if extra_labels:
            logger.warning(f"{split_name}中发现{len(extra_labels)}个标签文件没有对应的图像")

        logger.info(f"{split_name}统计:")
        logger.info(f"  图像文件: {len(image_files)}")
        logger.info(f"  标签文件: {len(label_files)}")
        logger.info(f"  有效样本: {len(image_files)}")

    def get_model_name(self) -> str:
        """获取完整的模型名称"""
        model_config = self.config['model']
        base_name = model_config['name']

        # 如果有预训练权重，使用预训练权重
        if 'pretrained' in model_config and model_config['pretrained']:
            return model_config['pretrained']

        return f"{base_name}.pt"

    def setup_training_args(self) -> Dict[str, Any]:
        """设置训练参数"""
        training_config = self.config['training']

        # 基础训练参数 - 针对火点检测优化
        args = {
            'data': self.config['dataset']['dataset_yaml'],
            'epochs': training_config.get('epochs', 100),
            'batch': training_config.get('batch_size', 16),
            'imgsz': training_config.get('image_size', 640),  # 火点检测通常不需要特别大的图像尺寸
            'lr0': training_config.get('learning_rate', 0.01),
            'device': training_config.get('device', '0' if self._has_cuda() else 'cpu'),
            'workers': training_config.get('workers', 8),
            'name': training_config.get('experiment_name', 'fire_detection_experiment'),
            'save_period': training_config.get('save_period', -1),
            'cache': training_config.get('cache', 'ram'),
            'exist_ok': training_config.get('exist_ok', False),
            'resume': training_config.get('resume', False),
            'verbose': training_config.get('verbose', True),
            'patience': training_config.get('patience', 50),
            'plots': training_config.get('plots', True),
            'rect': training_config.get('rect', False),
            'optimizer': training_config.get('optimizer', 'SGD'),  # SGD通常对火点检测效果更好
            'val': training_config.get('val', True),
            'save_json': training_config.get('save_json', False),
            'freeze': training_config.get('freeze', False),
            'multi_scale': training_config.get('multi_scale', True),  # 多尺度训练对火点检测很重要

            # 超参数 - 针对火点检测优化
            'lrf': self.config['training'].get('lrf', 0.01),
            'momentum': self.config['training'].get('momentum', 0.937),
            'weight_decay': self.config['training'].get('weight_decay', 0.0005),
            'warmup_epochs': self.config['training'].get('warmup_epochs', 3.0),
            'warmup_momentum': self.config['training'].get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config['training'].get('warmup_bias_lr', 0.1),
            'box': self.config['training'].get('box_loss_gain', 7.5),
            'cls': self.config['training'].get('cls_loss_gain', 0.5),
            'kobj': self.config['training'].get('obj_positive_weight', 1.0),
            'iou': self.config['training'].get('iou_threshold', 0.2),
        }

        # 数据增强参数 - 针对火点检测的合理增强
        if 'augmentation' in training_config:
            aug_config = training_config['augmentation']
            args.update({
                'hsv_h': aug_config.get('hsv_h', 0.015),  # 火点色调变化较小
                'hsv_s': aug_config.get('hsv_s', 0.7),    # 饱和度可适度变化
                'hsv_v': aug_config.get('hsv_v', 0.4),    # 亮度变化对火点检测重要
                'degrees': aug_config.get('degrees', 10.0),    # 火点检测可接受更大旋转
                'translate': aug_config.get('translate', 0.1),  # 平移
                'scale': aug_config.get('scale', 0.5),         # 缩放
                'shear': aug_config.get('shear', 0.0),         # 剪切
                'perspective': aug_config.get('perspective', 0.0),  # 透视变换
                'flipud': aug_config.get('flipud', 0.5),       # 上下翻转
                'fliplr': aug_config.get('fliplr', 0.5),       # 左右翻转
                'mosaic': aug_config.get('mosaic', 1.0),       # Mosaic增强
                'mixup': aug_config.get('mixup', 0.0),         # 火点检测不建议使用mixup
            })

        return args

    def setup_class_specific_params(self) -> Dict[str, Any]:
        """设置类别特定的强化参数"""
        class_specific = self.config.get('class_specific', {})

        if not class_specific:
            return {}

        # 验证类别ID是否有效
        valid_class_params = {}
        for class_id, params in class_specific.items():
            try:
                class_idx = int(class_id)
                if class_idx >= self.num_classes:
                    logger.warning(f"类别ID {class_idx} 超出范围(0-{self.num_classes-1})，跳过")
                    continue
                valid_class_params[class_idx] = params
                logger.info(f"启用类别 {class_idx} 的强化参数: {params}")
            except ValueError:
                logger.warning(f"无效的类别ID: {class_id}，跳过")

        return valid_class_params

    def _has_cuda(self) -> bool:
        """检查是否有CUDA支持"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def create_custom_hyp(self) -> Dict[str, Any]:
        """创建自定义超参数字典"""
        return {}

    def train(self):
        """开始训练"""
        logger.info("开始支持负样本的YOLO11/12火点检测训练...")

        # 首先验证数据集结构
        self.validate_dataset_structure()

        # 加载模型
        model_name = self.get_model_name()
        logger.info(f"加载模型: {model_name}")
        model = YOLO(model_name)

        # 设置训练参数
        training_args = self.setup_training_args()

        logger.info("训练参数:")
        for key, value in training_args.items():
            logger.info(f"  {key}: {value}")

        # 保存配置到训练结果目录
        save_dir = Path("runs/detect") / training_args['name']
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存训练配置
        config_save_path = save_dir / "train_config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"训练配置已保存到: {config_save_path}")

        # 开始训练
        try:
            results = model.train(**training_args)

            logger.info("训练完成!")
            logger.info(f"最佳模型保存在: {results.save_dir / 'weights' / 'best.pt'}")

            # 评估模型
            if self.config['training'].get('evaluate', True):
                logger.info("开始模型评估...")
                metrics = model.val(data=self.config['dataset']['dataset_yaml'])
                logger.info(f"mAP50: {metrics.box.map50:.4f}")
                logger.info(f"mAP50-95: {metrics.box.map:.4f}")

            return results

        except KeyboardInterrupt:
            logger.info("训练被用户中断，正在退出...")
            logger.info("检查是否保存了部分训练结果...")
            return None

        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    import signal
    import os
    import threading
    import time

    def signal_handler(signum, frame):
        logger.info(f"接收到中断信号 {signum}，正在清理...")

        # 强制终止所有相关进程
        logger.info("正在终止Ultralytics进程...")

        # 方法1: 终止当前进程组
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except:
            pass

        # 方法2: 查找并终止Ultralytics相关进程
        try:
            import subprocess
            subprocess.run(['pkill', '-f', 'Ultralytics'], timeout=5)
            subprocess.run(['pkill', '-f', '_temp_'], timeout=5)
        except:
            pass

        # 等待2秒后强制退出
        def delayed_exit():
            time.sleep(2)
            logger.info("强制退出...")
            os._exit(1)

        thread = threading.Thread(target=delayed_exit, daemon=True)
        thread.start()

        sys.exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 设置进程组，便于信号传播
    try:
        os.setpgrp()
    except:
        pass

    parser = argparse.ArgumentParser(description='支持负样本的YOLO11/12火点检测训练程序')
    parser.add_argument('--config', type=str, default='train_config_model2.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='恢复训练')
    parser.add_argument('--device', type=str, default=0,
                       help='指定设备，如0,1,2,3或cpu')

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        print("请创建train_config_model2.yaml文件或使用--config指定配置文件路径")
        return

    try:
        # 创建训练器
        trainer = YOLONegativeTrainer(args.config)

        # 如果指定了恢复训练
        if args.resume:
            trainer.config['training']['resume'] = True

        # 如果指定了设备
        if args.device:
            trainer.config['training']['device'] = args.device

        # 开始训练
        trainer.train()

    except Exception as e:
        logger.error(f"训练失败: {e}")
        return


if __name__ == "__main__":
    main()