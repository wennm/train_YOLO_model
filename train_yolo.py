#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11/12训练主程序
支持自定义配置文件，包括模型选择、训练参数、类别强化等功能
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List


def setup_interrupt_handlers():
    """设置优雅的中断处理器"""
    import signal
    import subprocess
    import time
    import threading

    def graceful_cleanup():
        """优雅的清理函数"""
        print("\n🛑 接收到中断信号，正在优雅地停止训练...")
        print("请稍候，正在保存当前训练状态...")

        cleanup_steps = [
            {
                'description': '保存训练状态',
                'command': ['pkill', '-TERM', '-f', 'Ultralytics'],
                'timeout': 10
            },
            {
                'description': '清理临时文件',
                'command': ['pkill', '-TERM', '-f', '_temp_'],
                'timeout': 5
            }
        ]

        for step in cleanup_steps:
            try:
                print(f"  🔄 {step['description']}...")
                subprocess.run(step['command'], timeout=step['timeout'], capture_output=True)
                time.sleep(1)  # 给系统一些处理时间
                print(f"  ✅ {step['description']}完成")
            except subprocess.TimeoutExpired:
                print(f"  ⚠️ {step['description']}超时")
            except Exception as e:
                print(f"  ⚠️ {step['description']}出现错误: {e}")

        # 等待清理完成后，再检查是否还有顽固进程
        time.sleep(2)

        # 只有在必要时才使用强制终止
        try:
            # 检查是否还有Ultralytics进程在运行
            result = subprocess.run(['pgrep', '-f', 'Ultralytics'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print("  🔄 发现仍在运行的进程，尝试温和终止...")
                subprocess.run(['pkill', '-TERM', '-f', 'yolo'], timeout=5)
                time.sleep(3)

                # 最后检查，如果还在运行才使用kill
                result = subprocess.run(['pgrep', '-f', 'yolo'],
                                      capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    print("  ⚠️ 温和终止失败，使用强制终止...")
                    subprocess.run(['pkill', '-KILL', '-f', 'yolo'], timeout=3)
            else:
                print("  ✅ 所有进程已正常终止")
        except Exception as e:
            print(f"  ⚠️ 进程检查出现错误: {e}")

        print("✅ 优雅清理完成")

    return graceful_cleanup

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    print("请先安装ultralytics库: pip install ultralytics")
    exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """YOLO11/12训练器类"""

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

        # 基础训练参数
        args = {
            'data': self.config['dataset']['dataset_yaml'],
            'epochs': training_config.get('epochs', 100),
            'batch': training_config.get('batch_size', 16),
            'imgsz': training_config.get('image_size', 640),
            'lr0': training_config.get('learning_rate', 0.01),
            'device': training_config.get('device', '0' if self._has_cuda() else 'cpu'),
            'workers': training_config.get('workers', 8),
            'name': training_config.get('experiment_name', 'yolo_experiment'),
            'save_period': training_config.get('save_period', -1),  # -1表示只保存最后一个
            'cache': training_config.get('cache', 'ram'),
            'exist_ok': training_config.get('exist_ok', False),
            'resume': training_config.get('resume', False),
            'verbose': training_config.get('verbose', True),
            'patience': training_config.get('patience', 50),
            'plots': training_config.get('plots', True),
            'rect': training_config.get('rect', False),
            'optimizer': training_config.get('optimizer', 'auto'),
            'val': training_config.get('val', True),
            'save_json': training_config.get('save_json', False),
            'freeze': training_config.get('freeze', False),
            'multi_scale': training_config.get('multi_scale', False),

            # 超参数直接添加到训练参数中
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

        # 数据增强参数
        if 'augmentation' in training_config:
            aug_config = training_config['augmentation']
            args.update({
                'hsv_h': aug_config.get('hsv_h', 0.015),
                'hsv_s': aug_config.get('hsv_s', 0.7),
                'hsv_v': aug_config.get('hsv_v', 0.4),
                'degrees': aug_config.get('degrees', 0.0),
                'translate': aug_config.get('translate', 0.1),
                'scale': aug_config.get('scale', 0.5),
                'shear': aug_config.get('shear', 0.0),
                'perspective': aug_config.get('perspective', 0.0),
                'flipud': aug_config.get('flipud', 0.0),
                'fliplr': aug_config.get('fliplr', 0.5),
                'mosaic': aug_config.get('mosaic', 1.0),
                'mixup': aug_config.get('mixup', 0.0),
                      })

        # 验证阈值参数现在在验证时设置，不在这里传递
        # 新版ultralytics不再支持在train时直接设置这些参数

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
        """创建自定义超参数字典，新版ultralytics不再使用hyp参数"""
        # 新版ultralytics将超参数直接集成在训练参数中
        # 这里返回一个空字典，因为我们已经将所有参数包含在setup_training_args中
        return {}

    def train(self):
        """开始训练"""
        logger.info("开始YOLO11/12模型训练...")

        # 检查磁盘空间
        self._check_disk_space()

        # 检查系统资源状态
        self._check_system_resources()

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
        try:
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"训练配置已保存到: {config_save_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

        # 启动资源监控
        self._setup_resource_monitoring()

        # 开始训练
        try:
            logger.info("🚀 开始训练，使用 Ctrl+C 可以优雅中断")
            logger.info("💡 资源监控已启动，会在内存/GPU不足时预警")
            results = model.train(**training_args)

            logger.info("✅ 训练完成!")
            logger.info(f"最佳模型保存在: {results.save_dir / 'weights' / 'best.pt'}")

            # 评估模型
            if self.config['training'].get('evaluate', True):
                logger.info("📊 开始模型评估...")
                try:
                    metrics = model.val(data=self.config['dataset']['dataset_yaml'])
                    logger.info(f"mAP50: {metrics.box.map50:.4f}")
                    logger.info(f"mAP50-95: {metrics.box.map:.4f}")
                except Exception as e:
                    logger.error(f"评估过程出现错误: {e}")

            return results

        except KeyboardInterrupt:
            logger.info("⏹️  训练被用户中断，正在优雅退出...")
            logger.info("💾 Ultralytics会自动保存当前训练状态")
            logger.info("📁 训练日志和权重文件保存在: runs/detect/")
            return None

        except Exception as e:
            logger.error(f"❌ 训练过程中出现错误: {e}")
            logger.info("🔄 尝试保存当前训练状态...")
            # 这里可以添加紧急保存逻辑
            raise

        finally:
            # 停止资源监控
            self._stop_resource_monitoring()
            # 清理资源
            logger.info("🧹 清理训练资源...")
            self._cleanup_resources()

    def _check_disk_space(self):
        """检查磁盘空间"""
        import shutil
        try:
            total, used, free = shutil.disk_usage(".")
            free_gb = free // (1024**3)

            if free_gb < 5:
                logger.warning(f"⚠️ 磁盘空间不足，剩余 {free_gb} GB")
                logger.warning("建议清理磁盘空间后再开始训练")
            else:
                logger.info(f"💾 磁盘空间充足，剩余 {free_gb} GB")

        except Exception as e:
            logger.warning(f"无法检查磁盘空间: {e}")

    def _check_system_resources(self):
        """检查系统资源状态，预防OOM等系统级中断"""
        try:
            import psutil
            import torch

            logger.info("🔍 检查系统资源状态...")

            # 检查内存使用情况
            memory = psutil.virtual_memory()
            used_memory_gb = memory.used / (1024**3)
            total_memory_gb = memory.total / (1024**3)
            memory_percent = memory.percent

            logger.info(f"💾 内存使用: {used_memory_gb:.1f}/{total_memory_gb:.1f} GB ({memory_percent:.1f}%)")

            if memory_percent > 95:
                logger.warning(f"⚠️ 内存使用过高 ({memory_percent:.1f}%)，可能导致OOM")
                logger.warning("建议：")
                logger.warning("  - 减少batch size")
                logger.warning("  - 减少workers数量")
                logger.warning("  - 关闭其他内存占用程序")

            # 检查GPU内存
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_cached = torch.cuda.memory_reserved(i) / (1024**3)
                    gpu_percent = (gpu_allocated / gpu_memory) * 100

                    logger.info(f"🎮 GPU {i}: {gpu_allocated:.1f}/{gpu_memory:.1f} GB 已分配 ({gpu_percent:.1f}%)")

                    if gpu_percent > 95:
                        logger.warning(f"⚠️ GPU {i} 内存使用过高 ({gpu_percent:.1f}%)")
                        logger.warning("建议减少batch size或模型大小")

                # 清理GPU缓存
                torch.cuda.empty_cache()

            # 检查CPU负载
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"🖥️  CPU使用率: {cpu_percent:.1f}%")

            if cpu_percent > 95:
                logger.warning(f"⚠️ CPU使用率过高 ({cpu_percent:.1f}%)")
                logger.warning("建议减少workers数量")

        except ImportError:
            logger.warning("⚠️ 未安装psutil，无法监控系统资源")
            logger.info("安装命令: pip install psutil")
        except Exception as e:
            logger.warning(f"系统资源检查失败: {e}")

    def _setup_resource_monitoring(self):
        """设置资源监控线程"""
        import threading
        import time
        import psutil
        import torch

        def resource_monitor():
            """资源监控函数"""
            try:
                while getattr(self, '_monitoring_active', True):
                    # 检查内存使用
                    memory = psutil.virtual_memory()
                    if memory.percent > 95:
                        logger.warning(f"⚠️ 内存使用危险 ({memory.percent:.1f}%) - 请准备中断训练")

                    # 检查GPU内存
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            gpu_memory = torch.cuda.get_device_properties(i).total_memory
                            gpu_allocated = torch.cuda.memory_allocated(i)
                            gpu_percent = (gpu_allocated / gpu_memory) * 100

                            if gpu_percent > 95:
                                logger.error(f"❌ GPU {i} 内存即将耗尽 ({gpu_percent:.1f}%)")
                                logger.error("⚠️ 系统可能随时中断进程！")
                                logger.error("建议立即使用 Ctrl+C 优雅退出")

                    time.sleep(30)  # 每30秒检查一次

            except Exception as e:
                logger.warning(f"资源监控线程出现错误: {e}")

        # 启动监控线程
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
        monitor_thread.start()
        logger.info("🔍 资源监控已启动 (每30秒检查一次)")

    def _stop_resource_monitoring(self):
        """停止资源监控"""
        self._monitoring_active = False
        logger.info("🔍 资源监控已停止")

    def _cleanup_resources(self):
        """清理训练资源"""
        try:
            import gc
            import torch

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🗑️ GPU缓存已清理")

            # 强制垃圾回收
            gc.collect()
            logger.info("🗑️ 内存垃圾回收完成")

        except Exception as e:
            logger.warning(f"资源清理过程中出现错误: {e}")


def main():
    """主函数"""
    import signal
    import os
    import threading
    import time
    import atexit

    # 优雅退出标志
    global graceful_shutdown_requested
    graceful_shutdown_requested = False

    def signal_handler(signum, frame):
        global graceful_shutdown_requested
        if graceful_shutdown_requested:
            logger.warning("强制终止信号已接收，立即退出...")
            os._exit(1)

        graceful_shutdown_requested = True
        logger.info(f"接收到中断信号 {signum}，开始优雅退出...")

        # 注册优雅清理函数
        graceful_cleanup = setup_interrupt_handlers()

        # 执行优雅清理
        graceful_cleanup()

        # 正常退出
        logger.info("优雅退出完成")
        sys.exit(0)

    # 注册退出时的清理函数
    def atexit_handler():
        """程序正常退出时的清理函数"""
        if not graceful_shutdown_requested:
            logger.info("程序正常退出，无需特殊清理")

    atexit.register(atexit_handler)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 设置进程组，便于信号传播
    try:
        os.setpgrp()
    except:
        pass

    parser = argparse.ArgumentParser(description='YOLO11/12训练程序')
    parser.add_argument('--config', type=str, default='train_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='恢复训练')
    parser.add_argument('--device', type=str, default=0,
                       help='指定设备，如0,1,2,3或cpu')

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        print("请创建train_config.yaml文件或使用--config指定配置文件路径")
        return

    try:
        # 创建训练器
        trainer = YOLOTrainer(args.config)

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