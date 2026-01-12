#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv13训练框架
基于ultralytics官方API，适配红外摩托车检测任务
"""

import os
import sys
import yaml
import argparse
import logging
import atexit
import signal
import subprocess
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("请先安装ultralytics库: pip install ultralytics")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量：记录当前训练进程信息
_training_process_info = {
    'pid': os.getpid(),
    'device': None,
    'cleanup_registered': False
}


class YOLOv13Trainer:
    """YOLOv13训练器类"""

    # 清理标记：确保只清理一次
    _cleaned = False

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

    def load_config(self, config_path: str) -> dict:
        """加载训练配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def load_dataset_config(self) -> dict:
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
        valid_models = ['yolov13n', 'yolov13s', 'yolov13l', 'yolov13x']
        model_name = self.config['model']['name']
        if model_name not in valid_models:
            raise ValueError(f"不支持的模型类型: {model_name}, 支持的类型: {valid_models}")

        logger.info("配置文件验证通过")

    def get_training_args(self) -> dict:
        """获取训练参数"""
        training_config = self.config['training']
        aug_config = self.config.get('augmentation', {})

        # 获取任务类型（默认为detect，可选obb）
        task = training_config.get('task', 'detect')

        # 基础训练参数
        args = {
            'data': self.config['dataset']['dataset_yaml'],
            'task': task,  # 添加任务类型参数
            'epochs': training_config.get('epochs', 100),
            'batch': training_config.get('batch_size', 16),
            'imgsz': training_config.get('image_size', 640),
            'lr0': training_config.get('learning_rate', 0.01),
            'lrf': training_config.get('lrf', 0.01),
            'momentum': training_config.get('momentum', 0.937),
            'weight_decay': training_config.get('weight_decay', 0.0005),
            'warmup_epochs': training_config.get('warmup_epochs', 3.0),
            'warmup_momentum': training_config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': training_config.get('warmup_bias_lr', 0.1),
            'box': training_config.get('box_loss_gain', 7.5),
            'cls': training_config.get('cls_loss_gain', 0.5),
            'dfl': training_config.get('obj_loss_gain', 1.0),
            'iou': training_config.get('iou_threshold', 0.2),
            'device': training_config.get('device', '0'),
            'workers': training_config.get('workers', 8),
            'name': training_config.get('experiment_name', 'yolov13_experiment'),
            'save_period': training_config.get('save_period', -1),
            'cache': training_config.get('cache', 'ram'),
            'exist_ok': training_config.get('exist_ok', False),
            'resume': training_config.get('resume', False),
            'verbose': training_config.get('verbose', True),
            'patience': training_config.get('patience', 50),
            'plots': training_config.get('plots', True),
            'rect': training_config.get('rect', False),
            'optimizer': training_config.get('optimizer', 'SGD'),
            'val': training_config.get('val', True),
            'save_json': training_config.get('save_json', False),
            'freeze': training_config.get('freeze', False),
            'multi_scale': training_config.get('multi_scale', True),

            # 数据增强参数
            'hsv_h': aug_config.get('hsv_h', 0.015),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4),
            'degrees': aug_config.get('degrees', 10.0),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.5),
            'shear': aug_config.get('shear', 0.0),
            'perspective': aug_config.get('perspective', 0.0),
            'flipud': aug_config.get('flipud', 0.5),
            'fliplr': aug_config.get('fliplr', 0.5),
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.0),
            'copy_paste': aug_config.get('copy_paste', 0.0),
        }

        return args

    def train(self):
        """开始训练"""
        logger.info("开始YOLOv13训练...")

        # 注册清理处理器
        self.register_cleanup_handlers()

        # 获取模型名称和任务类型
        model_name = self.config['model']['name']
        task = self.config['training'].get('task', 'detect')

        # 智能处理模型配置文件路径
        # 如果模型名已包含-obb后缀，直接使用；否则根据task添加后缀
        if model_name.endswith('-obb'):
            model_yaml = f'{model_name}.yaml'
            logger.info(f"加载OBB模型配置: {model_yaml}")
        elif task == 'obb':
            model_yaml = f'{model_name}-obb.yaml'
            logger.info(f"加载OBB模型配置: {model_yaml}")
        else:
            model_yaml = f'{model_name}.yaml'
            logger.info(f"加载检测模型配置: {model_yaml}")

        # 加载模型
        model = YOLO(model_yaml)

        # 获取训练参数
        training_args = self.get_training_args()

        logger.info("训练参数:")
        for key, value in training_args.items():
            logger.info(f"  {key}: {value}")

        # 开始训练（Ultralytics会自动创建和管理文件夹）
        try:
            logger.info("🚀 开始训练，使用 Ctrl+C 可以中断训练")
            results = model.train(**training_args)

            logger.info("✅ 训练完成!")

            # 获取训练保存目录
            if results and hasattr(results, 'save_dir'):
                save_dir = results.save_dir
            else:
                save_dir = Path("runs/detect") / training_args['name']

            logger.info(f"最佳模型保存在: {save_dir / 'weights' / 'best.pt'}")

            # 训练完成后保存配置文件到Ultralytics创建的文件夹
            config_save_path = save_dir / "train_config.yaml"
            try:
                with open(config_save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                logger.info(f"训练配置已保存到: {config_save_path}")
            except Exception as e:
                logger.error(f"保存配置文件失败: {e}")

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
            logger.info("⏹️  训练被用户中断")
            logger.info("💾 Ultralytics会自动保存当前训练状态")
            logger.info("📁 训练日志和权重文件保存在: runs/detect/")
            # 尝试保存配置文件到可能已创建的训练目录
            save_dir = Path("runs/detect") / training_args['name']
            if save_dir.exists():
                config_save_path = save_dir / "train_config.yaml"
                try:
                    with open(config_save_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                    logger.info(f"训练配置已保存到: {config_save_path}")
                except Exception as e:
                    logger.error(f"保存配置文件失败: {e}")
            return None

        except Exception as e:
            logger.error(f"❌ 训练过程中出现错误: {e}")

            # 检查是否是CUDA OOM错误
            error_msg = str(e).lower()
            if 'out of memory' in error_msg or 'oom' in error_msg or 'cuda' in error_msg:
                logger.error("🔴 检测到CUDA OOM错误！")
                logger.error("💡 建议:")
                logger.error("  1. 减小batch_size")
                logger.error("  2. 降低image_size")
                logger.error("  3. 使用更小的模型 (yolov13s/l)")
                logger.error("  4. 设置cache=false")

                # 自动清理GPU进程
                self._cleanup_gpu_processes()

            raise

    def _cleanup_gpu_processes(self):
        """清理GPU上的训练进程"""
        if self._cleaned:
            return

        self._cleaned = True
        logger.info("🧹 开始清理GPU进程...")

        try:
            # 查找所有与当前训练相关的Python进程
            current_pid = os.getpid()

            # 使用nvidia-smi查找GPU上的Python进程
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                gpu_pids = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            gpu_pids.append(int(line.strip()))
                        except ValueError:
                            continue

                # 清理子进程（DDP创建的worker进程）
                import psutil
                current_process = psutil.Process(current_pid)
                children = current_process.children(recursive=True)

                if children:
                    logger.info(f"发现 {len(children)} 个子进程，正在清理...")
                    for child in children:
                        try:
                            logger.info(f"  终止子进程 {child.pid}")
                            child.terminate()
                        except Exception as e:
                            logger.warning(f"终止进程 {child.pid} 失败: {e}")

                    # 等待进程结束
                    import time
                    time.sleep(2)

                    # 强制杀死仍在运行的子进程
                    for child in children:
                        if child.is_running():
                            try:
                                logger.warning(f"  强制杀死进程 {child.pid}")
                                child.kill()
                            except Exception:
                                pass

                logger.info("✅ GPU进程清理完成")

        except FileNotFoundError:
            logger.warning("未找到nvidia-smi命令，跳过GPU清理")
        except Exception as e:
            logger.error(f"GPU清理失败: {e}")

    @classmethod
    def cleanup_on_exit(cls):
        """程序退出时的清理函数（atexit注册）"""
        if cls._cleaned:
            return

        logger.info("🧹 程序退出，执行清理...")

        try:
            # 清理PyTorch CUDA缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("✅ CUDA缓存已清理")
            except Exception:
                pass

        except Exception as e:
            logger.error(f"清理失败: {e}")

    def register_cleanup_handlers(self):
        """注册清理处理器"""
        if _training_process_info['cleanup_registered']:
            return

        # 注册atexit清理函数
        atexit.register(self.cleanup_on_exit)

        # 注册信号处理器
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，执行清理...")
            self._cleanup_gpu_processes()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        _training_process_info['cleanup_registered'] = True
        logger.info("✅ 已注册清理处理器")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv13训练框架')
    parser.add_argument('--config', type=str, default='train_yolov13_obb_8class.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='恢复训练')
    parser.add_argument('--device', type=str, default=None,
                       help='指定设备，如0,1,2或cpu')

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        return

    try:
        # 创建训练器
        trainer = YOLOv13Trainer(args.config)

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
