import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetSupplementer:
    """YOLO数据集补充工具"""

    def __init__(self,
                 existing_dataset_dir: str,
                 new_data_dir: str,
                 supplement_ratios: List[float],
                 random_seed: int = 42,
                 shuffle: bool = True):
        """
        初始化数据集补充工具

        Args:
            existing_dataset_dir: 现有数据集目录
            new_data_dir: 新数据目录（包含images和labels子文件夹）
            supplement_ratios: 补充数据分配比例，如 [0.8, 0.2] 表示80%到训练集，20%到验证集
            random_seed: 随机种子
            shuffle: 是否随机打乱新数据
        """
        self.existing_path = Path(existing_dataset_dir)
        self.new_data_path = Path(new_data_dir)
        self.supplement_ratios = supplement_ratios
        self.random_seed = random_seed
        self.shuffle = shuffle

        # 设置随机种子
        random.seed(self.random_seed)

        # 验证目录存在
        if not self.existing_path.exists():
            raise FileNotFoundError(f"现有数据集目录不存在: {existing_dataset_dir}")
        if not self.new_data_path.exists():
            raise FileNotFoundError(f"新数据目录不存在: {new_data_dir}")

        # 读取现有数据集的配置
        self.dataset_config = self._load_dataset_config()

        # 检测数据集结构类型
        self.dataset_type = self._detect_dataset_type()
        logger.info(f"检测到数据集结构类型: {self.dataset_type}")

    def _load_dataset_config(self) -> Dict:
        """读取现有数据集的配置文件"""
        yaml_path = self.existing_path / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"找不到数据集配置文件: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info(f"成功加载数据集配置: {yaml_path}")
        logger.info(f"类别数量: {config.get('nc')}")
        logger.info(f"类别名称: {config.get('names')}")
        logger.info(f"训练集路径: {config.get('train')}")
        logger.info(f"验证集路径: {config.get('val')}")

        return config

    def _detect_dataset_type(self) -> str:
        """
        检测数据集结构类型
        Returns: 'nested' (train/images, val/images) 或 'standard' (images/train, images/val)
        """
        config = self.dataset_config

        # 从yaml配置中读取路径
        train_path = config.get('train', '')
        val_path = config.get('val', '')

        logger.info(f"从配置文件读取的train路径: {train_path}")
        logger.info(f"从配置文件读取的val路径: {val_path}")

        # 判断路径格式
        # 如果路径是 'train/images' 或 'valid/images'，说明是 nested 结构
        # 如果路径是 'images/train' 或 'images/val'，说明是 standard 结构
        if train_path.startswith('train/') or train_path.startswith('valid/'):
            logger.info("检测到 nested 结构: train/images, valid/images")
            return 'nested'
        elif train_path.startswith('images/'):
            logger.info("检测到 standard 结构: images/train, images/val")
            return 'standard'
        else:
            # 尝试从实际目录结构判断
            train_dir = self.existing_path / "train" / "images"
            images_train_dir = self.existing_path / "images" / "train"

            if train_dir.exists():
                logger.info("通过目录结构检测到 nested 结构")
                return 'nested'
            elif images_train_dir.exists():
                logger.info("通过目录结构检测到 standard 结构")
                return 'standard'
            else:
                logger.warning("无法自动检测数据集结构，默认使用 nested 结构")
                return 'nested'

    def validate_new_data(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        验证新数据并返回有效图片、有标签图片和无标签图片列表

        Returns:
            Tuple[List[Path], List[Path], List[Path]]: (所有图片列表, 有标签图片列表, 无标签图片列表)
        """
        logger.info("开始验证新数据...")

        images_dir = self.new_data_path / "images"
        labels_dir = self.new_data_path / "labels"

        # 检查目录是否存在
        if not images_dir.exists():
            raise FileNotFoundError(f"新数据图片目录不存在: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"新数据标签目录不存在: {labels_dir}")

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        # 去重
        image_files = list(set(image_files))

        # 获取所有标签文件
        label_files = list(labels_dir.glob("*.txt"))

        logger.info(f"找到 {len(image_files)} 个新图片文件")
        logger.info(f"找到 {len(label_files)} 个新标签文件")

        # 分类图片：有标签和无标签
        labeled_images = []
        unlabeled_images = []

        # 创建标签文件名集合以便快速查找
        label_filenames = {label_file.stem for label_file in label_files}

        for img_file in image_files:
            if img_file.stem in label_filenames:
                # 查找对应的标签文件
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    labeled_images.append(img_file)
            else:
                unlabeled_images.append(img_file)

        logger.info(f"有标签新图片: {len(labeled_images)}")
        logger.info(f"无标签新图片: {len(unlabeled_images)}")

        if len(image_files) == 0:
            raise ValueError("没有找到任何新图片文件")

        all_images = labeled_images + unlabeled_images
        return all_images, labeled_images, unlabeled_images

    def split_new_data(self, all_images: List[Path], labeled_images: List[Path],
                      unlabeled_images: List[Path]) -> Dict:
        """
        按照配置的比例分割新数据

        Args:
            all_images: 所有新图片文件列表
            labeled_images: 有标签新图片文件列表
            unlabeled_images: 无标签新图片文件列表

        Returns:
            Dict: 分割后的数据字典
        """
        logger.info("开始分割新数据...")

        # 根据配置决定是否打乱有标签和无标签图片
        if self.shuffle:
            random.shuffle(labeled_images)
            random.shuffle(unlabeled_images)
            logger.info("新数据已打乱")
        else:
            logger.info("保持新数据原始顺序（不打乱）")

        # 根据比例确定分割名称
        if len(self.supplement_ratios) == 2:
            split_names = ["train", "val"]
        elif len(self.supplement_ratios) == 3:
            split_names = ["train", "val", "test"]
        else:
            raise ValueError("supplement_ratios 必须是2个或3个值")

        split_data = {}

        total_count = len(all_images)
        total_labeled = len(labeled_images)
        total_unlabeled = len(unlabeled_images)

        logger.info(f"新数据总样本数: {total_count}, 正样本: {total_labeled}, 负样本: {total_unlabeled}")

        # 计算每个分割应该分配的样本数
        split_counts = []
        remaining_count = total_count

        for i, ratio in enumerate(self.supplement_ratios):
            if i == len(self.supplement_ratios) - 1:
                # 最后一个分割分配所有剩余样本
                count = remaining_count
            else:
                count = int(total_count * ratio)
                remaining_count -= count
            split_counts.append(count)

        logger.info(f"各分割样本数: {dict(zip(split_names, split_counts))}")

        # 为每个分割分配正负样本
        labeled_assigned = 0
        unlabeled_assigned = 0

        for i, (split_name, split_count) in enumerate(zip(split_names, split_counts)):
            if i == len(split_names) - 1:
                # 最后一个分割分配所有剩余样本
                split_labeled_images = labeled_images[labeled_assigned:]
                split_unlabeled_images = unlabeled_images[unlabeled_assigned:]
            else:
                # 计算该分割应该分配的正负样本数
                positive_ratio = total_labeled / total_count if total_count > 0 else 0

                target_labeled = int(split_count * positive_ratio)
                target_unlabeled = split_count - target_labeled

                # 确保不超出可用范围
                available_labeled = total_labeled - labeled_assigned
                available_unlabeled = total_unlabeled - unlabeled_assigned

                actual_labeled = min(target_labeled, available_labeled)
                actual_unlabeled = min(target_unlabeled, available_unlabeled)

                # 如果正样本不够，用负样本补充
                if actual_labeled < target_labeled:
                    extra_needed = target_labeled - actual_labeled
                    if available_unlabeled >= extra_needed:
                        actual_unlabeled += extra_needed
                    else:
                        actual_unlabeled += available_unlabeled

                # 获取对应范围的样本
                split_labeled_images = labeled_images[labeled_assigned:labeled_assigned + actual_labeled]
                split_unlabeled_images = unlabeled_images[unlabeled_assigned:unlabeled_assigned + actual_unlabeled]

                labeled_assigned += actual_labeled
                unlabeled_assigned += actual_unlabeled

            split_images = split_labeled_images + split_unlabeled_images

            split_data[split_name] = {
                'images': split_images,
                'labeled_images': split_labeled_images,
                'unlabeled_images': split_unlabeled_images
            }

            logger.info(f"{split_name}: {len(split_images)} 张图片 (正样本: {len(split_labeled_images)}, 负样本: {len(split_unlabeled_images)})")

        return split_data

    def copy_files(self, split_data: Dict):
        """
        复制新文件到现有数据集目录

        Args:
            split_data: 分割后的数据字典
        """
        logger.info("开始复制文件到现有数据集...")

        total_files = 0
        for split_name, data in split_data.items():
            # 根据数据集类型确定目标路径
            # 注意：需要处理 val/valid 的命名差异
            if self.dataset_type == 'nested':
                # nested 结构: train/images, valid/images
                # 需要检查 yaml 配置来确定实际目录名
                yaml_split_path = self.dataset_config.get(split_name, '')
                if yaml_split_path.startswith('valid/'):
                    actual_split_name = 'valid'
                else:
                    actual_split_name = split_name

                images_dir = self.existing_path / actual_split_name / "images"
                labels_dir = self.existing_path / actual_split_name / "labels"
            else:  # standard
                # standard 结构: images/train, labels/train
                images_dir = self.existing_path / "images" / split_name
                labels_dir = self.existing_path / "labels" / split_name

            logger.info(f"目标图片目录: {images_dir}")
            logger.info(f"目标标签目录: {labels_dir}")

            # 检查目标目录是否存在
            if not images_dir.exists():
                raise FileNotFoundError(f"目标图片目录不存在: {images_dir}")
            if not labels_dir.exists():
                raise FileNotFoundError(f"目标标签目录不存在: {labels_dir}")

            # 新数据的标签目录
            labels_dir_input = self.new_data_path / "labels"

            # 复制图片和标签文件
            for img_file in data['images']:
                # 检查文件是否已存在
                target_img = images_dir / img_file.name
                target_label = labels_dir / f"{img_file.stem}.txt"

                if target_img.exists():
                    logger.warning(f"图片 {img_file.name} 已存在于 {split_name}，跳过")
                    continue

                # 复制图片
                shutil.copy2(img_file, target_img)
                logger.debug(f"复制图片: {img_file.name} -> {split_name}")

                # 处理标签文件
                if img_file in data['labeled_images']:
                    label_file = labels_dir_input / f"{img_file.stem}.txt"
                    if label_file.exists():
                        shutil.copy2(label_file, target_label)
                        logger.debug(f"复制标签: {img_file.stem}.txt -> {split_name}")
                    else:
                        # 如果标签文件不存在，创建空标签
                        target_label.touch()
                        logger.warning(f"标签文件 {img_file.stem}.txt 不存在，创建空标签")
                else:
                    # 为负样本创建空标签
                    target_label.touch()
                    logger.debug(f"为负样本创建空标签: {img_file.stem}.txt")

                total_files += 1

            logger.info(f"已复制 {len(data['images'])} 个文件到 {split_name} 目录")

        logger.info(f"文件复制完成，总计: {total_files} 个新文件")

    def generate_statistics(self, split_data: Dict):
        """生成补充数据统计信息"""
        logger.info("生成补充数据统计信息...")

        stats_file = self.existing_path / "supplement_statistics.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("数据集补充统计信息\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"新数据来源: {self.new_data_path}\n")
            f.write(f"数据集结构类型: {self.dataset_type}\n")
            f.write(f"补充比例: {self.supplement_ratios}\n")
            f.write(f"是否打乱数据: {self.shuffle}\n")
            f.write(f"随机种子: {self.random_seed}\n\n")

            total_images = 0
            total_labeled = 0
            total_unlabeled = 0

            for split_name, data in split_data.items():
                count = len(data['images'])
                labeled_count = len(data['labeled_images'])
                unlabeled_count = len(data['unlabeled_images'])

                total_images += count
                total_labeled += labeled_count
                total_unlabeled += unlabeled_count

                f.write(f"{split_name.upper()}:\n")
                f.write(f"  新增图片数: {count}\n")
                f.write(f"  新增正样本数: {labeled_count}\n")
                f.write(f"  新增负样本数: {unlabeled_count}\n\n")

            f.write(f"总计新增: {total_images} 张图片\n")
            f.write(f"正样本新增: {total_labeled} 张\n")
            f.write(f"负样本新增: {total_unlabeled} 张\n")

        logger.info(f"补充统计信息已保存到: {stats_file}")

        # 同时更新总体统计信息
        self._update_overall_statistics()

    def _update_overall_statistics(self):
        """更新总体数据集统计信息"""
        logger.info("更新总体数据集统计信息...")

        # 根据yaml配置获取可用的数据集分割
        split_names = ["train", "val"]
        if self.dataset_config.get('test'):
            split_names.append("test")

        total_counts = {}
        labeled_counts = {}
        unlabeled_counts = {}

        for split_name in split_names:
            # 根据数据集类型确定路径
            # 注意：yaml中的split_name可能是 'valid' 而不是 'val'
            yaml_split_name = self.dataset_config.get(split_name, '')

            if self.dataset_type == 'nested':
                # nested 结构: train/images, valid/images
                # 需要处理 val/valid 的命名差异
                if yaml_split_name.startswith('valid/'):
                    actual_split_name = 'valid'
                else:
                    actual_split_name = split_name
                images_dir = self.existing_path / actual_split_name / "images"
                labels_dir = self.existing_path / actual_split_name / "labels"
            else:  # standard
                images_dir = self.existing_path / "images" / split_name
                labels_dir = self.existing_path / "labels" / split_name

            logger.info(f"统计 {split_name} 目录: {images_dir}")

            if not images_dir.exists():
                logger.warning(f"目录不存在，跳过: {images_dir}")
                continue

            # 统计图片数量
            image_files = list(images_dir.glob("*.*"))
            image_files = [f for f in image_files if f.suffix.lower() in
                          ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']]

            # 统计标签数量（非空标签）
            labeled_count = 0
            for label_file in labels_dir.glob("*.txt"):
                if label_file.stat().st_size > 0:  # 文件大小大于0表示有内容
                    labeled_count += 1

            total_counts[split_name] = len(image_files)
            labeled_counts[split_name] = labeled_count
            unlabeled_counts[split_name] = len(image_files) - labeled_count

        # 更新统计文件
        stats_file = self.existing_path / "dataset_statistics.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("YOLO数据集统计信息（更新后）\n")
            f.write("=" * 50 + "\n\n")

            total_images = 0
            total_labeled = 0
            total_unlabeled = 0

            for split_name in split_names:
                if split_name not in total_counts:
                    continue

                count = total_counts[split_name]
                labeled_count = labeled_counts[split_name]
                unlabeled_count = unlabeled_counts[split_name]

                total_images += count
                total_labeled += labeled_count
                total_unlabeled += unlabeled_count

                f.write(f"{split_name.upper()}:\n")
                f.write(f"  总图片数: {count}\n")
                f.write(f"  正样本数: {labeled_count}\n")
                f.write(f"  负样本数: {unlabeled_count}\n\n")

            f.write(f"总计: {total_images} 张图片\n")
            f.write(f"正样本总计: {total_labeled} 张\n")
            f.write(f"负样本总计: {total_unlabeled} 张\n")

            f.write("\n类别映射:\n")
            for idx, class_name in enumerate(self.dataset_config.get('names', [])):
                f.write(f"  {idx}: {class_name}\n")

        logger.info(f"总体统计信息已更新: {stats_file}")

    def print_config(self):
        """打印当前配置"""
        print("\n" + "=" * 50)
        print("数据集补充配置")
        print("=" * 50)
        print(f"现有数据集目录: {self.existing_path}")
        print(f"新数据目录: {self.new_data_path}")
        print(f"数据集结构类型: {self.dataset_type}")
        print(f"补充比例: {self.supplement_ratios}")
        print(f"是否打乱数据: {self.shuffle}")
        print(f"随机种子: {self.random_seed}")
        print("=" * 50)

    def process(self):
        """执行完整的数据补充流程"""
        try:
            logger.info("开始补充数据集...")

            # 打印配置信息
            self.print_config()

            # 1. 验证新数据
            all_images, labeled_images, unlabeled_images = self.validate_new_data()

            # 2. 分割新数据
            split_data = self.split_new_data(all_images, labeled_images, unlabeled_images)

            # 3. 复制文件
            self.copy_files(split_data)

            # 4. 生成统计信息
            self.generate_statistics(split_data)

            logger.info("数据集补充完成！")
            logger.info(f"更新后的数据集位于: {self.existing_path}")

        except Exception as e:
            logger.error(f"数据集补充过程中发生错误: {str(e)}")
            raise


def main():
    """主函数"""
    # 配置参数
    # EXISTING_DATASET_DIR = r"F:\wenw\work\dataset\infrared_1124_onlymotorcycle_processor"
    # NEW_DATA_DIR = r"F:\wenw\work\dataset\buchong\1230_onlymotorcycle"
    EXISTING_DATASET_DIR = r"F:\wenw\work\dataset\dataset_no_game_4class_1212"
    NEW_DATA_DIR = r"F:\wenw\work\dataset\buchong\traffic_accident1229"

    # 补充数据分配比例
    # [训练集比例, 验证集比例] 或 [训练集比例, 验证集比例, 测试集比例]
    # 例如：[0.8, 0.2] 表示 80% 到训练集，20% 到验证集
    SUPPLEMENT_RATIOS = [0.9, 0.1]

    RANDOM_SEED = 42

    # 是否随机打乱新数据
    # True: 随机打乱新数据后再分配
    # False: 保持新数据的原始顺序分配
    SHUFFLE = False

    try:
        # 创建补充器并执行补充
        supplementer = DatasetSupplementer(
            existing_dataset_dir=EXISTING_DATASET_DIR,
            new_data_dir=NEW_DATA_DIR,
            supplement_ratios=SUPPLEMENT_RATIOS,
            random_seed=RANDOM_SEED,
            shuffle=SHUFFLE
        )

        # 询问用户确认
        print("\n" + "=" * 60)
        print("警告：此操作将向现有数据集添加新数据")
        print("=" * 60)
        confirm = input("\n确认补充数据到现有数据集? (y/n): ").strip().lower()
        if confirm != 'y':
            print("操作已取消")
            return

        supplementer.process()

        print("\n" + "=" * 60)
        print("数据集补充完成!")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
