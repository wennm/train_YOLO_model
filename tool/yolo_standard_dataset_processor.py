import os
import shutil
import random
import yaml
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOStandardDatasetProcessor:
    """标准YOLO数据集格式处理器"""

    def __init__(self, config_path: str):
        """
        初始化数据处理器

        Args:
            config_path: YAML配置文件路径
        """
        self.config = self.load_config(config_path)
        # 处理路径字符串，移除可能的 r"" 前缀
        input_dir = self.config['input_dir']
        output_dir = self.config['output_dir']

        # 如果路径以 r" 开头且以 " 结尾，提取中间的路径
        if input_dir.startswith('r"') and input_dir.endswith('"'):
            input_dir = input_dir[2:-1]
        elif input_dir.startswith("r'") and input_dir.endswith("'"):
            input_dir = input_dir[2:-1]

        if output_dir.startswith('r"') and output_dir.endswith('"'):
            output_dir = output_dir[2:-1]
        elif output_dir.startswith("r'") and output_dir.endswith("'"):
            output_dir = output_dir[2:-1]

        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)

        # 设置随机种子
        random.seed(self.config.get('random_seed', 42))

    def load_config(self, config_path: str) -> Dict:
        """加载YAML配置文件"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证必要配置
        required_keys = ['input_dir', 'output_dir', 'class_mapping', 'split_ratios']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必要参数: {key}")

        # 设置默认值
        config.setdefault('images_subdir', 'images')
        config.setdefault('labels_subdir', 'labels')
        config.setdefault('output_yaml_name', 'dataset.yaml')
        config.setdefault('image_extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'])
        config.setdefault('random_seed', 42)
        config.setdefault('shuffle', True)
        config.setdefault('include_negative_samples', True)  # 是否包含负样本（无标签的图片）

        return config

    def print_config(self):
        """打印当前配置"""
        print("=" * 50)
        print("标准YOLO数据集配置")
        print("=" * 50)
        print(f"输入目录: {self.config['input_dir']}")
        print(f"图片子目录: {self.config.get('images_subdir', 'images')}")
        print(f"标签子目录: {self.config.get('labels_subdir', 'labels')}")
        print(f"输出目录: {self.config['output_dir']}")
        print(f"划分比例: {self.config['split_ratios']}")
        print(f"类别映射: {self.config['class_mapping']}")
        print(f"随机种子: {self.config.get('random_seed', 42)}")
        print(f"是否打乱: {self.config.get('shuffle', True)}")
        print(f"包含负样本: {self.config.get('include_negative_samples', True)}")
        print("=" * 50)

    def validate_input_data(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        验证输入数据并返回有效图片、有标签图片和无标签图片列表

        Returns:
            Tuple[List[Path], List[Path], List[Path]]: (所有图片列表, 有标签图片列表, 无标签图片列表)
        """
        logger.info("开始验证输入数据...")

        images_dir = self.input_path / self.config.get('images_subdir', 'images')
        labels_dir = self.input_path / self.config.get('labels_subdir', 'labels')

        # 检查目录是否存在
        if not images_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {images_dir}")

        # 标签目录可能不存在（当所有图片都是负样本时）
        if not labels_dir.exists():
            logger.warning(f"标签目录不存在: {labels_dir}，将创建空标签目录")
            labels_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图片文件（避免重复统计）
        image_files = []
        for ext in self.config.get('image_extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']):
            # 搜索大小写不敏感的扩展名
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        # 去重
        image_files = list(set(image_files))

        # 获取所有标签文件
        label_files = list(labels_dir.glob("*.txt"))

        logger.info(f"找到 {len(image_files)} 个图片文件")
        logger.info(f"找到 {len(label_files)} 个标签文件")

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

        logger.info(f"有标签图片: {len(labeled_images)}")
        logger.info(f"无标签图片: {len(unlabeled_images)}")

        if len(labeled_images) == 0 and len(unlabeled_images) == 0:
            raise ValueError("没有找到任何图片文件")

        all_images = labeled_images + unlabeled_images
        return all_images, labeled_images, unlabeled_images

    def get_split_names(self) -> List[str]:
        """获取数据集划分名称"""
        if len(self.config['split_ratios']) == 2:
            return ["train", "val"]
        else:
            return ["train", "val", "test"]

    def split_data(self, all_images: List[Path], labeled_images: List[Path], unlabeled_images: List[Path]) -> dict:
        """
        按照配置的比例分割数据，保持正负样本比例

        Args:
            all_images: 所有图片文件列表
            labeled_images: 有标签图片文件列表
            unlabeled_images: 无标签图片文件列表

        Returns:
            dict: 分割后的数据字典
        """
        logger.info("开始分割数据...")

        # 分别打乱有标签和无标签图片
        if self.config.get('shuffle', True):
            random.shuffle(labeled_images)
            random.shuffle(unlabeled_images)
            logger.info("数据已打乱")

        split_names = self.get_split_names()
        split_data = {}

        total_count = len(all_images)
        total_labeled = len(labeled_images)
        total_unlabeled = len(unlabeled_images)

        logger.info(f"总样本数: {total_count}, 正样本: {total_labeled}, 负样本: {total_unlabeled}")

        # 计算每个分割应该分配的样本数
        split_counts = []
        remaining_count = total_count

        for i, ratio in enumerate(self.config['split_ratios']):
            if i == len(self.config['split_ratios']) - 1:
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

    def create_output_structure(self):
        """创建标准YOLO输出目录结构"""
        logger.info("创建标准YOLO输出目录结构...")

        # 创建主输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 创建标准YOLO格式目录
        (self.output_path / "images").mkdir(exist_ok=True)
        (self.output_path / "labels").mkdir(exist_ok=True)

        # 为每个数据集分割创建子目录
        split_names = self.get_split_names()
        for split_name in split_names:
            (self.output_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (self.output_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        logger.info("标准YOLO输出目录结构创建完成")

    def copy_files(self, split_data: dict):
        """
        复制文件到标准YOLO格式输出目录

        Args:
            split_data: 分割后的数据字典
        """
        logger.info("开始复制文件...")

        total_files = 0
        for split_name, data in split_data.items():
            images_dir = self.output_path / "images" / split_name
            labels_dir = self.output_path / "labels" / split_name

            # 复制图片文件
            for img_file in data['images']:
                shutil.copy2(img_file, images_dir / img_file.name)

            # 处理标签文件
            labels_dir_input = self.input_path / self.config.get('labels_subdir', 'labels')

            # 复制或创建标签文件
            for img_file in data['images']:
                label_file = labels_dir_input / f"{img_file.stem}.txt"
                output_label_file = labels_dir / f"{img_file.stem}.txt"

                if img_file in data['labeled_images'] and label_file.exists():
                    # 复制现有标签文件
                    shutil.copy2(label_file, output_label_file)
                elif img_file in data['unlabeled_images']:
                    # 为负样本创建空标签文件
                    output_label_file.touch()
                    logger.debug(f"为负样本创建空标签文件: {output_label_file}")

            total_files += len(data['images'])
            logger.info(f"已复制 {len(data['images'])} 个文件到 {split_name} 目录")

        logger.info(f"文件复制完成，总计: {total_files} 个文件")

    def create_dataset_yaml(self):
        """创建标准YOLO数据集YAML配置文件"""
        logger.info("创建标准YOLO数据集配置文件...")

        class_mapping = self.config['class_mapping']
        # 确保类别映射的键是字符串格式
        string_mapping = {}
        for key, value in class_mapping.items():
            string_mapping[str(key)] = value

        yaml_content = {
            'path': str(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_mapping),
            'names': [string_mapping[str(i)] for i in sorted(map(int, string_mapping.keys()))]
        }

        # 如果有测试集，添加test字段
        if len(self.config['split_ratios']) == 3:
            yaml_content['test'] = 'images/test'

        yaml_path = self.output_path / self.config.get('output_yaml_name', 'dataset.yaml')

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False,
                     allow_unicode=True, indent=2)

        logger.info(f"标准YOLO配置文件已创建: {yaml_path}")

        # 打印配置内容
        print("\n生成的标准YOLO数据集配置:")
        print("-" * 40)
        for key, value in yaml_content.items():
            print(f"{key}: {value}")
        print("-" * 40)

    def generate_statistics(self, split_data: dict):
        """生成数据集统计信息"""
        logger.info("生成数据集统计信息...")

        stats_file = self.output_path / "dataset_statistics.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("标准YOLO数据集统计信息\n")
            f.write("=" * 50 + "\n\n")

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
                f.write(f"  总图片数: {count}\n")
                f.write(f"  正样本数: {labeled_count}\n")
                f.write(f"  负样本数: {unlabeled_count}\n\n")

            f.write(f"总计: {total_images} 张图片\n")
            f.write(f"正样本总计: {total_labeled} 张\n")
            f.write(f"负样本总计: {total_unlabeled} 张\n")
            f.write(f"划分比例: {self.config['split_ratios']}\n")
            f.write(f"类别数量: {len(self.config['class_mapping'])}\n")

            f.write("\n类别映射:\n")
            for idx, class_name in sorted(self.config['class_mapping'].items()):
                f.write(f"  {idx}: {class_name}\n")

            f.write(f"\n随机种子: {self.config.get('random_seed', 42)}\n")
            f.write(f"输入目录: {self.input_path}\n")
            f.write(f"输出目录: {self.output_path}\n")
            f.write(f"数据格式: 标准YOLO格式\n")

        logger.info(f"统计信息已保存到: {stats_file}")

    def process(self):
        """执行完整的数据处理流程"""
        try:
            logger.info("开始处理标准YOLO数据集...")

            # 打印配置信息
            self.print_config()

            # 1. 验证输入数据
            all_images, labeled_images, unlabeled_images = self.validate_input_data()

            # 2. 分割数据
            split_data = self.split_data(all_images, labeled_images, unlabeled_images)

            # 3. 创建输出目录结构
            self.create_output_structure()

            # 4. 复制文件
            self.copy_files(split_data)

            # 5. 创建YOLO配置文件
            self.create_dataset_yaml()

            # 6. 生成统计信息
            self.generate_statistics(split_data)

            logger.info("标准YOLO数据处理完成！")
            logger.info(f"处理后的数据集位于: {self.output_path}")
            logger.info("数据集结构:")
            logger.info("  yolo_dataset/")
            logger.info("  ├── images/")
            logger.info("  │   ├── train/")
            logger.info("  │   ├── val/")
            logger.info("  │   └── test/          # 可选")
            logger.info("  └── labels/")
            logger.info("      ├── train/")
            logger.info("      ├── val/")
            logger.info("      └── test/          # 可选")

        except Exception as e:
            logger.error(f"数据处理过程中发生错误: {str(e)}")
            raise


def main():
    """主函数 - 读取dataset_config.yaml配置文件"""
    config_file = r"F:\wenw\work\train_YOLO_model\dataset_config.yaml"

    try:
        # 检查配置文件是否存在
        if not Path(config_file).exists():
            print(f"错误: 配置文件 '{config_file}' 不存在!")
            print("请确保在当前目录下有 'dataset_config.yaml' 文件")
            sys.exit(1)

        # 创建处理器并执行处理
        processor = YOLOStandardDatasetProcessor(config_file)

        # 询问用户确认
        try:
            confirm = input("\n确认处理数据为标准YOLO格式? (y/n): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return
        except EOFError:
            # 处理无法读取输入的情况（如在某些IDE中运行）
            print("\n自动跳过确认，开始处理数据...")

        processor.process()

        print("\n" + "=" * 60)
        print("标准YOLO数据处理完成!")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()