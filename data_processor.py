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


class YOLODatasetProcessor:
    """YOLO数据集处理器"""

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

        return config

    def print_config(self):
        """打印当前配置"""
        print("=" * 50)
        print("YOLO数据集配置")
        print("=" * 50)
        print(f"输入目录: {self.config['input_dir']}")
        print(f"图片子目录: {self.config.get('images_subdir', 'images')}")
        print(f"标签子目录: {self.config.get('labels_subdir', 'labels')}")
        print(f"输出目录: {self.config['output_dir']}")
        print(f"划分比例: {self.config['split_ratios']}")
        print(f"类别映射: {self.config['class_mapping']}")
        print(f"随机种子: {self.config.get('random_seed', 42)}")
        print(f"是否打乱: {self.config.get('shuffle', True)}")
        print("=" * 50)

    def validate_input_data(self) -> Tuple[List[str], List[str]]:
        """
        验证输入数据并返回有效的图片和标签文件列表

        Returns:
            Tuple[List[str], List[str]]: (有效图片列表, 有效标签列表)
        """
        logger.info("开始验证输入数据...")

        images_dir = self.input_path / self.config.get('images_subdir', 'images')
        labels_dir = self.input_path / self.config.get('labels_subdir', 'labels')

        # 检查目录是否存在
        if not images_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

        # 获取所有图片文件
        image_files = []
        for ext in self.config.get('image_extensions', ['.jpg', '.jpeg', '.png', '.bmp']):
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        # 获取所有标签文件
        label_files = list(labels_dir.glob("*.txt"))

        logger.info(f"找到 {len(image_files)} 个图片文件")
        logger.info(f"找到 {len(label_files)} 个标签文件")

        # 验证图片和标签文件的对应关系
        valid_images = []
        valid_labels = []

        for img_file in image_files:
            # 对应的标签文件名
            label_file = labels_dir / f"{img_file.stem}.txt"

            if label_file in label_files:
                valid_images.append(img_file)
                valid_labels.append(label_file)
            else:
                logger.warning(f"图片 {img_file.name} 缺少对应的标签文件，已跳过")

        logger.info(f"有效数据对: {len(valid_images)}")

        if len(valid_images) == 0:
            raise ValueError("没有找到有效的图片-标签文件对")

        return valid_images, valid_labels

    def get_split_names(self) -> List[str]:
        """获取数据集划分名称"""
        if len(self.config['split_ratios']) == 2:
            return ["train", "val"]
        else:
            return ["train", "val", "test"]

    def split_data(self, images: List[str], labels: List[str]) -> dict:
        """
        按照配置的比例分割数据

        Args:
            images: 图片文件列表
            labels: 标签文件列表

        Returns:
            dict: 分割后的数据字典
        """
        logger.info("开始分割数据...")

        # 创建索引列表
        indices = list(range(len(images)))

        # 打乱数据
        if self.config.get('shuffle', True):
            random.shuffle(indices)
            logger.info("数据已打乱")

        # 计算分割点
        total_count = len(indices)
        split_names = self.get_split_names()
        split_data = {}

        start_idx = 0
        for i, ratio in enumerate(self.config['split_ratios']):
            end_idx = start_idx + int(total_count * ratio)

            # 最后一个分割包含剩余所有数据
            if i == len(self.config['split_ratios']) - 1:
                end_idx = total_count

            split_indices = indices[start_idx:end_idx]
            split_name = split_names[i]

            split_images = [images[idx] for idx in split_indices]
            split_labels = [labels[idx] for idx in split_indices]

            split_data[split_name] = {
                'images': split_images,
                'labels': split_labels
            }

            logger.info(f"{split_name}: {len(split_images)} 张图片")
            start_idx = end_idx

        return split_data

    def create_output_structure(self):
        """创建输出目录结构"""
        logger.info("创建输出目录结构...")

        # 创建主输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 为每个数据集分割创建子目录
        split_names = self.get_split_names()
        for split_name in split_names:
            (self.output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
            (self.output_path / split_name / "labels").mkdir(parents=True, exist_ok=True)

        logger.info("输出目录结构创建完成")

    def copy_files(self, split_data: dict):
        """
        复制文件到输出目录

        Args:
            split_data: 分割后的数据字典
        """
        logger.info("开始复制文件...")

        total_files = 0
        for split_name, data in split_data.items():
            images_dir = self.output_path / split_name / "images"
            labels_dir = self.output_path / split_name / "labels"

            # 复制图片文件
            for img_file in data['images']:
                shutil.copy2(img_file, images_dir / img_file.name)

            # 复制标签文件
            for label_file in data['labels']:
                shutil.copy2(label_file, labels_dir / label_file.name)

            total_files += len(data['images'])
            logger.info(f"已复制 {len(data['images'])} 个文件到 {split_name} 目录")

        logger.info(f"文件复制完成，总计: {total_files} 个文件")

    def create_dataset_yaml(self):
        """创建YOLO数据集YAML配置文件"""
        logger.info("创建YOLO数据集配置文件...")

        class_mapping = self.config['class_mapping']
        # 确保类别映射的键是字符串格式
        string_mapping = {}
        for key, value in class_mapping.items():
            string_mapping[str(key)] = value

        yaml_content = {
            'path': str(self.output_path),
            'train': 'train',
            'val': 'val',
            'nc': len(class_mapping),
            'names': [string_mapping[str(i)] for i in sorted(map(int, string_mapping.keys()))]
        }

        # 如果有测试集，添加test字段
        if len(self.config['split_ratios']) == 3:
            yaml_content['test'] = 'test'

        yaml_path = self.output_path / self.config.get('output_yaml_name', 'dataset.yaml')

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False,
                     allow_unicode=True, indent=2)

        logger.info(f"YOLO配置文件已创建: {yaml_path}")

        # 打印配置内容
        print("\n生成的YOLO数据集配置:")
        print("-" * 40)
        for key, value in yaml_content.items():
            print(f"{key}: {value}")
        print("-" * 40)

    def generate_statistics(self, split_data: dict):
        """生成数据集统计信息"""
        logger.info("生成数据集统计信息...")

        stats_file = self.output_path / "dataset_statistics.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("YOLO数据集统计信息\n")
            f.write("=" * 50 + "\n\n")

            total_images = 0
            for split_name, data in split_data.items():
                count = len(data['images'])
                total_images += count
                f.write(f"{split_name.upper()}: {count} 张图片\n")

            f.write(f"\n总计: {total_images} 张图片\n")
            f.write(f"划分比例: {self.config['split_ratios']}\n")
            f.write(f"类别数量: {len(self.config['class_mapping'])}\n")

            f.write("\n类别映射:\n")
            for idx, class_name in sorted(self.config['class_mapping'].items()):
                f.write(f"  {idx}: {class_name}\n")

            f.write(f"\n随机种子: {self.config.get('random_seed', 42)}\n")
            f.write(f"输入目录: {self.input_path}\n")
            f.write(f"输出目录: {self.output_path}\n")

        logger.info(f"统计信息已保存到: {stats_file}")

    def process(self):
        """执行完整的数据处理流程"""
        try:
            logger.info("开始处理YOLO数据集...")

            # 打印配置信息
            self.print_config()

            # 1. 验证输入数据
            valid_images, valid_labels = self.validate_input_data()

            # 2. 分割数据
            split_data = self.split_data(valid_images, valid_labels)

            # 3. 创建输出目录结构
            self.create_output_structure()

            # 4. 复制文件
            self.copy_files(split_data)

            # 5. 创建YOLO配置文件
            self.create_dataset_yaml()

            # 6. 生成统计信息
            self.generate_statistics(split_data)

            logger.info("数据处理完成！")
            logger.info(f"处理后的数据集位于: {self.output_path}")

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
        processor = YOLODatasetProcessor(config_file)

        # 询问用户确认
        try:
            confirm = input("\n确认处理数据? (y/n): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return
        except EOFError:
            # 处理无法读取输入的情况（如在某些IDE中运行）
            print("\n自动跳过确认，开始处理数据...")

        processor.process()

        print("\n" + "=" * 60)
        print("数据处理完成!")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()