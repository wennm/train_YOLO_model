#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO数据集标签统计工具
自动识别数据集结构，统计各类别标签的数量和比例
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict


class YOLODatasetAnalyzer:
    """YOLO数据集标签分析器"""

    def __init__(self, dataset_path: str):
        """
        初始化分析器

        Args:
            dataset_path: 数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)

        # 自动检测数据集结构
        self.dataset_type = self._detect_dataset_type()

        # 读取类别信息
        self.class_names = self._load_class_names()

    def _detect_dataset_type(self) -> str:
        """
        检测数据集结构类型
        Returns: 'nested' (train/val/test 直接在根目录，各自包含images/labels) 或
                'standard' (images/labels 在根目录，下面有train/val/test子目录)
        """
        # 检查第一种结构：nested (train/, val/, test/ 直接在根目录)
        train_dir = self.dataset_path / "train"
        val_dir = self.dataset_path / "val"

        # 检查第二种结构：standard (images/, labels/ 在根目录)
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        # 第一种结构判断：根目录下有 train/val 目录且这些目录下有 images 子目录
        if (train_dir.exists() and val_dir.exists() and
            (train_dir / "images").exists() and (val_dir / "images").exists()):
            print("[INFO] 检测到第一种结构：train/val/test 目录在根目录")
            return "nested"

        # 第二种结构判断：根目录下有 images/labels 目录
        elif (images_dir.exists() and labels_dir.exists() and
              (images_dir / "train").exists() and (labels_dir / "train").exists()):
            print("[INFO] 检测到第二种结构：images/labels 目录在根目录，下面有 train/val/test")
            return "standard"

        # 容错处理：只有一种结构的部分特征
        elif train_dir.exists() and val_dir.exists():
            print("[INFO] 检测到第一种结构（部分特征）")
            return "nested"
        elif images_dir.exists() and labels_dir.exists():
            print("[INFO] 检测到第二种结构（部分特征）")
            return "standard"
        else:
            print(f"[ERROR] 无法识别的数据集结构: {self.dataset_path}")
            print(f"[ERROR] 请确保数据集符合以下两种结构之一：")
            print("  第一种结构：dataset/train/images/, dataset/val/images/, dataset/test/images/")
            print("  第二种结构：dataset/images/train/, dataset/labels/train/, dataset/images/val/, dataset/labels/val/")
            raise ValueError(f"无法识别的数据集结构: {self.dataset_path}")

    def _load_class_names(self) -> List[str]:
        """
        从数据集配置文件加载类别名称
        Returns: 类别名称列表
        """
        # 尝试找到配置文件
        yaml_files = list(self.dataset_path.glob("*.yaml")) + list(self.dataset_path.glob("*.yml"))

        if yaml_files:
            try:
                import yaml
                with open(yaml_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                if 'names' in config:
                    if isinstance(config['names'], list):
                        return config['names']
                    elif isinstance(config['names'], dict):
                        # 按索引排序
                        return [config['names'][i] for i in sorted(config['names'].keys())]
            except Exception as e:
                print(f"[WARNING] 无法解析配置文件 {yaml_files[0]}: {e}")

        # 默认类别名称
        print("[INFO] 使用默认类别名称: class_0, class_1, ...")
        return [f'class_{i}' for i in range(10)]  # 预设10个类别

    def _get_dataset_splits(self) -> List[str]:
        """
        获取数据集的分割列表
        Returns: 包含 'train', 'val', 'test' 的列表（如果存在）
        """
        splits = []

        if self.dataset_type == "nested":
            # 第一种结构：检查 train/val/test 目录
            for split in ['train', 'val', 'test']:
                split_dir = self.dataset_path / split
                if split_dir.exists() and (split_dir / "images").exists():
                    splits.append(split)
        else:  # standard
            # 第二种结构：检查 images/train, images/val, images/test
            images_dir = self.dataset_path / "images"
            labels_dir = self.dataset_path / "labels"
            for split in ['train', 'val', 'test']:
                if ((images_dir / split).exists() and
                    (labels_dir / split).exists()):
                    splits.append(split)

        return splits

    def analyze_split(self, split_name: str) -> Dict[int, int]:
        """
        分析某个数据集分割的标签统计

        Args:
            split_name: 分割名称（train、val 或 test）

        Returns:
            类别ID到标签数量的映射字典
        """
        print(f"\n[INFO] 分析 {split_name} 数据集...")

        # 根据数据集类型设置路径
        if self.dataset_type == "nested":
            # 第一种结构：train/val/test 直接在根目录，各自包含 images/labels
            labels_dir = self.dataset_path / split_name / "labels"
        else:  # standard
            # 第二种结构：所有标签在 labels/train, labels/val, labels/test
            labels_dir = self.dataset_path / "labels" / split_name

        # 验证输入路径存在性
        if not labels_dir.exists():
            print(f"[WARNING] 标签目录不存在: {labels_dir}")
            return {}

        # 获取所有标签文件
        label_files = list(labels_dir.glob("*.txt"))

        print(f"[INFO] 找到 {len(label_files)} 个标签文件")

        if len(label_files) == 0:
            print(f"[WARNING] 在 {labels_dir} 中没有找到标签文件")
            return {}

        # 统计每个类别的标签数量
        label_counts = defaultdict(int)
        empty_labels = 0  # 空标签文件数量（负样本）
        total_labels = 0  # 总标签数

        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 检查是否为空标签文件
                if not lines or all(line.strip() == '' for line in lines):
                    empty_labels += 1
                    continue

                # 统计每个标签
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        parts = line.split()
                        class_id = int(parts[0])
                        label_counts[class_id] += 1
                        total_labels += 1
                    except (IndexError, ValueError) as e:
                        print(f"[WARNING] 解析标签失败 {label_file}: {line} - {e}")
                        continue

            except Exception as e:
                print(f"[ERROR] 读取标签文件失败 {label_file}: {e}")
                continue

        # 输出统计结果
        print(f"\n{'='*60}")
        print(f"{split_name.upper()} 数据集标签统计")
        print(f"{'='*60}")
        print(f"总文件数: {len(label_files)}")
        print(f"有标签文件数: {len(label_files) - empty_labels}")
        print(f"空标签文件数（负样本）: {empty_labels}")
        print(f"总标签数: {total_labels}")
        print(f"\n各类别标签统计:")

        if total_labels > 0:
            # 按类别ID排序输出
            for class_id in sorted(label_counts.keys()):
                count = label_counts[class_id]
                percentage = (count / total_labels) * 100
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                print(f"  类别 {class_id} ({class_name}): {count} 个标签, 占比 {percentage:.2f}%")
        else:
            print("  没有找到有效的标签")

        print(f"{'='*60}")

        return dict(label_counts)

    def analyze_dataset(self):
        """分析整个数据集"""
        print("[INFO] 开始分析YOLO数据集标签...")
        print(f"[INFO] 数据集路径: {self.dataset_path}")
        print(f"[INFO] 数据集结构类型: {self.dataset_type}")
        print(f"[INFO] 检测到的类别: {self.class_names}")

        # 显示检测到的数据集结构
        if self.dataset_type == "nested":
            print("[INFO] 数据集结构:")
            print("  dataset/")
            print("    ├── train/images/")
            print("    ├── train/labels/")
            print("    ├── val/images/")
            print("    ├── val/labels/")
            print("    └── test/(optional)")
        else:  # standard
            print("[INFO] 数据集结构:")
            print("  dataset/")
            print("    ├── images/train/")
            print("    ├── labels/train/")
            print("    ├── images/val/")
            print("    ├── labels/val/")
            print("    └── images/test/, labels/test/(optional)")

        # 获取所有可用的数据集分割
        splits = self._get_dataset_splits()
        print(f"\n[INFO] 检测到数据集分割: {splits}")

        if not splits:
            print("[ERROR] 未找到有效的数据集分割！")
            print("[ERROR] 请检查数据集结构是否正确")
            return

        # 统计所有分割的数据
        all_stats = {}
        total_stats = defaultdict(int)

        for split in splits:
            stats = self.analyze_split(split)
            all_stats[split] = stats

            # 累计到总体统计
            for class_id, count in stats.items():
                total_stats[class_id] += count

        # 输出总体统计
        print(f"\n{'='*60}")
        print("全数据集汇总统计")
        print(f"{'='*60}")

        total_labels = sum(total_stats.values())
        print(f"总标签数: {total_labels}")

        if total_labels > 0:
            print(f"\n各类别标签汇总:")
            # 按类别ID排序输出
            for class_id in sorted(total_stats.keys()):
                count = total_stats[class_id]
                percentage = (count / total_labels) * 100
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                print(f"  类别 {class_id} ({class_name}): {count} 个标签, 占比 {percentage:.2f}%")

        # 输出各分割的对比
        print(f"\n{'='*60}")
        print("各数据集分割对比")
        print(f"{'='*60}")

        # 找出所有出现的类别
        all_class_ids = set()
        for stats in all_stats.values():
            all_class_ids.update(stats.keys())

        if all_class_ids:
            # 表头
            header = f"{'类别ID':<8} {'类别名称':<20}"
            for split in splits:
                header += f" {split.upper():<12}"
            print(header)

            # 分隔线
            separator = "-" * 60
            print(separator)

            # 每个类别一行
            for class_id in sorted(all_class_ids):
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                row = f"{class_id:<8} {class_name:<20}"

                for split in splits:
                    count = all_stats[split].get(class_id, 0)
                    # 计算在该分割中的占比
                    split_total = sum(all_stats[split].values())
                    if split_total > 0:
                        percentage = (count / split_total) * 100
                        row += f" {count} ({percentage:>5.1f}%)"
                    else:
                        row += f" {count} (  0.0%)"

                print(row)
        else:
            print("没有找到任何标签")

        print(f"{'='*60}")
        print("\n[SUCCESS] 数据集标签分析完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO数据集标签统计工具')
    parser.add_argument('--input', type=str,
                       default='F:\wenw\work\dataset\dataset_no_game_7class_1231',
                       help='输入数据集路径（YOLO格式）')

    args = parser.parse_args()

    # 检查输入路径
    if not Path(args.input).exists():
        print(f"[ERROR] 输入路径不存在: {args.input}")
        return

    print(f"[INFO] 开始分析YOLO数据集标签...")
    print(f"[INFO] 输入路径: {args.input}")

    # 创建分析器并执行
    analyzer = YOLODatasetAnalyzer(args.input)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
