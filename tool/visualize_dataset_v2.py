#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集标签可视化脚本 v2
将YOLO格式的标签以纯边界框形式绘制在图片上，目标框内部无填充颜色
支持标准YOLO数据集结构，自动检测是否存在测试集
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import argparse


class DatasetVisualizerV2:
    """数据集可视化类v2 - 纯边界框版本"""

    def __init__(self, dataset_path: str, output_path: str):
        """
        初始化可视化器

        Args:
            dataset_path: 数据集根目录路径
            output_path: 输出目录路径
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

        # 自动检测数据集结构
        self.dataset_type = self._detect_dataset_type()

        # 读取类别信息
        self.class_names = self._load_class_names()
        self.class_colors = self._generate_class_colors()

        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)

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

        # 第一种结构判断：根目录下有 train 目录（val/test 可选）且包含 images 子目录
        if train_dir.exists() and (train_dir / "images").exists():
            if val_dir.exists():
                print("[INFO] 检测到第一种结构：train/val/test 目录在根目录")
            else:
                print("[INFO] 检测到第一种结构（仅包含 train）：train 目录在根目录")
            return "nested"

        # 第二种结构判断：根目录下有 images/labels 目录
        elif images_dir.exists() and labels_dir.exists() and (images_dir / "train").exists():
            if (images_dir / "val").exists():
                print("[INFO] 检测到第二种结构：images/labels 目录在根目录，下面有 train/val/test")
            else:
                print("[INFO] 检测到第二种结构（仅包含 train）：images/labels 目录在根目录，下面有 train")
            return "standard"

        # 容错处理：只有一种结构的部分特征
        elif train_dir.exists():
            print("[INFO] 检测到第一种结构（部分特征，仅 train 目录）")
            return "nested"
        elif images_dir.exists() and labels_dir.exists():
            print("[INFO] 检测到第二种结构（部分特征）")
            return "standard"
        else:
            print(f"[ERROR] 无法识别的数据集结构: {self.dataset_path}")
            print(f"[ERROR] 请确保数据集符合以下两种结构之一：")
            print("  第一种结构：dataset/train/images/, dataset/val/images/(可选), dataset/test/images/(可选)")
            print("  第二种结构：dataset/images/train/, dataset/labels/train/, dataset/images/val/(可选), dataset/labels/val/(可选)")
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

    def _generate_class_colors(self) -> dict:
        """
        为类别生成颜色映射
        Returns: 类别ID到颜色的映射字典
        """
        colors = {}
        n_classes = len(self.class_names)

        # 生成不同的颜色
        for i in range(n_classes):
            hue = int(180.0 * i / n_classes)  # 使用HSV色彩空间，均匀分布色调
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors[i] = tuple(int(c) for c in color)

        return colors

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

    def parse_yolo_bbox_label(self, label_line: str) -> Tuple[int, Tuple[float, float, float, float]]:
        """
        解析YOLO边界框标签行

        Args:
            label_line: 标签行，格式为 "class_id x_center y_center width height"

        Returns:
            class_id: 类别ID
            bbox: 边界框坐标 (x_center, y_center, width, height)
        """
        parts = label_line.strip().split()
        class_id = int(parts[0])

        # 提取边界框坐标
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        return class_id, (x_center, y_center, width, height)

    def draw_bbox_on_image(self, image: np.ndarray, bbox: Tuple[float, float, float, float],
                         class_id: int) -> np.ndarray:
        """
        在图片上绘制YOLO边界框标签（纯边框，无填充颜色）

        Args:
            image: 原始图片
            bbox: 边界框坐标 (x_center, y_center, width, height) - 归一化坐标
            class_id: 类别ID

        Returns:
            绘制后的图片
        """
        h, w = image.shape[:2]
        x_center, y_center, width, height = bbox

        # 将归一化坐标转换为像素坐标
        x_center_abs = int(x_center * w)
        y_center_abs = int(y_center * h)
        width_abs = int(width * w)
        height_abs = int(height * h)

        # 计算左上角和右下角坐标
        x1 = max(0, x_center_abs - width_abs // 2)
        y1 = max(0, y_center_abs - height_abs // 2)
        x2 = min(w - 1, x_center_abs + width_abs // 2)
        y2 = min(h - 1, y_center_abs + height_abs // 2)

        # 获取类别颜色
        color = self.class_colors.get(class_id, (255, 255, 255))

        result = image.copy()

        # 只绘制矩形边框，不填充颜色
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 绘制类别标签
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'

        # 计算文本位置
        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = max(y1 - 10, text_size[1] + 5)

        # 确保文本框在图片范围内
        if text_x + text_size[0] + 10 > w:
            text_x = w - text_size[0] - 10
        if text_y < 0:
            text_y = y1 + text_size[1] + 10

        # 绘制文本背景
        cv2.rectangle(result,
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 10, text_y + 5),
                     color, -1)

        # 绘制文本
        cv2.putText(result, class_name,
                   (text_x + 5, text_y - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result

    def process_image(self, image_path: Path, label_path: Path, output_path: Path):
        """
        处理单个图片和对应的标签文件

        Args:
            image_path: 图片路径
            label_path: 标签文件路径
            output_path: 输出图片路径
        """
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARNING] 无法读取图片: {image_path}")
                return

            # 如果标签文件不存在，直接复制原图（负样本）
            if not label_path.exists():
                cv2.imwrite(str(output_path), image)
                print(f"[INFO] 复制负样本: {output_path.name}")
                return

            # 读取并解析标签
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            result_image = image.copy()

            # 绘制每个标签
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    class_id, bbox = self.parse_yolo_bbox_label(line)
                    result_image = self.draw_bbox_on_image(result_image, bbox, class_id)
                except Exception as e:
                    print(f"[WARNING] 解析标签失败 {label_path}: {line} - {e}")
                    continue

            # 保存结果
            cv2.imwrite(str(output_path), result_image)
            print(f"[INFO] 生成可视化图片: {output_path.name}")

        except Exception as e:
            print(f"[ERROR] 处理失败 {image_path}: {e}")

    def process_dataset_split(self, split_name: str):
        """
        处理数据集的某个分割（train/val/test）

        Args:
            split_name: 分割名称（train、val 或 test）
        """
        print(f"\n[INFO] 开始处理 {split_name} 数据集...")

        # 根据数据集类型设置路径
        if self.dataset_type == "nested":
            # 第一种结构：train/val/test 直接在根目录，各自包含 images/labels
            images_dir = self.dataset_path / split_name / "images"
            labels_dir = self.dataset_path / split_name / "labels"
            output_dir = self.output_path / split_name / "images"
        else:  # standard
            # 第二种结构：所有图片在 images/train, images/val, images/test
            images_dir = self.dataset_path / "images" / split_name
            labels_dir = self.dataset_path / "labels" / split_name
            output_dir = self.output_path / "images" / split_name

        # 验证输入路径存在性
        if not images_dir.exists():
            print(f"[ERROR] 图片目录不存在: {images_dir}")
            return

        # 创建输出目录（保持与输入相同的结构）
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        print(f"[INFO] 找到 {len(image_files)} 张图片")
        if len(image_files) == 0:
            print(f"[WARNING] 在 {images_dir} 中没有找到图片文件")
            return

        processed_count = 0
        skipped_count = 0

        for i, image_file in enumerate(image_files):
            # 检查是否达到限制（如果有设置的话）
            # 这里使用全局变量来传递limit限制
            if hasattr(self, '_limit') and processed_count >= self._limit:
                break

            # 查找对应的标签文件
            label_file = labels_dir / f"{image_file.stem}.txt"

            # 生成输出文件名（添加visual后缀）
            output_filename = f"{image_file.stem}_visual_v2{image_file.suffix}"
            output_file = output_dir / output_filename

            # 检查是否已经处理过（避免重复处理）
            if output_file.exists():
                print(f"[INFO] 跳过已存在的文件: {output_filename}")
                skipped_count += 1
                continue

            # 处理图片
            self.process_image(image_file, label_file, output_file)
            processed_count += 1

            # 打印进度（每处理100张图片打印一次）
            if processed_count % 100 == 0:
                print(f"[INFO] 已处理 {processed_count} 张图片...")

        print(f"[INFO] {split_name} 数据集处理完成")
        print(f"  - 新处理: {processed_count} 张图片")
        print(f"  - 跳过已存在: {skipped_count} 张图片")

    def visualize_dataset(self):
        """可视化整个数据集"""
        print("[INFO] 开始数据集标签可视化（v2 - 纯边界框版本）...")
        print(f"[INFO] 输入路径: {self.dataset_path}")
        print(f"[INFO] 输出路径: {self.output_path}")
        print(f"[INFO] 数据集结构类型: {self.dataset_type}")
        print(f"[INFO] 检测到的类别: {self.class_names}")

        # 显示检测到的数据集结构
        if self.dataset_type == "nested":
            print("[INFO] 数据集结构:")
            print("  输入: dataset/")
            print("        ├── train/images/")
            print("        ├── train/labels/")
            print("        ├── val/images/")
            print("        ├── val/labels/")
            print("        └── test/(optional)")
            print("  输出: output/")
            print("        ├── train/images/")
            print("        ├── val/images/")
            print("        └── test/images/(optional)")
        else:  # standard
            print("[INFO] 数据集结构:")
            print("  输入: dataset/")
            print("        ├── images/train/")
            print("        ├── labels/train/")
            print("        ├── images/val/")
            print("        ├── labels/val/")
            print("        └── images/test/labels/test/(optional)")
            print("  输出: output/")
            print("        ├── images/train/")
            print("        ├── images/val/")
            print("        └── images/test/(optional)")

        # 复制数据集配置文件
        yaml_files = list(self.dataset_path.glob("*.yaml")) + list(self.dataset_path.glob("*.yml"))
        if yaml_files:
            import shutil
            for yaml_file in yaml_files:
                output_yaml = self.output_path / yaml_file.name
                shutil.copy2(yaml_file, output_yaml)
                print(f"[INFO] 复制数据集配置文件: {output_yaml}")

        # 获取所有可用的数据集分割
        splits = self._get_dataset_splits()
        print(f"[INFO] 检测到数据集分割: {splits}")

        if not splits:
            print("[ERROR] 未找到有效的数据集分割！")
            print("[ERROR] 请检查数据集结构是否正确")
            return

        # 处理每个分割
        total_processed = 0
        for split in splits:
            print(f"\n{'='*50}")
            self.process_dataset_split(split)
            print(f"{'='*50}")

        print(f"\n[SUCCESS] 数据集可视化完成！")
        print(f"[INFO] 输出目录: {self.output_path}")
        print("[INFO] 输出说明:")
        print("  - 有标签的图片: 绘制纯边界框和类别标签（无填充颜色）")
        print("  - 无标签的图片: 直接复制原图（负样本）")
        print("  - 每个类别使用不同的边框颜色进行区分")
        print("  - 输出文件名格式: 原文件名_visual_v2.jpg")
        print(f"  - 类别数量: {len(self.class_names)}")
        print("[INFO] 可以直接查看输出目录中的可视化图片来验证标注质量")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='标准YOLO数据集标签可视化工具 v2 - 纯边界框版本')
    parser.add_argument('--input', type=str, default='F:\\wenw\\work\\dataset\\dataset_no_game_8class_1231',
                       help='输入数据集路径（YOLO格式）')
    parser.add_argument('--output', type=str, default='F:\\wenw\\work\\dataset\\dataset_no_game_8class_1231-visual-v2',
                       help='输出数据集路径（可视化后的结果）')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理的图片数量（用于快速测试）')

    args = parser.parse_args()

    # 检查输入路径
    if not Path(args.input).exists():
        print(f"[ERROR] 输入路径不存在: {args.input}")
        return

    print(f"[INFO] 开始可视化YOLO数据集（v2 - 纯边界框版本）...")
    print(f"[INFO] 输入路径: {args.input}")
    print(f"[INFO] 输出路径: {args.output}")
    if args.limit:
        print(f"[INFO] 限制处理图片数量: {args.limit}")

    # 创建可视化器并执行
    visualizer = DatasetVisualizerV2(args.input, args.output)

    # 如果设置了限制，将其存储为类属性
    if args.limit:
        visualizer._limit = args.limit
        print(f"[INFO] 全局限制处理图片数量: {args.limit}")

    visualizer.visualize_dataset()


if __name__ == "__main__":
    main()