#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集无标签图片分析工具
分析哪些图片没有对应的标签文件或标签文件为空
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnlabeledImageAnalyzer:
    """无标签图片分析器"""

    def __init__(self, dataset_path: str):
        """
        初始化分析器

        Args:
            dataset_path: 数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        # 检测数据集结构类型
        self.dataset_type = self._detect_dataset_type()
        logger.info(f"检测到数据集结构类型: {self.dataset_type}")

        # 获取数据集分割
        self.splits = self._get_dataset_splits()
        logger.info(f"检测到数据集分割: {self.splits}")

    def _detect_dataset_type(self) -> str:
        """
        检测数据集结构类型
        Returns: 'nested' (train/images, val/images) 或 'standard' (images/train, images/val)
        """
        # 方法1: 检查 nested 结构
        train_dir = self.dataset_path / "train" / "images"
        valid_dir = self.dataset_path / "valid" / "images"

        # 方法2: 检查 standard 结构
        images_train_dir = self.dataset_path / "images" / "train"
        images_val_dir = self.dataset_path / "images" / "val"

        if train_dir.exists() and valid_dir.exists():
            logger.info("检测到 nested 结构: train/images, valid/images")
            return 'nested'
        elif images_train_dir.exists() and images_val_dir.exists():
            logger.info("检测到 standard 结构: images/train, images/val")
            return 'standard'
        else:
            # 尝试只检查 train 目录
            if train_dir.exists():
                logger.info("通过 train 目录检测到 nested 结构")
                return 'nested'
            elif images_train_dir.exists():
                logger.info("通过 train 目录检测到 standard 结构")
                return 'standard'
            else:
                logger.warning("无法自动检测数据集结构，默认使用 nested 结构")
                return 'nested'

    def _get_dataset_splits(self) -> List[str]:
        """
        获取数据集的分割列表
        Returns: 包含 'train', 'val', 'valid', 'test' 的列表（如果存在）
        """
        splits = []

        if self.dataset_type == "nested":
            # 检查 train/valid/test 目录
            for split in ['train', 'valid', 'val', 'test']:
                split_dir = self.dataset_path / split / "images"
                if split_dir.exists():
                    splits.append(split)
        else:  # standard
            # 检查 images/train, images/val, images/test
            images_dir = self.dataset_path / "images"
            for split in ['train', 'val', 'valid', 'test']:
                split_dir = images_dir / split
                if split_dir.exists():
                    splits.append(split)

        return splits

    def analyze_split(self, split_name: str) -> Dict:
        """
        分析单个数据集分割

        Args:
            split_name: 分割名称（train、val、valid 或 test）

        Returns:
            Dict: 分析结果
        """
        logger.info(f"分析 {split_name} 数据集...")

        # 根据数据集类型确定路径
        if self.dataset_type == 'nested':
            images_dir = self.dataset_path / split_name / "images"
            labels_dir = self.dataset_path / split_name / "labels"
        else:  # standard
            images_dir = self.dataset_path / "images" / split_name
            labels_dir = self.dataset_path / "labels" / split_name

        if not images_dir.exists():
            logger.warning(f"图片目录不存在: {images_dir}")
            return {
                'split': split_name,
                'total_images': 0,
                'labeled_images': 0,
                'unlabeled_images': 0,
                'empty_label_files': 0,
                'missing_label_files': 0,
                'unlabeled_list': []
            }

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        # 去重并排序
        image_files = sorted(list(set(image_files)))

        logger.info(f"找到 {len(image_files)} 个图片文件")

        # 分析每个图片
        unlabeled_images = []
        missing_label_files = []
        empty_label_files = []
        labeled_count = 0

        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"

            if not label_file.exists():
                # 标签文件不存在
                missing_label_files.append(img_file)
                unlabeled_images.append({
                    'image': str(img_file.relative_to(self.dataset_path)),
                    'reason': '标签文件不存在',
                    'image_path': str(img_file),
                    'expected_label_path': str(label_file)
                })
            elif label_file.stat().st_size == 0:
                # 标签文件为空
                empty_label_files.append(img_file)
                unlabeled_images.append({
                    'image': str(img_file.relative_to(self.dataset_path)),
                    'reason': '标签文件为空',
                    'image_path': str(img_file),
                    'label_path': str(label_file)
                })
            else:
                # 有标签
                labeled_count += 1

        result = {
            'split': split_name,
            'total_images': len(image_files),
            'labeled_images': labeled_count,
            'unlabeled_images': len(unlabeled_images),
            'empty_label_files': len(empty_label_files),
            'missing_label_files': len(missing_label_files),
            'unlabeled_list': unlabeled_images
        }

        logger.info(f"{split_name}: 总计 {result['total_images']} 张，"
                   f"有标签 {result['labeled_images']} 张，"
                   f"无标签 {result['unlabeled_images']} 张 "
                   f"(缺失标签: {result['missing_label_files']}, 空标签: {result['empty_label_files']})")

        return result

    def analyze_all(self) -> Dict[str, Dict]:
        """
        分析所有数据集分割

        Returns:
            Dict: 所有分割的分析结果
        """
        logger.info("开始分析数据集...")

        results = {}
        for split in self.splits:
            results[split] = self.analyze_split(split)

        return results

    def generate_report(self, results: Dict[str, Dict], output_file: str = None):
        """
        生成分析报告

        Args:
            results: 分析结果
            output_file: 输出文件路径（可选）
        """
        logger.info("生成分析报告...")

        lines = []
        lines.append("=" * 70)
        lines.append("数据集无标签图片分析报告")
        lines.append("=" * 70)
        lines.append(f"数据集路径: {self.dataset_path}")
        lines.append(f"数据集结构: {self.dataset_type}")
        lines.append(f"数据集分割: {', '.join(self.splits)}")
        lines.append("=" * 70)
        lines.append("")

        # 统计总体情况
        total_images = sum(r['total_images'] for r in results.values())
        total_labeled = sum(r['labeled_images'] for r in results.values())
        total_unlabeled = sum(r['unlabeled_images'] for r in results.values())
        total_missing = sum(r['missing_label_files'] for r in results.values())
        total_empty = sum(r['empty_label_files'] for r in results.values())

        lines.append("总体统计:")
        lines.append(f"  总图片数: {total_images}")
        lines.append(f"  有标签图片: {total_labeled} ({total_labeled/total_images*100:.2f}%)" if total_images > 0 else "  有标签图片: 0")
        lines.append(f"  无标签图片: {total_unlabeled} ({total_unlabeled/total_images*100:.2f}%)" if total_images > 0 else "  无标签图片: 0")
        lines.append(f"    - 缺失标签文件: {total_missing}")
        lines.append(f"    - 空标签文件: {total_empty}")
        lines.append("")

        # 各分割详情
        for split_name, result in results.items():
            lines.append("-" * 70)
            lines.append(f"{split_name.upper()} 数据集:")
            lines.append(f"  总图片数: {result['total_images']}")
            lines.append(f"  有标签图片: {result['labeled_images']} "
                        f"({result['labeled_images']/result['total_images']*100:.2f}%)" if result['total_images'] > 0 else "  有标签图片: 0")
            lines.append(f"  无标签图片: {result['unlabeled_images']} "
                        f"({result['unlabeled_images']/result['total_images']*100:.2f}%)" if result['total_images'] > 0 else "  无标签图片: 0")
            lines.append(f"    - 缺失标签文件: {result['missing_label_files']}")
            lines.append(f"    - 空标签文件: {result['empty_label_files']}")

            # 如果有无标签图片，列出前20个
            if result['unlabeled_list']:
                lines.append(f"\n  无标签图片列表 (前20个):")
                for i, item in enumerate(result['unlabeled_list'][:20], 1):
                    lines.append(f"    {i}. {item['image']}")
                    lines.append(f"       原因: {item['reason']}")

                if len(result['unlabeled_list']) > 20:
                    lines.append(f"    ... 还有 {len(result['unlabeled_list']) - 20} 张图片未显示")

            lines.append("")

        lines.append("=" * 70)
        lines.append("分析完成")
        lines.append("=" * 70)

        report = "\n".join(lines)

        # 打印到控制台
        print("\n" + report)

        # 保存到文件
        if output_file is None:
            output_file = self.dataset_path / "unlabeled_images_report.txt"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"报告已保存到: {output_path}")

        # 生成详细列表文件（包含所有无标签图片）
        if total_unlabeled > 0:
            detail_file = self.dataset_path / "unlabeled_images_detail.txt"
            with open(detail_file, 'w', encoding='utf-8') as f:
                f.write("无标签图片详细列表\n")
                f.write("=" * 70 + "\n\n")

                for split_name, result in results.items():
                    if result['unlabeled_list']:
                        f.write(f"{split_name.upper()} 数据集 ({len(result['unlabeled_list'])} 张):\n")
                        f.write("-" * 70 + "\n")
                        for i, item in enumerate(result['unlabeled_list'], 1):
                            f.write(f"{i}. {item['image']}\n")
                            f.write(f"   原因: {item['reason']}\n")
                            if 'image_path' in item:
                                f.write(f"   图片路径: {item['image_path']}\n")
                            if 'expected_label_path' in item:
                                f.write(f"   期望标签路径: {item['expected_label_path']}\n")
                            if 'label_path' in item:
                                f.write(f"   标签路径: {item['label_path']}\n")
                            f.write("\n")
                        f.write("\n")

            logger.info(f"详细列表已保存到: {detail_file}")

        return report

    def print_unlabeled_images(self, results: Dict[str, Dict], limit: int = None):
        """
        打印无标签图片的列表

        Args:
            results: 分析结果
            limit: 限制显示数量（None表示显示全部）
        """
        print("\n" + "=" * 70)
        print("无标签图片详细列表")
        print("=" * 70)

        for split_name, result in results.items():
            if not result['unlabeled_list']:
                continue

            print(f"\n{split_name.upper()} 数据集 ({len(result['unlabeled_list'])} 张):")
            print("-" * 70)

            items_to_show = result['unlabeled_list'][:limit] if limit else result['unlabeled_list']

            for i, item in enumerate(items_to_show, 1):
                print(f"{i}. {item['image']}")
                print(f"   原因: {item['reason']}")

            if limit and len(result['unlabeled_list']) > limit:
                print(f"\n... 还有 {len(result['unlabeled_list']) - limit} 张图片未显示")


def main():
    """主函数"""
    # 数据集路径
    DATASET_PATH = r"F:\wenw\work\dataset\dataset_no_game_4class_1212"

    # 是否在控制台显示详细列表（None显示全部，或指定数量如100）
    SHOW_LIMIT = None  # None, 10, 50, 100 等

    try:
        print("=" * 70)
        print("数据集无标签图片分析工具")
        print("=" * 70)
        print(f"数据集路径: {DATASET_PATH}")
        print()

        # 创建分析器
        analyzer = UnlabeledImageAnalyzer(DATASET_PATH)

        # 分析所有数据集分割
        results = analyzer.analyze_all()

        # 生成报告
        analyzer.generate_report(results)

        # 打印详细列表
        if SHOW_LIMIT:
            analyzer.print_unlabeled_images(results, limit=SHOW_LIMIT)

        print("\n" + "=" * 70)
        print("分析完成！")
        print("=" * 70)

    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
