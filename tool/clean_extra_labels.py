#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗多余的标签文件脚本
删除没有对应图片的标签文件
支持标准YOLO数据集结构
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import shutil


class LabelCleaner:
    """标签清洗类"""

    def __init__(self, dataset_path: str, dry_run: bool = False):
        """
        初始化标签清洗器

        Args:
            dataset_path: 数据集根目录路径
            dry_run: 是否只是模拟运行（不实际删除文件）
        """
        self.dataset_path = Path(dataset_path)
        self.dry_run = dry_run

        # 自动检测数据集结构
        self.dataset_type = self._detect_dataset_type()

        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'orphan_labels': 0,
            'deleted_labels': 0,
            'kept_labels': 0
        }

    def _detect_dataset_type(self) -> str:
        """
        检测数据集结构类型
        Returns: 'nested' (train/val/test 直接在根目录，各自包含images/labels) 或
                'standard' (images/labels 在根目录，下面有train/val/test子目录)
                'simple' (只有images和labels在根目录)
        """
        # 检查simple结构：根目录直接有images和labels
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        if images_dir.exists() and labels_dir.exists():
            # 检查是否有train/val子目录
            has_subdirs = any((images_dir / d).exists() for d in ['train', 'val', 'test'])
            if not has_subdirs:
                print("[INFO] 检测到simple结构：images/和labels/直接在根目录")
                return "simple"

        # 检查nested结构：train/val/test直接在根目录
        train_dir = self.dataset_path / "train"
        val_dir = self.dataset_path / "val"

        if (train_dir.exists() and val_dir.exists() and
            (train_dir / "images").exists() and (val_dir / "images").exists()):
            print("[INFO] 检测到nested结构：train/val/test目录在根目录")
            return "nested"

        # 检查standard结构：images/labels在根目录，下面有train/val/test
        if (images_dir.exists() and labels_dir.exists() and
            (images_dir / "train").exists() and (labels_dir / "train").exists()):
            print("[INFO] 检测到standard结构：images/labels目录在根目录，下面有train/val/test")
            return "standard"

        # 容错处理
        if images_dir.exists() and labels_dir.exists():
            print("[INFO] 检测到standard结构（部分特征）")
            return "standard"
        elif train_dir.exists() and val_dir.exists():
            print("[INFO] 检测到nested结构（部分特征）")
            return "nested"
        else:
            print(f"[ERROR] 无法识别的数据集结构: {self.dataset_path}")
            print(f"[ERROR] 请确保数据集符合以下结构之一：")
            print("  Simple结构：dataset/images/, dataset/labels/")
            print("  Nested结构：dataset/train/images/, dataset/val/images/, dataset/test/images/")
            print("  Standard结构：dataset/images/train/, dataset/labels/train/")
            raise ValueError(f"无法识别的数据集结构: {self.dataset_path}")

    def _get_dataset_splits(self) -> List[str]:
        """
        获取数据集的分割列表
        Returns: 包含 'train', 'val', 'test', 或 'simple' 的列表
        """
        splits = []

        if self.dataset_type == "simple":
            # simple结构：直接在根目录的images和labels
            splits.append("simple")
        elif self.dataset_type == "nested":
            # nested结构：检查train/val/test目录
            for split in ['train', 'val', 'test']:
                split_dir = self.dataset_path / split
                if split_dir.exists() and (split_dir / "images").exists():
                    splits.append(split)
        else:  # standard
            # standard结构：检查images/train, images/val, images/test
            images_dir = self.dataset_path / "images"
            labels_dir = self.dataset_path / "labels"
            for split in ['train', 'val', 'test']:
                if ((images_dir / split).exists() and
                    (labels_dir / split).exists()):
                    splits.append(split)

        return splits

    def _get_image_stems(self, images_dir: Path) -> set:
        """
        获取图片目录中所有文件的主干（不含扩展名）

        Args:
            images_dir: 图片目录路径

        Returns: 图片文件主干集合
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                           '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF']

        image_stems = set()
        for ext in image_extensions:
            for image_file in images_dir.glob(f"*{ext}"):
                image_stems.add(image_file.stem)

        return image_stems

    def _get_label_files(self, labels_dir: Path) -> List[Path]:
        """
        获取标签目录中所有txt文件

        Args:
            labels_dir: 标签目录路径

        Returns: 标签文件列表
        """
        return list(labels_dir.glob("*.txt"))

    def clean_split(self, split_name: str):
        """
        清理某个数据集分割中多余的标签文件

        Args:
            split_name: 分割名称（train、val、test或simple）
        """
        print(f"\n{'='*60}")
        print(f"[INFO] 开始处理 {split_name} 数据集...")
        print(f"{'='*60}")

        # 根据数据集类型设置路径
        if self.dataset_type == "simple":
            images_dir = self.dataset_path / "images"
            labels_dir = self.dataset_path / "labels"
        elif self.dataset_type == "nested":
            images_dir = self.dataset_path / split_name / "images"
            labels_dir = self.dataset_path / split_name / "labels"
        else:  # standard
            images_dir = self.dataset_path / "images" / split_name
            labels_dir = self.dataset_path / "labels" / split_name

        # 验证路径存在性
        if not images_dir.exists():
            print(f"[ERROR] 图片目录不存在: {images_dir}")
            return

        if not labels_dir.exists():
            print(f"[WARNING] 标签目录不存在: {labels_dir}")
            return

        # 获取所有图片和标签
        image_stems = self._get_image_stems(images_dir)
        label_files = self._get_label_files(labels_dir)

        num_images = len(image_stems)
        num_labels = len(label_files)

        self.stats['total_images'] += num_images
        self.stats['total_labels'] += num_labels

        print(f"[INFO] 图片数量: {num_images}")
        print(f"[INFO] 标签数量: {num_labels}")

        if num_labels > num_images:
            print(f"[INFO] 标签文件比图片多 {num_labels - num_images} 个")
        elif num_labels < num_images:
            print(f"[WARNING] 标签文件比图片少 {num_images - num_labels} 个（可能存在负样本）")

        # 找出没有对应图片的标签文件
        orphan_labels = []
        for label_file in label_files:
            if label_file.stem not in image_stems:
                orphan_labels.append(label_file)

        num_orphans = len(orphan_labels)
        self.stats['orphan_labels'] += num_orphans

        if num_orphans == 0:
            print(f"[SUCCESS] 所有标签文件都有对应的图片！")
            self.stats['kept_labels'] += num_labels
            return

        print(f"\n[INFO] 发现 {num_orphans} 个多余的标签文件（没有对应图片）:")

        # 显示要删除的文件
        for i, label_file in enumerate(orphan_labels[:20], 1):  # 只显示前20个
            print(f"  {i}. {label_file.name}")

        if num_orphans > 20:
            print(f"  ... 还有 {num_orphans - 20} 个文件")

        # 删除多余的标签文件
        if self.dry_run:
            print(f"\n[DRY RUN] 模拟运行：不会实际删除文件")
            print(f"[DRY RUN] 将删除 {num_orphans} 个标签文件")
        else:
            print(f"\n[WARNING] 准备删除这 {num_orphans} 个多余的标签文件")
            confirm = input("确认删除？(yes/no): ").strip().lower()

            if confirm in ['yes', 'y']:
                deleted_count = 0
                for label_file in orphan_labels:
                    try:
                        label_file.unlink()
                        deleted_count += 1
                        if deleted_count % 100 == 0:
                            print(f"[INFO] 已删除 {deleted_count}/{num_orphans} 个文件...")
                    except Exception as e:
                        print(f"[ERROR] 删除失败 {label_file}: {e}")

                self.stats['deleted_labels'] += deleted_count
                self.stats['kept_labels'] += (num_labels - deleted_count)
                print(f"[SUCCESS] 成功删除 {deleted_count} 个多余的标签文件")
            else:
                print(f"[INFO] 取消删除操作")
                self.stats['kept_labels'] += num_labels

    def clean_all(self):
        """清洗整个数据集中多余的标签文件"""
        print("[INFO] 开始清洗多余的标签文件...")
        print(f"[INFO] 数据集路径: {self.dataset_path}")
        print(f"[INFO] 数据集结构类型: {self.dataset_type}")

        if self.dry_run:
            print("[INFO] 模拟运行模式（不会实际删除文件）")

        # 显示检测到的数据集结构
        if self.dataset_type == "simple":
            print("[INFO] 数据集结构:")
            print("  输入: dataset/")
            print("        ├── images/")
            print("        └── labels/")
        elif self.dataset_type == "nested":
            print("[INFO] 数据集结构:")
            print("  输入: dataset/")
            print("        ├── train/images/")
            print("        ├── train/labels/")
            print("        ├── val/images/")
            print("        ├── val/labels/")
            print("        └── test/(optional)")
        else:  # standard
            print("[INFO] 数据集结构:")
            print("  输入: dataset/")
            print("        ├── images/train/")
            print("        ├── labels/train/")
            print("        ├── images/val/")
            print("        ├── labels/val/")
            print("        └── images/test/labels/test/(optional)")

        # 获取所有可用的数据集分割
        splits = self._get_dataset_splits()
        print(f"[INFO] 检测到数据集分割: {splits}")

        if not splits:
            print("[ERROR] 未找到有效的数据集分割！")
            print("[ERROR] 请检查数据集结构是否正确")
            return

        # 处理每个分割
        for split in splits:
            self.clean_split(split)

        # 打印总结
        self._print_summary()

    def _print_summary(self):
        """打印清洗统计摘要"""
        print(f"\n{'='*60}")
        print("[INFO] 清洗完成统计")
        print(f"{'='*60}")
        print(f"  总图片数量: {self.stats['total_images']}")
        print(f"  总标签数量: {self.stats['total_labels']}")
        print(f"  多余标签数量: {self.stats['orphan_labels']}")
        if self.dry_run:
            print(f"  模拟删除数量: {self.stats['orphan_labels']}")
            print(f"  保留标签数量: {self.stats['total_labels']}")
        else:
            print(f"  实际删除数量: {self.stats['deleted_labels']}")
            print(f"  保留标签数量: {self.stats['kept_labels']}")

        if self.stats['total_labels'] > 0:
            keep_rate = (self.stats['kept_labels'] / self.stats['total_labels']) * 100
            print(f"  标签保留率: {keep_rate:.2f}%")

        print(f"{'='*60}")

        if self.dry_run:
            print("[INFO] 这是模拟运行，没有实际删除文件")
            print("[INFO] 如果结果符合预期，请去掉 --dry-run 参数重新运行")


# ==================== 配置区域 ====================
# 在这里设置你的默认参数
DEFAULT_INPUT = "F:\wenw\work\dataset\原始补充\红外人像1230\images"  # 数据集路径
DEFAULT_DRY_RUN = False  # 是否模拟运行（不实际删除文件）- 首次运行建议设为True
DEFAULT_BACKUP = True  # 是否自动备份（设为True更安全）
# =================================================


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='清洗YOLO数据集中多余的标签文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 清洗simple结构数据集（只有images和labels）
  python clean_extra_labels.py --input /path/to/dataset

  # 清洗nested/standard结构数据集
  python clean_extra_labels.py --input /path/to/dataset

  # 模拟运行（不实际删除文件）
  python clean_extra_labels.py --input /path/to/dataset --dry-run

  # 备份后再清洗（推荐）
  python clean_extra_labels.py --input /path/to/dataset --backup

注意: 可以在脚本开头的配置区域直接设置默认参数
        """
    )

    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                       help='输入数据集路径（YOLO格式）')
    parser.add_argument('--dry-run', action='store_true', default=DEFAULT_DRY_RUN,
                       help='模拟运行，不实际删除文件（用于预览）')
    parser.add_argument('--backup', action='store_true', default=DEFAULT_BACKUP,
                       help='在清洗前自动备份labels文件夹')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='取消dry-run模式，实际执行删除')
    parser.add_argument('--no-backup', action='store_true',
                       help='取消自动备份')

    args = parser.parse_args()

    # 处理命令行参数覆盖
    if args.no_dry_run:
        args.dry_run = False
    if args.no_backup:
        args.backup = False

    # 显示当前配置
    print(f"[INFO] 当前配置:")
    print(f"  - 输入路径: {args.input}")
    print(f"  - 模拟运行: {args.dry_run}")
    print(f"  - 自动备份: {args.backup}")
    print(f"  - 提示: 要修改配置，请编辑脚本开头的配置区域\n")

    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入路径不存在: {args.input}")
        return

    print(f"[INFO] 输入路径: {args.input}")

    # 备份选项
    if args.backup and not args.dry_run:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.parent / f"{input_path.name}_backup_{timestamp}"

        print(f"\n[INFO] 创建备份: {backup_path}")
        try:
            shutil.copytree(input_path, backup_path)
            print(f"[SUCCESS] 备份创建成功: {backup_path}")
        except Exception as e:
            print(f"[ERROR] 备份创建失败: {e}")
            print(f"[WARNING] 继续执行清洗操作...")
    elif args.backup and args.dry_run:
        print(f"\n[INFO] --backup 参数在 --dry-run 模式下不执行")

    # 创建清洗器并执行
    cleaner = LabelCleaner(args.input, dry_run=args.dry_run)
    cleaner.clean_all()


if __name__ == "__main__":
    main()
