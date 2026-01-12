#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除val文件夹中没有标签的图片
支持两种数据集结构：nested (val/images) 和 standard (images/val)
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValImageCleaner:
    """验证集无标签图片清理工具"""

    def __init__(self, dataset_path: str, backup: bool = True):
        """
        初始化清理工具

        Args:
            dataset_path: 数据集根目录路径
            backup: 是否在删除前备份图片
        """
        self.dataset_path = Path(dataset_path)
        self.backup = backup

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        # 检测数据集结构类型
        self.dataset_type = self._detect_dataset_type()
        logger.info(f"检测到数据集结构类型: {self.dataset_type}")

        # 获取val目录路径
        self.val_images_dir, self.val_labels_dir = self._get_val_paths()

        # 创建备份目录
        if self.backup:
            self.backup_dir = self.dataset_path / "backup_unlabeled_val_images"
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"备份目录: {self.backup_dir}")

    def _detect_dataset_type(self) -> str:
        """
        检测数据集结构类型
        Returns: 'nested' (val/images) 或 'standard' (images/val)
        """
        # 检查 nested 结构: val/images 或 valid/images
        val_dir = self.dataset_path / "val" / "images"
        valid_dir = self.dataset_path / "valid" / "images"

        # 检查 standard 结构: images/val
        images_val_dir = self.dataset_path / "images" / "val"

        if val_dir.exists() or valid_dir.exists():
            logger.info("检测到 nested 结构: val/images 或 valid/images")
            return 'nested'
        elif images_val_dir.exists():
            logger.info("检测到 standard 结构: images/val")
            return 'standard'
        else:
            logger.warning("无法自动检测数据集结构，默认使用 nested 结构")
            return 'nested'

    def _get_val_paths(self) -> tuple:
        """
        获取val目录的图片和标签路径

        Returns:
            tuple: (images_dir, labels_dir)
        """
        if self.dataset_type == 'nested':
            # 尝试 val 和 valid 两种命名
            val_images = self.dataset_path / "val" / "images"
            valid_images = self.dataset_path / "valid" / "images"

            if val_images.exists():
                images_dir = val_images
                labels_dir = self.dataset_path / "val" / "labels"
            elif valid_images.exists():
                images_dir = valid_images
                labels_dir = self.dataset_path / "valid" / "labels"
            else:
                raise FileNotFoundError("找不到 val 或 valid 目录")
        else:  # standard
            images_dir = self.dataset_path / "images" / "val"
            labels_dir = self.dataset_path / "labels" / "val"

            if not images_dir.exists():
                raise FileNotFoundError(f"找不到目录: {images_dir}")

        logger.info(f"验证集图片目录: {images_dir}")
        logger.info(f"验证集标签目录: {labels_dir}")

        return images_dir, labels_dir

    def find_unlabeled_images(self) -> List[Dict]:
        """
        查找所有没有标签的图片

        Returns:
            List[Dict]: 无标签图片信息列表
        """
        logger.info("开始扫描验证集图片...")

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.val_images_dir.glob(f"*{ext}"))
            image_files.extend(self.val_images_dir.glob(f"*{ext.upper()}"))

        # 去重并排序
        image_files = sorted(list(set(image_files)))

        logger.info(f"找到 {len(image_files)} 个图片文件")

        # 检查每个图片
        unlabeled_images = []
        labeled_count = 0

        for img_file in image_files:
            label_file = self.val_labels_dir / f"{img_file.stem}.txt"

            if not label_file.exists():
                # 标签文件不存在
                unlabeled_images.append({
                    'image': img_file,
                    'reason': '标签文件不存在',
                    'expected_label': label_file
                })
            elif label_file.stat().st_size == 0:
                # 标签文件为空
                unlabeled_images.append({
                    'image': img_file,
                    'reason': '标签文件为空',
                    'label_file': label_file
                })
            else:
                # 有标签
                labeled_count += 1

        logger.info(f"有标签图片: {labeled_count}")
        logger.info(f"无标签图片: {len(unlabeled_images)}")

        return unlabeled_images

    def preview_unlabeled_images(self, unlabeled_images: List[Dict], limit: int = 20):
        """
        预览无标签图片列表

        Args:
            unlabeled_images: 无标签图片列表
            limit: 显示数量限制
        """
        print("\n" + "=" * 80)
        print("无标签图片预览")
        print("=" * 80)
        print(f"总计: {len(unlabeled_images)} 张图片将被删除\n")

        for i, item in enumerate(unlabeled_images[:limit], 1):
            print(f"{i}. {item['image'].name}")
            print(f"   原因: {item['reason']}")
            print(f"   路径: {item['image']}")

        if len(unlabeled_images) > limit:
            print(f"\n... 还有 {len(unlabeled_images) - limit} 张图片未显示")

        print("=" * 80)

    def backup_images(self, unlabeled_images: List[Dict]) -> int:
        """
        备份无标签图片

        Args:
            unlabeled_images: 无标签图片列表

        Returns:
            int: 成功备份的数量
        """
        logger.info("开始备份无标签图片...")

        backup_count = 0
        for item in unlabeled_images:
            img_file = item['image']
            backup_path = self.backup_dir / img_file.name

            try:
                shutil.copy2(img_file, backup_path)
                backup_count += 1
                logger.debug(f"备份: {img_file.name}")
            except Exception as e:
                logger.error(f"备份失败 {img_file.name}: {e}")

        logger.info(f"成功备份 {backup_count} 张图片到: {self.backup_dir}")
        return backup_count

    def delete_images(self, unlabeled_images: List[Dict]) -> int:
        """
        删除无标签图片

        Args:
            unlabeled_images: 无标签图片列表

        Returns:
            int: 成功删除的数量
        """
        logger.info("开始删除无标签图片...")

        deleted_count = 0
        failed_count = 0

        for item in unlabeled_images:
            img_file = item['image']

            try:
                img_file.unlink()
                deleted_count += 1
                logger.debug(f"删除: {img_file.name}")
            except Exception as e:
                logger.error(f"删除失败 {img_file.name}: {e}")
                failed_count += 1

        logger.info(f"成功删除 {deleted_count} 张图片")
        if failed_count > 0:
            logger.warning(f"删除失败 {failed_count} 张图片")

        return deleted_count

    def generate_report(self, unlabeled_images: List[Dict], deleted_count: int):
        """
        生成清理报告

        Args:
            unlabeled_images: 无标签图片列表
            deleted_count: 删除数量
        """
        report_file = self.dataset_path / "val_cleaning_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("验证集无标签图片清理报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"数据集路径: {self.dataset_path}\n")
            f.write(f"验证集目录: {self.val_images_dir}\n")
            f.write(f"数据集结构: {self.dataset_type}\n")
            f.write(f"备份启用: {self.backup}\n")
            if self.backup:
                f.write(f"备份目录: {self.backup_dir}\n")
            f.write("\n")

            f.write(f"删除的图片数量: {deleted_count}\n")
            f.write(f"删除的图片列表:\n")
            f.write("-" * 70 + "\n")

            for i, item in enumerate(unlabeled_images, 1):
                f.write(f"{i}. {item['image'].name}\n")
                f.write(f"   原因: {item['reason']}\n")
                f.write(f"   原始路径: {item['image']}\n")

        logger.info(f"清理报告已保存到: {report_file}")

    def clean(self, preview_only: bool = False):
        """
        执行清理操作

        Args:
            preview_only: 是否只预览不删除
        """
        try:
            logger.info("开始清理验证集无标签图片...")

            # 1. 查找无标签图片
            unlabeled_images = self.find_unlabeled_images()

            if len(unlabeled_images) == 0:
                print("\n✓ 没有发现无标签的图片，无需清理！")
                return

            # 2. 预览
            self.preview_unlabeled_images(unlabeled_images)

            if preview_only:
                print("\n预览模式，不执行实际删除操作")
                return

            # 3. 询问用户确认
            print("\n" + "=" * 80)
            print("警告：此操作将删除上述无标签的图片！")
            if self.backup:
                print(f"图片将在删除前备份到: {self.backup_dir}")
            else:
                print("警告：未启用备份，删除后无法恢复！")
            print("=" * 80)

            confirm = input("\n确认删除这些图片? (yes/no): ").strip().lower()
            if confirm not in ['yes', 'y']:
                print("\n操作已取消")
                return

            # 4. 备份（如果启用）
            if self.backup:
                self.backup_images(unlabeled_images)

            # 5. 删除图片
            deleted_count = self.delete_images(unlabeled_images)

            # 6. 生成报告
            self.generate_report(unlabeled_images, deleted_count)

            print("\n" + "=" * 80)
            print("清理完成！")
            print(f"成功删除: {deleted_count} 张图片")
            if self.backup:
                print(f"备份位置: {self.backup_dir}")
            print(f"清理报告: {self.dataset_path / 'val_cleaning_report.txt'}")
            print("=" * 80)

        except Exception as e:
            logger.error(f"清理过程中发生错误: {str(e)}")
            raise


def main():
    """主函数"""
    # 配置参数
    DATASET_PATH = r"F:\wenw\work\dataset\dataset_no_game_4class_1212"

    # 是否在删除前备份图片
    BACKUP = True

    # 是否只预览不删除（设置为 True 时只显示列表，不执行删除）
    PREVIEW_ONLY = False

    try:
        print("=" * 80)
        print("验证集无标签图片清理工具")
        print("=" * 80)
        print(f"数据集路径: {DATASET_PATH}")
        print(f"备份启用: {BACKUP}")
        print(f"预览模式: {PREVIEW_ONLY}")
        print()

        # 创建清理器
        cleaner = ValImageCleaner(DATASET_PATH, backup=BACKUP)

        # 执行清理
        cleaner.clean(preview_only=PREVIEW_ONLY)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
