#!/usr/bin/env python3
"""
YOLO数据集全面清洗脚本
功能：
1. 支持用户选择要清洗的标签类别（0,1,2等）
2. 自动检测数据集子集（train/val/test）
3. 清洗指定标签后重新编号剩余标签
4. 删除清洗后没有标签的图片
5. 保留负样本（没有标签的图片）不处理
6. 支持备份和试运行模式
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


class YOLODatasetCleaner:
    """YOLO数据集清洗器"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

    def detect_subsets(self) -> List[str]:
        """检测数据集中存在的子集，支持两种常见的数据集结构"""
        possible_subsets = ['train', 'val', 'test']
        existing_subsets = []

        # 检测结构1: dataset/train/images, dataset/train/labels
        for subset in possible_subsets:
            subset_path = self.dataset_path / subset
            if subset_path.exists():
                images_path = subset_path / 'images'
                labels_path = subset_path / 'labels'
                if images_path.exists() and labels_path.exists():
                    existing_subsets.append(subset)
                    self.dataset_structure = 'type1'  # 记录数据集结构类型
                    continue

        # 检测结构2: dataset/images/train, dataset/labels/train
        if not existing_subsets:
            images_path = self.dataset_path / 'images'
            labels_path = self.dataset_path / 'labels'
            if images_path.exists() and labels_path.exists():
                for subset in possible_subsets:
                    subset_images = images_path / subset
                    subset_labels = labels_path / subset
                    if subset_images.exists() and subset_labels.exists():
                        existing_subsets.append(subset)
                if existing_subsets:
                    self.dataset_structure = 'type2'  # 记录数据集结构类型

        return existing_subsets

    def get_existing_labels(self, subset: str) -> Set[int]:
        """获取子集中存在的所有标签类别"""
        if not hasattr(self, 'dataset_structure'):
            self.detect_subsets()

        if self.dataset_structure == 'type1':
            labels_path = self.dataset_path / subset / 'labels'
        else:  # type2
            labels_path = self.dataset_path / 'labels' / subset

        existing_labels = set()

        for label_file in labels_path.glob('*.txt'):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                class_id = int(parts[0])
                                existing_labels.add(class_id)
            except (ValueError, IndexError, UnicodeDecodeError):
                continue

        return existing_labels

    def find_image_file(self, label_file: Path, images_path: Path) -> Optional[Path]:
        """查找标签文件对应的图片文件"""
        for ext in self.image_extensions:
            potential_image = images_path / (label_file.stem + ext)
            if potential_image.exists():
                return potential_image
        return None

    def clean_subset(self, subset: str, labels_to_remove: Set[int],
                    backup: bool = True) -> Dict[str, int]:
        """清洗指定子集"""
        if not hasattr(self, 'dataset_structure'):
            self.detect_subsets()

        if self.dataset_structure == 'type1':
            images_path = self.dataset_path / subset / 'images'
            labels_path = self.dataset_path / subset / 'labels'
            subset_path = self.dataset_path / subset
        else:  # type2
            images_path = self.dataset_path / 'images' / subset
            labels_path = self.dataset_path / 'labels' / subset
            subset_path = self.dataset_path

        stats = {
            'processed_files': 0,
            'removed_labels': 0,
            'removed_images': 0,
            'renamed_files': 0,
            'negative_samples': 0
        }

        # 创建子集备份
        if backup:
            if self.dataset_structure == 'type1':
                subset_backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_backup" / subset
                # 复制当前子集到备份
                if subset_backup_path.exists():
                    shutil.rmtree(subset_backup_path)
                shutil.copytree(subset_path, subset_backup_path)
            else:  # type2
                # 对于type2结构，备份整个images和labels目录
                backup_root = self.dataset_path.parent / f"{self.dataset_path.name}_backup"
                backup_root.mkdir(parents=True, exist_ok=True)

                backup_images = backup_root / 'images' / subset
                backup_labels = backup_root / 'labels' / subset

                backup_images.parent.mkdir(parents=True, exist_ok=True)
                backup_labels.parent.mkdir(parents=True, exist_ok=True)

                if backup_images.exists():
                    shutil.rmtree(backup_images)
                if backup_labels.exists():
                    shutil.rmtree(backup_labels)

                shutil.copytree(images_path, backup_images)
                shutil.copytree(labels_path, backup_labels)

        label_files = list(labels_path.glob('*.txt'))

        print(f"\n处理 {subset} 集合，共 {len(label_files)} 个标签文件...")

        for label_file in label_files:
            stats['processed_files'] += 1

            # 检查是否为负样本（空标签文件）
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    original_lines = [line.strip() for line in f.readlines()]
            except Exception as e:
                print(f"警告: 读取标签文件失败 {label_file}: {e}")
                continue

            # 检查是否为负样本（空文件或只有空行）
            has_content = any(line.strip() for line in original_lines)
            if not has_content:
                stats['negative_samples'] += 1
                print(f"负样本，跳过: {label_file.name}")
                continue

            # 处理标签
            filtered_labels = []
            removed_in_file = 0

            for line in original_lines:
                if not line.strip():
                    continue

                try:
                    parts = line.split()
                    class_id = int(parts[0])

                    if class_id in labels_to_remove:
                        removed_in_file += 1
                        stats['removed_labels'] += 1
                    else:
                        filtered_labels.append(line)

                except (ValueError, IndexError) as e:
                    print(f"警告: 标签格式错误在 {label_file}: {line}")
                    continue

            # 如果过滤后没有标签了，删除图片和标签文件
            if not filtered_labels:
                image_file = self.find_image_file(label_file, images_path)
                if image_file:
                    print(f"删除空标签文件和对应图片: {label_file.name}")
                    label_file.unlink()
                    image_file.unlink()
                    stats['removed_images'] += 1
                else:
                    print(f"警告: 找不到对应图片，只删除标签文件: {label_file.name}")
                    label_file.unlink()
                    stats['removed_images'] += 1

            else:
                # 需要检查是否需要重新编号标签
                remaining_class_ids = set()
                for label in filtered_labels:
                    try:
                        class_id = int(label.split()[0])
                        if class_id not in labels_to_remove:
                            remaining_class_ids.add(class_id)
                    except (ValueError, IndexError):
                        continue

                # 检查是否需要重编号：只有当存在大于被删除标签的ID时才需要重编号
                needs_renumbering = any(
                    class_id > min(labels_to_remove)
                    for class_id in remaining_class_ids
                )

                if needs_renumbering or removed_in_file > 0:
                    # 创建映射关系：只对大于被删除标签的ID进行重新编号
                    # 例如：删除标签6，保留[1,4,5,7,8]，则7->6, 8->7
                    id_mapping = {}
                    for old_id in remaining_class_ids:
                        # 计算有多少个被删除的标签小于当前标签
                        offset = sum(1 for removed_id in labels_to_remove if removed_id < old_id)
                        new_id = old_id - offset
                        id_mapping[old_id] = new_id

                    # 重新编号并写入文件
                    renumbered_labels = []
                    for label in filtered_labels:
                        try:
                            parts = label.split()
                            old_id = int(parts[0])
                            if old_id in id_mapping:
                                new_id = id_mapping[old_id]
                                parts[0] = str(new_id)
                                renumbered_labels.append(' '.join(parts))
                        except (ValueError, IndexError):
                            continue

                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(renumbered_labels) + ('\n' if renumbered_labels else ''))

                    if removed_in_file > 0:
                        print(f"更新 {label_file.name}: 删除 {removed_in_file} 个标签，重新编号剩余标签")
                    else:
                        print(f"更新 {label_file.name}: 重新编号标签 {list(id_mapping.keys())} -> {list(id_mapping.values())}")
                    stats['renamed_files'] += 1

        return stats

    def clean_dataset(self, labels_to_remove: List[int], subsets: Optional[List[str]] = None,
                     backup: bool = True) -> Dict[str, Dict[str, int]]:
        """清洗整个数据集"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")

        # 自动检测子集
        if subsets is None:
            subsets = self.detect_subsets()

        if not subsets:
            raise ValueError("未找到有效的数据集子集 (train/val/test)")

        print(f"检测到子集: {', '.join(subsets)}")
        print(f"将要删除的标签: {labels_to_remove}")

        # 创建整体备份
        if backup:
            backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_backup"
            print(f"创建备份到: {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(self.dataset_path, backup_path)

        # 清洗各子集
        all_stats = {}
        labels_to_remove_set = set(labels_to_remove)

        for subset in subsets:
            subset_stats = self.clean_subset(subset, labels_to_remove_set, backup=False)
            all_stats[subset] = subset_stats

            # 打印子集统计
            print(f"\n{subset} 集合统计:")
            print(f"  处理文件数: {subset_stats['processed_files']}")
            print(f"  删除标签数: {subset_stats['removed_labels']}")
            print(f"  删除图片数: {subset_stats['removed_images']}")
            print(f"  重新编号文件数: {subset_stats['renamed_files']}")
            print(f"  负样本数: {subset_stats['negative_samples']}")

        return all_stats

    def preview_dataset(self, subsets: Optional[List[str]] = None) -> Dict[str, Set[int]]:
        """预览数据集信息"""
        if subsets is None:
            subsets = self.detect_subsets()

        dataset_info = {}

        print("=== 数据集预览 ===")
        print(f"数据集路径: {self.dataset_path}")
        print(f"检测到子集: {', '.join(subsets)}")

        for subset in subsets:
            existing_labels = self.get_existing_labels(subset)
            dataset_info[subset] = existing_labels

            # 统计每个类别的数量
            label_counts = defaultdict(int)
            labels_path = self.dataset_path / subset / 'labels'

            for label_file in labels_path.glob('*.txt'):
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    class_id = int(line.split()[0])
                                    label_counts[class_id] += 1
                                except (ValueError, IndexError):
                                    continue
                except:
                    continue

            print(f"\n{subset} 集合:")
            print(f"  存在的标签类别: {sorted(existing_labels)}")

            if label_counts:
                print("  标签数量统计:")
                for label_id in sorted(label_counts.keys()):
                    print(f"    类别 {label_id}: {label_counts[label_id]} 个标注")

        return dataset_info


# =================== 配置区域 - 在这里修改你的设置 ===================

# 数据集路径 - 请修改为你的实际数据集路径
DATASET_PATH = 'F:\wenw\work\dataset\dataset_no_game_6class_0113'

# 要删除的标签类别列表，例如: [1] 删除标签1， [1, 2] 删除标签1和标签2
LABELS_TO_REMOVE = [6]

# 要处理的子集，None表示自动检测所有存在的子集
# 可选值: ['train', 'val', 'test'] 或 None
SUBSETS_TO_PROCESS = None

# 是否创建备份
CREATE_BACKUP = True

# 运行模式
MODE = 'clean'  # 'preview' (预览) | 'dry_run' (试运行) | 'clean' (实际清洗)

# ======================================
# 常用配置示例:
#
# 示例1: 删除标签1(motorcycle)
# LABELS_TO_REMOVE = [1]
# MODE = 'clean'
#
# 示例2: 删除多个标签
# LABELS_TO_REMOVE = [1, 2]
# MODE = 'clean'
#
# 示例3: 只处理train和val子集
# SUBSETS_TO_PROCESS = ['train', 'val']
#
# 示例4: 不创建备份(谨慎使用)
# CREATE_BACKUP = False
# ======================================

# ================================================================

def main():
    print("=== YOLO数据集全面清洗工具 ===")
    print(f"数据集路径: {DATASET_PATH}")
    print(f"要删除的标签: {LABELS_TO_REMOVE}")
    print(f"运行模式: {MODE}")
    print(f"创建备份: {CREATE_BACKUP}")
    if SUBSETS_TO_PROCESS:
        print(f"处理子集: {SUBSETS_TO_PROCESS}")
    else:
        print("处理子集: 自动检测")

    try:
        cleaner = YOLODatasetCleaner(DATASET_PATH)

        # 预览模式
        if MODE == 'preview':
            cleaner.preview_dataset(SUBSETS_TO_PROCESS)
            return 0

        # 检测子集
        available_subsets = cleaner.detect_subsets()
        if not available_subsets:
            print("错误: 未找到有效的数据集子集")
            return 1

        # 显示数据集信息
        dataset_info = cleaner.preview_dataset(available_subsets)

        # 检查要删除的标签是否存在
        all_labels = set()
        for labels in dataset_info.values():
            all_labels.update(labels)

        labels_to_remove = [int(x) for x in LABELS_TO_REMOVE]
        invalid_labels = [label for label in labels_to_remove if label not in all_labels]

        if invalid_labels:
            print(f"警告: 数据集中不存在以下标签: {invalid_labels}")
            print(f"数据集中存在的标签: {sorted(all_labels)}")

        valid_labels_to_remove = [label for label in labels_to_remove if label in all_labels]
        if not valid_labels_to_remove:
            print("没有有效的标签可以删除")
            return 0

        print(f"\n实际将要删除的标签: {valid_labels_to_remove}")

        # 试运行模式
        if MODE == 'dry_run':
            print("\n=== 试运行模式 ===")
            print("以下是将要删除的内容统计:")

            total_removed_labels = 0
            total_removed_images = 0

            for subset in available_subsets:
                existing_labels = cleaner.get_existing_labels(subset)
                labels_to_remove_in_subset = set(valid_labels_to_remove) & existing_labels

                if not labels_to_remove_in_subset:
                    print(f"{subset}: 无需删除的标签")
                    continue

                # 模拟统计
                removed_in_subset = 0
                images_to_remove = 0

                if cleaner.dataset_structure == 'type1':
                    labels_path = cleaner.dataset_path / subset / 'labels'
                    images_path = cleaner.dataset_path / subset / 'images'
                else:  # type2
                    labels_path = cleaner.dataset_path / 'labels' / subset
                    images_path = cleaner.dataset_path / 'images' / subset

                for label_file in labels_path.glob('*.txt'):
                    try:
                        with open(label_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                    except:
                        continue

                    remaining_labels = []
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                class_id = int(line.split()[0])
                                if class_id not in labels_to_remove_in_subset:
                                    remaining_labels.append(line)
                                else:
                                    removed_in_subset += 1
                            except:
                                continue

                    if not remaining_labels:
                        image_file = cleaner.find_image_file(label_file, images_path)
                        if image_file:
                            images_to_remove += 1

                total_removed_labels += removed_in_subset
                total_removed_images += images_to_remove

                print(f"{subset}: 将删除 {removed_in_subset} 个标签，{images_to_remove} 张图片")

            print(f"\n总计: 将删除 {total_removed_labels} 个标签，{total_removed_images} 张图片")
            return 0

        # 实际执行清洗
        if MODE == 'clean':
            print(f"\n开始清洗数据集...")
            print("正在处理，请稍候...")

            try:
                all_stats = cleaner.clean_dataset(
                    labels_to_remove=valid_labels_to_remove,
                    subsets=SUBSETS_TO_PROCESS,
                    backup=CREATE_BACKUP
                )

                # 汇总统计
                total_removed_labels = sum(stats['removed_labels'] for stats in all_stats.values())
                total_removed_images = sum(stats['removed_images'] for stats in all_stats.values())
                total_processed = sum(stats['processed_files'] for stats in all_stats.values())
                total_negative_samples = sum(stats['negative_samples'] for stats in all_stats.values())
                total_renamed = sum(stats['renamed_files'] for stats in all_stats.values())

                print(f"\n=== 清洗完成 ===")
                print(f"总计处理文件数: {total_processed}")
                print(f"总计删除标签数: {total_removed_labels}")
                print(f"总计删除图片数: {total_removed_images}")
                print(f"重新编号文件数: {total_renamed}")
                print(f"负样本数量: {total_negative_samples}")

                if CREATE_BACKUP:
                    print(f"备份已创建到: {cleaner.dataset_path.parent / f'{cleaner.dataset_path.name}_backup'}")

                return 0

            except Exception as e:
                print(f"清洗过程中出现错误: {e}")
                return 1
        else:
            print(f"错误: 未知的运行模式 {MODE}")
            return 1

    except Exception as e:
        print(f"程序执行出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main())