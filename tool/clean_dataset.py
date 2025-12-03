#!/usr/bin/env python3
"""
YOLO数据集清洗脚本
功能：
1. 删除所有标签为1(motorcycle)的标注
2. 保留标签为0(person)的标注
3. 如果一张图片在删除motorcycle标签后没有其他标签，则删除该图片和对应的标签文件
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import argparse


def clean_yolo_dataset(dataset_path: str, backup: bool = True) -> Tuple[int, int, int]:
    """
    清洗YOLO数据集，删除motorcycle标签

    Args:
        dataset_path: YOLO数据集根目录路径
        backup: 是否创建备份

    Returns:
        (删除的motorcycle标签数量, 删除的图片数量, 处理的标签文件数量)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    # 统计信息
    removed_labels_count = 0
    removed_images_count = 0
    processed_files_count = 0

    # 支持的数据子目录
    subsets = ['train', 'val', 'test']

    # 创建备份
    if backup:
        backup_path = dataset_path.parent / f"{dataset_path.name}_backup"
        print(f"创建备份到: {backup_path}")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(dataset_path, backup_path)

    for subset in subsets:
        subset_path = dataset_path / subset
        if not subset_path.exists():
            print(f"子目录不存在，跳过: {subset_path}")
            continue

        images_path = subset_path / 'images'
        labels_path = subset_path / 'labels'

        if not images_path.exists() or not labels_path.exists():
            print(f"images或labels目录不存在，跳过: {subset}")
            continue

        # 获取所有标签文件
        label_files = list(labels_path.glob('*.txt'))

        print(f"\n处理 {subset} 集合，共 {len(label_files)} 个标签文件...")

        for label_file in label_files:
            processed_files_count += 1

            # 对应的图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
            image_file = None

            for ext in image_extensions:
                potential_image = images_path / (label_file.stem + ext)
                if potential_image.exists():
                    image_file = potential_image
                    break

            if not image_file:
                print(f"警告: 找不到对应的图片文件: {label_file.stem}")
                continue

            # 读取并处理标签文件
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"读取标签文件失败 {label_file}: {e}")
                continue

            # 过滤标签，只保留person(0)标签
            person_labels = []
            removed_in_file = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split()
                    class_id = int(parts[0])

                    if class_id == 0:  # person标签
                        person_labels.append(line)
                    elif class_id == 1:  # motorcycle标签
                        removed_in_file += 1
                        removed_labels_count += 1
                    # 其他类别的标签也保留（如果有的话）
                    elif class_id not in [0, 1]:
                        person_labels.append(line)
                        print(f"警告: 发现未知类别 {class_id} 在文件 {label_file}")

                except (ValueError, IndexError) as e:
                    print(f"警告: 标签格式错误在 {label_file}: {line}")
                    continue

            # 如果文件中只有motorcycle标签，删除图片和标签文件
            if len(person_labels) == 0:
                print(f"删除空标签文件和对应图片: {label_file.name}")
                label_file.unlink()
                image_file.unlink()
                removed_images_count += 1
            elif removed_in_file > 0:
                # 重写标签文件，只保留person标签
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(person_labels) + ('\n' if person_labels else ''))
                print(f"更新 {label_file.name}: 删除 {removed_in_file} 个motorcycle标签")

    return removed_labels_count, removed_images_count, processed_files_count


def main():
    parser = argparse.ArgumentParser(description='清洗YOLO数据集，删除motorcycle标签')
    parser.add_argument('--dataset_path', type=str, default='F:\wenw\work\dataset\Infrared_dataset_yolo', help='YOLO数据集根目录路径')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份')
    parser.add_argument('--dry-run', action='store_true', help='试运行模式，只显示将要删除的内容，不实际删除')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if args.dry_run:
        print("=== 试运行模式 ===")
        print("以下是将要删除的内容统计:")

        # 在试运行模式下，我们只统计但不实际删除
        removed_labels = 0
        removed_images = 0
        processed_files = 0

        subsets = ['train', 'val', 'test']
        for subset in subsets:
            subset_path = dataset_path / subset
            if not subset_path.exists():
                continue

            labels_path = subset_path / 'labels'
            images_path = subset_path / 'images'

            if not labels_path.exists() or not images_path.exists():
                continue

            label_files = list(labels_path.glob('*.txt'))

            for label_file in label_files:
                processed_files += 1

                # 检查对应的图片
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
                image_file = None
                for ext in image_extensions:
                    potential_image = images_path / (label_file.stem + ext)
                    if potential_image.exists():
                        image_file = potential_image
                        break

                if not image_file:
                    continue

                # 统计标签
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except:
                    continue

                person_labels = []
                motorcycle_count = 0

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        class_id = int(line.split()[0])
                        if class_id == 0:
                            person_labels.append(line)
                        elif class_id == 1:
                            motorcycle_count += 1
                            removed_labels += 1
                    except:
                        continue

                if len(person_labels) == 0:
                    removed_images += 1
                    print(f"将删除: {label_file.name} (只有motorcycle标签)")
                elif motorcycle_count > 0:
                    print(f"将更新: {label_file.name} (删除 {motorcycle_count} 个motorcycle标签)")

        print(f"\n=== 统计结果 ===")
        print(f"处理的标签文件数量: {processed_files}")
        print(f"将删除的motorcycle标签数量: {removed_labels}")
        print(f"将删除的图片数量: {removed_images}")

    else:
        # 实际执行
        try:
            backup = not args.no_backup
            print(f"开始清洗数据集: {dataset_path}")

            removed_labels, removed_images, processed_files = clean_yolo_dataset(
                dataset_path, backup=backup
            )

            print(f"\n=== 清洗完成 ===")
            print(f"处理的标签文件数量: {processed_files}")
            print(f"删除的motorcycle标签数量: {removed_labels}")
            print(f"删除的图片数量: {removed_images}")

            if backup:
                print(f"备份已创建到: {dataset_path.parent / f'{dataset_path.name}_backup'}")

        except Exception as e:
            print(f"清洗过程中出现错误: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())