#!/usr/bin/env python3
"""
YOLO OBB数据集标签修改和重新排序工具

功能：
1. 将指定的标签ID改为另一个标签ID（例如将标签5改为标签0）
2. 自动重新排序其余标签，确保标签ID连续且从0开始

示例：将标签5改为标签0
   原标签：0,1,2,3,4,5,6,7
   修改后：0,1,2,3,4,0,5,6
   重新排序后：1,2,3,4,5,6,7,0
   最终映射：
   - 原0 -> 1
   - 原1 -> 2
   - 原2 -> 3
   - 原3 -> 4
   - 原4 -> 5
   - 原5 -> 0
   - 原6 -> 6
   - 原7 -> 7
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Set
import shutil


def parse_obb_line(line: str) -> tuple:
    """解析OBB格式的标签行"""
    parts = line.strip().split()
    if len(parts) != 9:  # class_id + 8 coordinates
        return None, None
    class_id = int(parts[0])
    coordinates = ' '.join(parts[1:])
    return class_id, coordinates


def build_label_mapping(source_label: int, target_label: int, total_classes: int) -> Dict[int, int]:
    """
    构建标签映射关系

    Args:
        source_label: 要修改的源标签ID
        target_label: 目标标签ID
        total_classes: 总类别数

    Returns:
        标签映射字典 {old_label: new_label}
    """
    # 首先将source_label改为target_label
    temp_labels = list(range(total_classes))
    temp_labels[source_label] = target_label

    # 收集所有唯一的标签并排序
    unique_labels = sorted(set(temp_labels))

    # 创建映射关系：保持相对顺序，但从0开始重新编号
    label_mapping = {}
    for old_label in range(total_classes):
        temp_value = temp_labels[old_label]
        new_label = unique_labels.index(temp_value)
        label_mapping[old_label] = new_label

    return label_mapping


def process_label_file(file_path: Path, label_mapping: Dict[int, int], dry_run: bool = False) -> Set[int]:
    """
    处理单个标签文件

    Args:
        file_path: 标签文件路径
        label_mapping: 标签映射字典
        dry_run: 是否只是演示，不实际修改

    Returns:
        该文件中出现的标签ID集合
    """
    new_lines = []
    found_labels = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                class_id, coordinates = parse_obb_line(line)
                if class_id is None:
                    # 格式不正确的行，保持原样
                    new_lines.append(line)
                    continue

                found_labels.add(class_id)

                # 应用标签映射
                new_class_id = label_mapping.get(class_id, class_id)
                new_lines.append(f"{new_class_id} {coordinates}")

        # 写入文件
        if not dry_run and new_lines:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines) + '\n')

        return found_labels

    except Exception as e:
        print(f"错误：处理文件 {file_path} 时出错: {e}")
        return set()


def process_dataset(dataset_path: str, source_label: int, target_label: int,
                   dry_run: bool = False, backup: bool = True,
                   update_yaml: bool = True) -> None:
    """
    处理整个数据集

    Args:
        dataset_path: 数据集路径
        source_label: 要修改的源标签ID
        target_label: 目标标签ID
        dry_run: 是否只是演示，不实际修改
        backup: 是否备份原始标签
        update_yaml: 是否更新dataset.yaml中的类别名称
    """
    dataset_path = Path(dataset_path)

    # 检查数据集路径
    if not dataset_path.exists():
        print(f"错误：数据集路径不存在: {dataset_path}")
        return

    # 读取dataset.yaml获取类别数
    yaml_path = dataset_path / "dataset.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
            total_classes = dataset_config.get('nc', 8)
            class_names = dataset_config.get('names', [])
    else:
        # 默认8个类别
        total_classes = 8
        class_names = []

    # 验证标签ID
    if source_label >= total_classes or target_label >= total_classes:
        print(f"错误：标签ID必须在0-{total_classes-1}范围内")
        return

    print(f"\n数据集信息：")
    print(f"  路径: {dataset_path}")
    print(f"  总类别数: {total_classes}")
    if class_names:
        print(f"  类别名称: {class_names}")
    print(f"\n操作：将标签 {source_label} 改为标签 {target_label}")
    print(f"       然后重新排序所有标签")

    # 构建标签映射
    label_mapping = build_label_mapping(source_label, target_label, total_classes)

    print(f"\n标签映射关系：")
    print(f"  原标签 -> 新标签")
    for old_label in sorted(label_mapping.keys()):
        new_label = label_mapping[old_label]
        old_name = class_names[old_label] if old_label < len(class_names) else ""
        new_name = class_names[new_label] if new_label < len(class_names) else ""
        if old_name or new_name:
            print(f"  {old_label} ({old_name}) -> {new_label} ({new_name})")
        else:
            print(f"  {old_label} -> {new_label}")

    # 备份标签目录
    splits = ['train', 'val']
    all_found_labels = set()

    for split in splits:
        labels_dir = dataset_path / split / 'labels'

        if not labels_dir.exists():
            print(f"\n警告：{split} 标签目录不存在: {labels_dir}")
            continue

        # 备份
        if backup and not dry_run:
            backup_dir = labels_dir.parent / f'{labels_dir.name}_backup'
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(labels_dir, backup_dir)
            print(f"\n已备份 {split} 标签到: {backup_dir}")

        # 处理所有标签文件
        print(f"\n处理 {split} 标签...")
        label_files = list(labels_dir.glob('*.txt'))

        for i, label_file in enumerate(label_files, 1):
            found_labels = process_label_file(label_file, label_mapping, dry_run)
            all_found_labels.update(found_labels)

            if i % 100 == 0:
                print(f"  已处理 {i}/{len(label_files)} 个文件...")

        print(f"  完成！共处理 {len(label_files)} 个文件")

    # 显示统计信息
    print(f"\n数据集中出现的标签: {sorted(all_found_labels)}")

    # 更新dataset.yaml
    if not dry_run and update_yaml and yaml_path.exists():
        new_class_names = []
        # 根据映射关系重新排列类别名称
        for new_label in range(total_classes):
            # 找到映射到这个新标签的旧标签
            for old_label, nl in label_mapping.items():
                if nl == new_label and old_label < len(class_names):
                    new_class_names.append(class_names[old_label])
                    break

        if new_class_names:
            print(f"\n新的类别名称: {new_class_names}")
            dataset_config['nc'] = total_classes
            dataset_config['names'] = new_class_names

            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(dataset_config, f,
                         allow_unicode=True,
                         default_flow_style=False,
                         sort_keys=False)
            print(f"已更新 dataset.yaml")


def main():
    parser = argparse.ArgumentParser(
        description='修改和重新排序YOLO OBB数据集标签',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 将标签5改为标签0，其余标签重新排序（自动备份和更新yaml）
  python modify_and_reorder_labels.py --dataset F:\\wenw\\work\\dataset\\dataset_no_game_8class_1231 --source 5 --target 0

  # 预览修改，不实际执行
  python modify_and_reorder_labels.py --dataset F:\\wenw\\work\\dataset\\dataset_no_game_8class_1231 --source 5 --target 0 --dry-run

  # 不备份原始标签
  python modify_and_reorder_labels.py --dataset F:\\wenw\\work\\dataset\\dataset_no_game_8class_1231 --source 5 --target 0 --no-backup

  # 不更新dataset.yaml
  python modify_and_reorder_labels.py --dataset F:\\wenw\\work\\dataset\\dataset_no_game_8class_1231 --source 5 --target 0 --no-update-yaml
        """
    )

    parser.add_argument('--dataset', '-d', type=str, default='F:\\wenw\\work\\dataset\\dataset_no_game_8class_1231',
                       help='数据集根目录路径')
    parser.add_argument('--source', '-s', type=int, default=4,
                       help='要修改的源标签ID')
    parser.add_argument('--target', '-t', type=int, default=0,
                       help='目标标签ID')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览修改，不实际修改文件')
    parser.add_argument('--no-backup', action='store_true',
                       help='不备份原始标签文件')
    parser.add_argument('--update-yaml', action='store_true', default=True,
                       help='自动更新dataset.yaml中的类别名称（默认启用）')
    parser.add_argument('--no-update-yaml', dest='update_yaml', action='store_false',
                       help='不更新dataset.yaml中的类别名称')

    args = parser.parse_args()

    print("=" * 80)
    print("YOLO OBB数据集标签修改和重新排序工具")
    print("=" * 80)

    if args.dry_run:
        print("\n[预览模式] 不会实际修改文件\n")

    process_dataset(
        dataset_path=args.dataset,
        source_label=args.source,
        target_label=args.target,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        update_yaml=args.update_yaml
    )

    if args.dry_run:
        print("\n[预览模式] 使用 --dry-run=False 或移除该参数来实际执行修改")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
