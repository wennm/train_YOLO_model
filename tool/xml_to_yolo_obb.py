import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def visualize_obb_labels(image_path, label_path, class_names=None, output_dir=None):
    """
    可视化YOLO OBB标签
    参数:
        image_path: 图像文件路径
        label_path: 对应的标签文件路径
        class_names: 类别名称列表（如['car', 'person']），若为None则显示类别ID
        output_dir: 保存可视化结果的目录，若为None则仅显示不保存
    """
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # 读取标签
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 绘制每个旋转框
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        class_id = int(parts[0])
        points = list(map(float, parts[1:]))

        # 将归一化坐标转换为像素坐标
        pixel_points = []
        for i in range(0, len(points), 2):
            x = int(points[i] * img_w)
            y = int(points[i + 1] * img_h)
            pixel_points.append([x, y])

        # 转换为NumPy数组并reshape为多边形
        polygon = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))

        # 绘制旋转框（多边形）
        cv2.polylines(image, [polygon], isClosed=True,
                      color=(255, 0, 0), thickness=2)

        # 显示类别标签
        label = class_names[class_id] if class_names else str(class_id)
        cv2.putText(image, label, (pixel_points[0][0], pixel_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示或保存结果
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"vis_{Path(image_path).name}"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"可视化结果已保存至: {output_path}")
    else:
        plt.show()


def xml_to_yolo_obb(xml_path, output_dir, class_mapping=None):
    """
    将roLabelImg XML格式转换为YOLO OBB格式
    参数:
        xml_path: XML文件路径
        output_dir: 输出目录
        class_mapping: 类别名称到ID的映射字典（如{'car':0}），若为None则使用XML中的name作为类别
    返回:
        dict: 转换统计信息 {'bndbox': N, 'robndbox': M}
    """
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find('size')
    if size is None:
        raise ValueError(f"XML文件缺少size标签: {xml_path}")

    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # 准备输出内容和统计
    yolo_lines = []
    stats = {'bndbox': 0, 'robndbox': 0}

    # 遍历所有object标签
    for obj in root.findall('object'):
        # 获取类别
        class_name = obj.find('name').text

        # 检查类别是否在映射中
        if class_mapping and class_name not in class_mapping:
            print(f"警告: 类别 '{class_name}' 不在class_mapping中，文件: {Path(xml_path).name}")
            continue

        class_id = class_mapping[class_name] if class_mapping else class_name

        # 检查object类型
        obj_type = obj.find('type')
        if obj_type is not None and obj_type.text == 'bndbox':
            # 处理水平框（bndbox）
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            # 读取水平框坐标
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 转换为OBB格式（4个角点，角度为0）
            # 顺序：左下(xmin,ymin) -> 右下(xmax,ymin) -> 右上(xmax,ymax) -> 左上(xmin,ymax)
            points = [
                [xmin, ymin],  # 左下
                [xmax, ymin],  # 右下
                [xmax, ymax],  # 右上
                [xmin, ymax]   # 左上
            ]

            # 归一化坐标并构建YOLO OBB行
            normalized_coords = []
            for x, y in points:
                normalized_coords.extend([
                    x / img_width,
                    y / img_height
                ])

            yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
            yolo_lines.append(yolo_line)
            stats['bndbox'] += 1  # 统计水平框数量

        else:
            # 处理旋转框（robndbox）
            robndbox = obj.find('robndbox')
            if robndbox is None:
                # 如果既不是bndbox也没有robndox，跳过
                continue

            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)

            # 转换为YOLO OBB格式（归一化坐标）
            # YOLO OBB格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
            # 计算旋转矩形的四个角点（未旋转时的初始坐标）
            half_w = w / 2
            half_h = h / 2

            # 四个角点的相对坐标（以中心点为原点）
            points = [
                [-half_w, -half_h],  # 左下
                [half_w, -half_h],  # 右下
                [half_w, half_h],  # 右上
                [-half_w, half_h]  # 左上
            ]

            # 旋转点（绕中心点旋转）
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            normalized_coords = []
            for x, y in points:
                # 旋转后的坐标
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                # 转换为图像坐标系（中心点偏移）
                x_img = (cx + x_rot) / img_width
                y_img = (cy + y_rot) / img_height
                normalized_coords.extend([x_img, y_img])

            # 构建YOLO OBB行
            yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
            yolo_lines.append(yolo_line)
            stats['robndbox'] += 1  # 统计旋转框数量

    # 写入输出文件
    if yolo_lines:
        output_path = Path(output_dir) / (Path(xml_path).stem + '.txt')
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines))

    return stats


if __name__ == "__main__":

    # 1、xml转换为yolo-obb
    #
    # # 配置参数
    # xml_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\val\xml"  # 替换为你的XML文件夹路径
    # output_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\val\labels"  # 输出目录
    # class_mapping = {"0": 0,"1": 1,"2": 2,"3": 3,"4": 4,"5": 5,"6": 6,"7": 7}  # 类别映射（根据你的数据集调整）
    xml_dir = r"F:\wenw\work\dataset\buchong\traffic1229\xml"  # 替换为你的XML文件夹路径
    output_dir = r"F:\wenw\work\dataset\buchong\traffic1229\labels"  # 输出目录
    class_mapping = {"0": 0,"1": 1,"2": 2,"3": 3,"4": 4,"5": 5,"6": 6,"7": 7}  # 类别映射（根据你的数据集调整）

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 记录失败的文件和统计信息
    failed_files = []
    success_count = 0
    total_bndbox = 0
    total_robndbox = 0

    # 处理所有XML文件
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"总共找到 {len(xml_files)} 个XML文件\n")

    for xml_file in tqdm(xml_files, desc="Converting XML to YOLO OBB"):
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            stats = xml_to_yolo_obb(xml_path, output_dir, class_mapping)
            success_count += 1
            total_bndbox += stats['bndbox']
            total_robndbox += stats['robndbox']
        except Exception as e:
            error_msg = f"{xml_file}: {str(e)}"
            failed_files.append(error_msg)
            print(f"\n错误: {error_msg}")
            continue

    # 打印统计信息
    print(f"\n" + "=" * 70)
    print(f"转换完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {len(failed_files)} 个文件")
    print(f"\n转换的标注数量:")
    print(f"  水平框（bndbox）: {total_bndbox} 个")
    print(f"  旋转框（robndbox）: {total_robndbox} 个")
    print(f"  总计: {total_bndbox + total_robndbox} 个标注")
    print(f"\n结果保存在: {output_dir}")

    if failed_files:
        print(f"\n失败的文件列表:")
        for i, error in enumerate(failed_files, 1):
            print(f"  {i}. {error}")

        # 保存失败文件列表到日志
        log_file = os.path.join(output_dir, "conversion_errors.log")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("XML转换失败文件列表\n")
            f.write("=" * 70 + "\n\n")
            for error in failed_files:
                f.write(error + "\n")
        print(f"\n错误日志已保存到: {log_file}")

    print("=" * 70)

    # 2、 可视化某个转换后的文件以验证准确性
    # image_path = r"C:\Users\laiwe\Desktop\temp\frame_2025-04-10_00-00-00-076.jpg"
    # label_path = r"C:\Users\laiwe\Desktop\temp\frame_2025-04-10_00-00-00-076.txt"
    # visualize_obb_labels(image_path, label_path)