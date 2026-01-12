#!/usr/bin/env python3
"""
红外图片处理脚本
功能：将三通道红外图片转换为灰度图，并反转灰度值
适用于红外图片中黑色人物转为白色的处理
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def preprocess_infrared_image(image_path, output_path=None):
    """
    处理红外图片：灰度转换 + 灰度反转

    Args:
        image_path (str): 输入图片路径
        output_path (str): 输出图片路径，如果为None则自动生成

    Returns:
        str: 输出图片路径
    """
    # 读取图片
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 读取原始图片（可能是三通道或单通道）
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 如果是三通道图片，先转为灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # 灰度反转：将黑色（低灰度值）转为白色（高灰度值）
    # 公式：inverted_pixel = 255 - original_pixel
    inverted_image = cv2.bitwise_not(gray_image)

    # 生成输出路径
    if output_path is None:
        input_path = Path(image_path)
        output_path = str(input_path.parent / f"{input_path.stem}_inverted{input_path.suffix}")

    # 保存处理后的图片
    cv2.imwrite(output_path, inverted_image)
    print(f"处理完成: {image_path} -> {output_path}")

    return output_path


def batch_process_infrared_images(input_dir, output_dir=None, suffix="_inverted"):
    """
    批量处理红外图片

    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径，如果为None则在输入目录下创建子目录
        suffix (str): 输出文件名后缀
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 设置输出目录
    if output_dir is None:
        output_path = input_path / "processed"
    else:
        output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    processed_count = 0
    total_files = 0

    # 遍历输入目录
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            total_files += 1
            try:
                # 生成输出路径
                output_file = output_path / f"{file_path.stem}{suffix}{file_path.suffix}"

                # 如果输出文件已存在，跳过
                if output_file.exists():
                    print(f"跳过已存在文件: {output_file}")
                    continue

                # 处理单张图片
                process_single_image(str(file_path), str(output_file))
                processed_count += 1

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

    print(f"\n处理完成! 共处理 {processed_count}/{total_files} 个文件")
    print(f"输出目录: {output_path}")


def process_single_image(image_path, output_path):
    """
    处理单张图片的内部函数
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 转换为灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # 灰度反转
    inverted_image = cv2.bitwise_not(gray_image)

    # 保存结果
    cv2.imwrite(output_path, inverted_image)


def main():
    parser = argparse.ArgumentParser(description="红外图片灰度转换和反转工具")
    parser.add_argument("-i", "--input", default="F:\\localsend_share\\infrared_before_1117_origin\\images", help="输入图片路径或目录路径")
    parser.add_argument("-o", "--output", default="F:\\localsend_share\\infrared_before_1117_origin\\images-inverted", help="输出图片路径或目录路径")
    parser.add_argument("-b", "--batch", action="store_true", help="批量处理模式")
    parser.add_argument("-s", "--suffix", default="_inverted", help="批量处理时的文件名后缀")

    args = parser.parse_args()

    try:
        # 检查输入是文件还是目录
        if os.path.isfile(args.input):
            # 输入是文件，使用单文件处理
            output_path = preprocess_infrared_image(args.input, args.output)
            print(f"处理完成: {output_path}")
        elif os.path.isdir(args.input):
            # 输入是目录，自动使用批量处理模式
            print("检测到输入是目录，使用批量处理模式...")
            batch_process_infrared_images(args.input, args.output, args.suffix)
        else:
            raise FileNotFoundError(f"输入路径不存在: {args.input}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())