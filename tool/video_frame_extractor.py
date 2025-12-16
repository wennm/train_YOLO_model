#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频切割脚本
将视频按指定时间间隔保存为JPG图片
"""

import cv2
import os
import argparse
import time
from pathlib import Path


class VideoFrameExtractor:
    def __init__(self, video_path, output_dir, interval=1.0):
        """
        初始化视频帧提取器

        Args:
            video_path (str): 视频文件路径
            output_dir (str): 输出文件夹路径
            interval (float): 时间间隔（秒），默认为1秒
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval = interval

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 打开视频文件
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # 计算每隔多少帧提取一次
        self.frame_interval = int(self.fps * self.interval)

        print(f"视频信息:")
        print(f"  路径: {video_path}")
        print(f"  帧率: {self.fps:.2f} FPS")
        print(f"  总帧数: {self.total_frames}")
        print(f"  时长: {self.duration:.2f} 秒")
        print(f"  每隔 {interval} 秒提取一帧 (每 {self.frame_interval} 帧)")
        print(f"  输出目录: {output_dir}")
        print("-" * 50)

    def extract_frames(self):
        """
        提取视频帧并保存为JPG图片
        """
        frame_count = 0
        saved_count = 0
        last_saved_frame = -self.frame_interval

        # 获取视频文件名（不含扩展名）作为图片前缀
        video_name = Path(self.video_path).stem

        start_time = time.time()

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            current_time = frame_count / self.fps if self.fps > 0 else 0

            # 检查是否到了保存时间
            if frame_count - last_saved_frame >= self.frame_interval:
                # 生成文件名: 视频名_时间戳.jpg
                timestamp = f"{current_time:06.2f}"
                filename = f"{video_name}_{timestamp.replace('.', '')}.jpg"
                output_path = os.path.join(self.output_dir, filename)

                # 保存图片
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                saved_count += 1
                last_saved_frame = frame_count

                # 显示进度
                progress = (frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
                print(f"进度: {progress:.1f}% | 保存第 {saved_count} 帧: {filename} (时间: {current_time:.2f}s)")

            frame_count += 1

        elapsed_time = time.time() - start_time
        print("-" * 50)
        print(f"提取完成!")
        print(f"  总共处理帧数: {frame_count}")
        print(f"  保存图片数量: {saved_count}")
        print(f"  耗时: {elapsed_time:.2f} 秒")
        print(f"  平均每帧处理时间: {(elapsed_time/frame_count)*1000:.2f} ms" if frame_count > 0 else "")

    def release(self):
        """
        释放资源
        """
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='视频切割脚本 - 将视频按指定时间间隔保存为JPG图片')

    parser.add_argument('--video_path', type=str, default=r'F:\wenw\work\test_video\night.mp4', help='输入视频文件路径')
    parser.add_argument('--output_dir', type=str, default=r'F:\wenw\work\test_photo\fire', help='输出文件夹路径')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                       help='时间间隔（秒），默认为1秒')

    args = parser.parse_args()

    # 检查视频文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在: {args.video_path}")
        return

    # 检查间隔时间是否合理
    if args.interval <= 0:
        print(f"错误: 时间间隔必须大于0，当前值: {args.interval}")
        return

    try:
        # 创建提取器并执行提取
        extractor = VideoFrameExtractor(args.video_path, args.output_dir, args.interval)
        extractor.extract_frames()

    except Exception as e:
        print(f"错误: {str(e)}")

    finally:
        # 确保释放资源
        if 'extractor' in locals():
            extractor.release()


if __name__ == "__main__":
    # 直接使用main函数运行
    main()