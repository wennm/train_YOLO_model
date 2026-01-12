import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from tqdm import tqdm
import numpy as np


def yolo_obb_to_xml(txt_path, xml_path, image_filename, image_width, image_height, image_depth=3, folder="图片样本"):
    """
    将YOLO OBB格式 (x1 y1 x2 y2 x3 y3 x4 y4) 转换为VOC robndbox格式 (cx cy w h angle)
    """

    annotation = ET.Element("annotation")
    annotation.set("verified", "no")

    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = image_filename
    ET.SubElement(annotation, "path").text = f"E:\\项目\\松山湖公安分局无人机自动巡检项目\\事故检测\\标注数据\\OBB\\250402\\图片样本\\{image_filename}"

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_width)
    ET.SubElement(size, "height").text = str(image_height)
    ET.SubElement(size, "depth").text = str(image_depth)
    ET.SubElement(annotation, "segmented").text = "0"

    # 读取txt文件
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            continue  # 不是旋转框格式

        cls = parts[0]
        coords = list(map(float, parts[1:]))

        # 归一化坐标转像素
        pts = np.array([
            [coords[0] * image_width, coords[1] * image_height],
            [coords[2] * image_width, coords[3] * image_height],
            [coords[4] * image_width, coords[5] * image_height],
            [coords[6] * image_width, coords[7] * image_height]
        ], dtype=np.float32)

        # 用opencv计算旋转矩形
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), angle = rect

        # 调整角度方向（OpenCV的角度定义是 -90°~0°）
        angle_rad = np.deg2rad(angle)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "type").text = "robndbox"
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        robndbox = ET.SubElement(obj, "robndbox")
        ET.SubElement(robndbox, "cx").text = f"{cx:.4f}"
        ET.SubElement(robndbox, "cy").text = f"{cy:.4f}"
        ET.SubElement(robndbox, "w").text = f"{w:.4f}"
        ET.SubElement(robndbox, "h").text = f"{h:.4f}"
        ET.SubElement(robndbox, "angle").text = f"{angle_rad:.6f}"  # 弧度形式

    xml_str = ET.tostring(annotation, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


if __name__ == "__main__":
    # txt_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\train\labels"
    # xml_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\train\xml"
    # image_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\train\images"
    txt_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\val\labels"
    xml_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\val\xml"
    image_dir = r"F:\wenw\work\dataset\dataset_no_game_4class_1212\val\images"

    os.makedirs(xml_dir, exist_ok=True)

    for file in tqdm(os.listdir(image_dir)):
        base_name = os.path.splitext(file)[0]
        txt_file = os.path.join(txt_dir, base_name + ".txt")
        xml_file = os.path.join(xml_dir, base_name + ".xml")
        image_path = os.path.join(image_dir, file)

        if not os.path.exists(txt_file):
            continue

        with Image.open(image_path) as img:
            w, h = img.size

        yolo_obb_to_xml(txt_file, xml_file, file, w, h)
