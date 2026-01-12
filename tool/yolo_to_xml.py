import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def yolo_to_xml(txt_path, xml_path, image_filename, image_width, image_height, image_depth=3, folder="图片样本"):
    """
    将YOLO格式的txt文件转换为VOC格式的XML文件

    参数:
        txt_path: YOLO格式的txt文件路径
        xml_path: 要保存的XML文件路径
        image_filename: 图片文件名(不带路径)
        image_width: 图片宽度
        image_height: 图片高度
        image_depth: 图片通道数(默认3)
        folder: 文件夹名称(默认"图片样本")
    """
    # 创建XML根节点
    annotation = ET.Element("annotation")
    annotation.set("verified", "no")

    # 添加文件夹信息
    folder_elem = ET.SubElement(annotation, "folder")
    folder_elem.text = folder

    # 添加文件名
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = image_filename

    # 添加路径(示例路径，可根据需要修改)
    path_elem = ET.SubElement(annotation, "path")
    path_elem.text = f"E:\\项目\\松山湖公安分局无人机自动巡检项目\\事故检测\\标注数据\\OBB\\250402\\图片样本\\{image_filename}"

    # 添加source信息
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    # 添加图片尺寸信息
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_width)
    height = ET.SubElement(size, "height")
    height.text = str(image_height)
    depth = ET.SubElement(size, "depth")
    depth.text = str(image_depth)

    # 添加segmented信息
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # 读取YOLO格式的txt文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # 转换为绝对坐标
        abs_x_center = x_center * image_width
        abs_y_center = y_center * image_height
        abs_width = width * image_width
        abs_height = height * image_height

        # 创建object节点
        obj = ET.SubElement(annotation, "object")

        # 添加object信息
        type_elem = ET.SubElement(obj, "type")
        type_elem.text = "robndbox"

        name_elem = ET.SubElement(obj, "name")
        name_elem.text = class_id

        pose_elem = ET.SubElement(obj, "pose")
        pose_elem.text = "Unspecified"

        truncated_elem = ET.SubElement(obj, "truncated")
        truncated_elem.text = "0"

        difficult_elem = ET.SubElement(obj, "difficult")
        difficult_elem.text = "0"

        # 添加robndbox信息
        robndbox = ET.SubElement(obj, "robndbox")

        cx = ET.SubElement(robndbox, "cx")
        cx.text = f"{abs_x_center:.4f}"

        cy = ET.SubElement(robndbox, "cy")
        cy.text = f"{abs_y_center:.4f}"

        w = ET.SubElement(robndbox, "w")
        w.text = f"{abs_width:.4f}"

        h = ET.SubElement(robndbox, "h")
        h.text = f"{abs_height:.4f}"

        angle = ET.SubElement(robndbox, "angle")
        angle.text = "0"  # YOLO格式不包含角度，默认为0

    # 生成XML字符串并美化输出
    xml_str = ET.tostring(annotation, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # 保存XML文件
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


# 示例使用
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from PIL import Image

    txt_dir = r"E:\00lyh\04_数据处理\数据标注\交通事故\无人机视角整理0402-0708\labels"
    xml_dir = r"E:\00lyh\04_数据处理\数据标注\交通事故\无人机视角整理0402-0708\xml"
    image_dir = r"E:\00lyh\04_数据处理\数据标注\交通事故\无人机视角整理0402-0708\image"

    os.makedirs(xml_dir, exist_ok=True)

    for file in tqdm(os.listdir(image_dir)):
        base_name = os.path.splitext(file)[0]
        # 示例参数 - 请根据实际情况修改
        txt_file = os.path.join(txt_dir, base_name + ".txt")  # YOLO格式的txt文件
        xml_file = os.path.join(xml_dir, base_name + ".xml")  # 输出的XML文件
        image_name = os.path.join(image_dir, file)  # 图片文件名

        a = Image.open(image_name)

        img_width = a.width  # 图片宽度
        img_height = a.height  # 图片高度

        # 调用转换函数
        yolo_to_xml(txt_file, xml_file, image_name, img_width, img_height)
