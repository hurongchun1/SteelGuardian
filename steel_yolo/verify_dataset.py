# -*- coding: utf-8 -*-
"""
数据集验证脚本
检查YOLO数据集配置是否正确
"""

import os
import yaml
from pathlib import Path

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "dataset.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")


def verify_dataset():
    """验证数据集配置"""
    print("=" * 60)
    print("YOLO 数据集验证")
    print("=" * 60)
    
    # 1. 检查dataset.yaml是否存在
    if not os.path.exists(DATASET_YAML):
        print(f"[错误] 数据集配置文件不存在: {DATASET_YAML}")
        return False
    
    print(f"[✓] 数据集配置文件存在: {DATASET_YAML}")
    
    # 2. 读取配置文件
    with open(DATASET_YAML, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"[✓] 配置文件读取成功")
    print(f"    数据集路径: {config.get('path', '未设置')}")
    print(f"    训练集路径: {config.get('train', '未设置')}")
    print(f"    验证集路径: {config.get('val', '未设置')}")
    print(f"    测试集路径: {config.get('test', '未设置')}")
    
    # 3. 检查数据集根目录
    data_path = config.get('path', '')
    if not os.path.exists(data_path):
        print(f"[错误] 数据集根目录不存在: {data_path}")
        return False
    
    print(f"[✓] 数据集根目录存在: {data_path}")
    
    # 4. 检查训练集、验证集、测试集文件
    train_file = os.path.join(data_path, config.get('train', ''))
    val_file = os.path.join(data_path, config.get('val', ''))
    test_file = os.path.join(data_path, config.get('test', ''))
    
    files_to_check = [
        ("训练集", train_file),
        ("验证集", val_file),
        ("测试集", test_file),
    ]
    
    for name, file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"[✓] {name}文件存在: {file_path}")
            print(f"    包含 {len(lines)} 张图片路径")
            
            # 检查第一张图片是否存在
            if lines:
                first_img_path = os.path.join(data_path, lines[0].strip())
                if os.path.exists(first_img_path):
                    print(f"    第一张图片存在: {first_img_path}")
                else:
                    print(f"    [警告] 第一张图片不存在: {first_img_path}")
        else:
            print(f"[错误] {name}文件不存在: {file_path}")
            return False
    
    # 5. 检查图片和标签目录
    images_dir = os.path.join(data_path, "train", "images")
    labels_dir = os.path.join(data_path, "train", "labels")
    
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        print(f"[✓] 图片目录存在: {images_dir}")
        print(f"    包含 {len(image_files)} 张图片")
    else:
        print(f"[错误] 图片目录不存在: {images_dir}")
        return False
    
    if os.path.exists(labels_dir):
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        print(f"[✓] 标签目录存在: {labels_dir}")
        print(f"    包含 {len(label_files)} 个标签文件")
        
        # 检查图片和标签是否匹配
        # TODO: 在这里写你的代码
        # 提示：比较图片文件名和标签文件名是否匹配
        # 1. 提取图片文件名（不含扩展名）
        # 2. 提取标签文件名（不含扩展名）
        # 3. 找出没有对应标签的图片
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}
        
        missing_labels = image_basenames - label_basenames
        if missing_labels:
            print(f"[警告] 有 {len(missing_labels)} 张图片没有对应的标签文件")
        else:
            print(f"[✓] 所有图片都有对应的标签文件")
    else:
        print(f"[错误] 标签目录不存在: {labels_dir}")
        return False
    
    # 6. 检查类别配置
    names = config.get('names', {})
    if names:
        print(f"[✓] 类别配置:")
        for class_id, class_name in names.items():
            print(f"    {class_id}: {class_name}")
    else:
        print(f"[错误] 未配置类别")
        return False
    
    print()
    print("=" * 60)
    print("数据集验证完成！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    verify_dataset()