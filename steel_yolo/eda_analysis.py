# -*- coding: utf-8 -*-
"""
钢铁表面缺陷数据集 EDA 分析
功能：读取标注文件、类别分布统计、标注框分析、可视化
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from pandas.core.arraylike import dispatch_reduction_ufunc

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LABELS_DIR = os.path.join(DATA_DIR, "train", "labels")
IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")

# 6种缺陷类别名称
CLASS_NAMES = {
    0: "crazing (龟裂)",
    1: "inclusion (夹杂)",
    2: "pitted_surface (点蚀)",
    3: "scratches (划痕)",
    4: "patches (斑块)",
    5: "rolled-in_scale (氧化铁皮压入)",
}


def load_all_labels(labels_dir):
    """
    第1步：读取所有标注文件
    
    功能：
    1. 找到 labels_dir 下所有 .txt 文件
    2. 每个文件逐行读取，解析出5个数字
    3. 存入字典 {文件名: [(类别ID, x, y, w, h), ...]}
    
    参数：
        labels_dir: 标注文件所在目录路径
    返回：
        annotations: 字典，key是文件名(不含后缀)，value是标注框列表
    """
    # 存储所有标注信息
    annotations = {}

    # 第1步 - 找到所有 .txt 标注文件
    label_files = glob.glob(os.path.join(labels_dir,"*.txt"))
    
    # 第2步 - 逐个读取每个文件
    for label_file in label_files:
    # 提取文件名（去掉路径和后缀），如 "D:/.../labels/0.txt" → "0"
    # os.path.basename(label_file):这样是获取到 0.txt
        basename = os.path.splitext(os.path.basename(label_file))[0]
    # 存储当前文件的所有标注框
        boxes = []
    # 第3步 - 解析每行的5个数字
        with open(label_file,"r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append((cls_id,x,y,w,h))  
    # 将当前文件的标注框列表存入字典
        annotations[basename] = boxes
    return annotations


def analyze_class_distribution(annotations):
    """
    第2步：分析类别分布
    统计6种缺陷各出现了多少次，每张图有几个标注框
    
    参数：
        annotations: load_all_labels 返回的字典
    返回：
        class_counts: Counter对象，{类别ID: 出现次数}
        boxes_per_image: 列表，每张图的标注框数量
    """
    class_counts = Counter()  # 统计每个类别出现了多少次
    boxes_per_image = []      # 记录每张图有几个标注框

    # TODO: 在这里写你的代码
    # 提示：
    # 1. 遍历 annotations 字典
    for basename,boxes in annotations.items():
        # 2. 对每张图，先把 len(boxes) 加到 boxes_per_image
        boxes_per_image.append(len(boxes))
        for cls_id,*_ in boxes:
            # 3. 再遍历每个标注框，class_counts[cls_id] += 1
            class_counts[cls_id] += 1
    # 打印结果
    print("=" * 60)
    print("类别分布统计")
    print("=" * 60)
    total = sum(class_counts.values())
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        ratio = count / total * 100
        print(f"  {CLASS_NAMES[cls_id]:30s} | {count:5d} ({ratio:5.1f}%)")
    print(f"  {'总计':30s} | {total:5d}")
    print(f"\n图片数量: {len(annotations)}")
    print(f"每张图的标注框数: 最少{min(boxes_per_image)}, 最多{max(boxes_per_image)}, "
          f"平均{sum(boxes_per_image)/len(boxes_per_image):.1f}")
    print()

    return class_counts, boxes_per_image


def analyze_box_sizes(annotations):
    """
    第3步：分析标注框尺寸分布
    统计所有缺陷框的宽度、高度、面积
    作用是：
    了解缺陷大小分布，如果大部分缺陷很小，模型检测就会困难需要调高输入分辨率
    发现异常标注，如果出现面积接近0或者1的框，可能是标注错误
    为了数据增强做参考，知道框的大小分布，可以设计合理的缩放、裁剪策略
    评估模型难度，框大小差异越大，模型学习越难
    
    参数：
        annotations: load_all_labels 返回的字典
    返回：
        widths: 列表，所有框的宽度
        heights: 列表，所有框的高度
        areas: 列表，所有框的面积
    """
    widths = []    # 存储所有标注框的宽度
    heights = []   # 存储所有标注框的高度
    areas = []     # 存储所有标注框的面积

    # TODO: 在这里写你的代码
    # 分析标注框尺寸的目的：
    # 1. 了解缺陷大小分布 - 小目标检测更难，可能需要调高输入分辨率
    # 2. 发现异常标注 - 面积接近0或1的框可能是标注错误
    # 3. 为数据增强做参考 - 知道框的大小分布，设计合理的缩放策略
    # 4. 评估模型难度 - 框大小差异越大，模型学习越难
    #
    # 提示：和第2步一样，双层 for 循环遍历 annotations
    # 对每个标注框(cls_id, x, y, w, h)：
    #   - 把 w 加到 widths 列表
    #   - 把 h 加到 heights 列表
    #   - 算出 area = w * h，加到 areas 列表
    for basename,boxes in annotations.items():
        for cls_id,x,y,w,h in boxes:
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    # 打印结果
    print("=" * 60)
    print("标注框尺寸分析 (归一化坐标)")
    print("=" * 60)
    print(f"  宽度: 最小{min(widths):.3f}, 最大{max(widths):.3f}, 平均{sum(widths)/len(widths):.3f}")
    print(f"  高度: 最小{min(heights):.3f}, 最大{max(heights):.3f}, 平均{sum(heights)/len(heights):.3f}")
    print(f"  面积: 最小{min(areas):.4f}, 最大{max(areas):.4f}, 平均{sum(areas)/len(areas):.4f}")
    print()

    return widths, heights, areas


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_output")


def plot_class_distribution(class_counts, output_dir):
    """
    第4步-1：绘制类别分布柱状图
    用 matplotlib 画一个柱状图，展示6种缺陷各有多少个
    """
    # 准备数据
    classes = sorted(class_counts.keys())
    names = [CLASS_NAMES[c].split(" ")[0] for c in classes]  # 取英文名，如 "crazing"
    counts = [class_counts[c] for c in classes]

    # TODO: 画柱状图
    # 提示：
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 5))
    # 画柱状图
    ax.bar(names, counts)
    # 加标题
    ax.set_title("各类别缺陷数量")
    ax.set_ylabel("标注框数量")
    plt.tight_layout()
    # 保存图片
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close()
    print(f"[OK] 类别分布图 -> {output_dir}/class_distribution.png")


def plot_box_analysis(widths, heights, areas, output_dir):
    """
    第4步-2：绘制标注框尺寸分析图
    画4个子图：宽度分布、高度分布、面积分布、宽高散点图
    """
    # TODO: 创建 2x2 的子图画布
    # 提示：fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig,axes = plt.subplots(2,2,figsize=(12,10))

    # TODO: 画4个子图
    # axes[0, 0].hist(widths, bins=30)   → 宽度分布直方图
    # axes[0, 1].hist(heights, bins=30)  → 高度分布直方图
    # axes[1, 0].hist(areas, bins=30)    → 面积分布直方图
    # axes[1, 1].scatter(widths, heights) → 宽度 vs 高度 散点图
    axes[0,0].hist(widths,bins=30)
    axes[0,1].hist(heights,bins=30)
    axes[1,0].hist(areas,bins=30)
    axes[1,1].scatter(widths,heights)
    # TODO: 保存图片
    # plt.savefig(os.path.join(output_dir, "box_analysis.png"), dpi=150)
    plt.savefig(os.path.join(output_dir,"box_analysis.png"),dpi=150)
    # plt.close()
    plt.close()
    # print(f"[OK] 标注框分析图 -> {output_dir}/box_analysis.png")
    print(f"[OK] 标注框分析图 -> {output_dir}/box_analysis.png")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("正在加载标注数据...")
    annotations = load_all_labels(LABELS_DIR)
    print(f"共读取 {len(annotations)} 个标注文件\n")

    class_counts, boxes_per_image = analyze_class_distribution(annotations)
    widths, heights, areas = analyze_box_sizes(annotations)

    print("生成可视化图表...")
    plot_class_distribution(class_counts, OUTPUT_DIR)
    plot_box_analysis(widths, heights, areas, OUTPUT_DIR)
    print(f"\n所有EDA分析结果已保存到: {OUTPUT_DIR}")
