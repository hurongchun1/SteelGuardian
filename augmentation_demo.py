# -*- coding: utf-8 -*-
"""
钢铁表面缺陷数据集 - 数据增强可视化
功能：实现各种数据增强技术，展示增强效果，帮助理解为什么增强对缺陷检测很重要

运行方式：
    python augmentation_demo.py
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")
LABELS_DIR = os.path.join(DATA_DIR, "train", "labels")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "augmentation_output")

CLASS_NAMES = {
    0: "crazing (龟裂)",
    1: "inclusion (夹杂)",
    2: "pitted_surface (点蚀)",
    3: "scratches (划痕)",
    4: "patches (斑块)",
    5: "rolled-in_scale (氧化铁皮压入)",
}


# ========================================================================
# 第1步：实现基础增强函数
# 每个函数接收一张 PIL Image，返回增强后的 PIL Image
# ========================================================================

def augment_horizontal_flip(img):
    """
    水平翻转
    原理：img.transpose(Image.FLIP_LEFT_RIGHT)
    作用：缺陷可能出现在左右任意位置，翻转不改变缺陷特征
    """
    # TODO: 写一行代码实现水平翻转
    # 提示：使用 img.transpose() 方法，参数是 Image.FLIP_LEFT_RIGHT
    img = img.transpose(1)
    return img


def augment_vertical_flip(img):
    """
    垂直翻转
    原理：img.transpose(Image.FLIP_TOP_BOTTOM)
    作用：钢铁表面缺陷方向不固定，垂直翻转增加方向多样性
    """
    # TODO: 写一行代码实现垂直翻转
    # 提示：使用 img.transpose() 方法，参数是 Image.FLIP_TOP_BOTTOM
    img = img.transpose(2)
    return img


def augment_rotate_90(img):
    """
    旋转90度
    原理：img.rotate(90, expand=False)
    作用：划痕可以是水平/垂直/斜向，旋转让模型学习各种方向
    """
    img = img.rotate(90, expand=False)
    return img


def augment_rotate_random(img):
    """
    随机小角度旋转
    原理：生成 -20 到 20 之间的随机角度，然后 img.rotate(angle)
    作用：模拟相机轻微偏转
    注意：旋转后空白区域用灰色(128)填充，expand=False 保持尺寸不变
    """
    # TODO: 写代码实现随机旋转
    # 提示：
    # 1. angle = random.uniform(-20, 20) 生成随机角度
    # 2. img.rotate(angle, fillcolor=128, expand=False)
    angle = random.uniform(-20,20)
    img = img.rotate(angle,fillcolor=128,expand=False)
    return img


def augment_brightness(img):
    """
    亮度调整
    原理：使用 ImageEnhance.Brightness(img).enhance(factor)
    作用：模拟光照变化，工业相机光源不稳定
    注意：factor < 1 变暗，factor > 1 变亮
    """
    # TODO: 写代码实现亮度调整
    # 提示：
    # 1. factor = random.uniform(0.6, 1.4)  随机亮度因子
    # 2. ImageEnhance.Brightness(img).enhance(factor)
    factor = random.uniform(0.6, 1.4)
    img = ImageEnhance.Brightness(img).enhance(factor)
    return img


def augment_contrast(img):
    """
    对比度调整
    原理：使用 ImageEnhance.Contrast(img).enhance(factor)
    作用：有些缺陷与背景对比度很低，调整对比度可以增强缺陷可见性
    """
    # TODO: 写代码实现对比度调整
    # 提示：和亮度调整类似，用 ImageEnhance.Contrast
    # factor = random.uniform(0.6, 1.5)
    factor = random.uniform(0.6,1.5)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return img


def augment_gaussian_noise(img):
    """
    高斯噪声
    原理：将图片转为 numpy 数组，加上均值为0、标准差为15的高斯噪声，再转回 Image
    作用：模拟传感器噪声，增加模型鲁棒性
    """
    # TODO: 写代码实现高斯噪声
    # 提示：
    # 1. arr = np.array(img, dtype=np.float32)
    # 2. noise = np.random.normal(0, 15, arr.shape)  生成噪声
    # 3. arr = np.clip(arr + noise, 0, 255).astype(np.uint8)  加噪声并裁剪到0-255
    # 4. return Image.fromarray(arr)
    arr = np.array(img,dtype=np.float32)
    noise = np.random.normal(0,15,arr.shape)
    arr = np.clip(arr + noise,0,255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img


def augment_blur(img):
    """
    高斯模糊
    原理：img.filter(ImageFilter.GaussianBlur(radius=1.5))
    作用：模拟相机失焦或运动模糊，让模型不依赖清晰的边缘特征
    """
    # TODO: 写一行代码实现高斯模糊
    # 提示：使用 img.filter() 方法，参数是 ImageFilter.GaussianBlur(radius=1.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


def augment_sharpen(img):
    """
    锐化
    原理：img.filter(ImageFilter.SHARPEN)
    作用：增强缺陷边缘，使裂纹、划痕更清晰
    """
    # TODO: 写一行代码实现锐化
    # 提示：使用 img.filter() 方法，参数是 ImageFilter.SHARPEN
    img = img.filter(ImageFilter.SHARPEN)
    return img


def augment_random_erasing(img):
    """
    随机擦除
    原理：在图片上随机画1-3个灰色/彩色矩形方块，遮挡部分区域
    作用：迫使模型学习更多特征，防止过拟合（不能只看局部特征就判断）
    """
    # TODO: 写代码实现随机擦除
    # 提示：
    # 1. arr = np.array(img).copy()  复制一份（别改原图）
    # 2. h, w = arr.shape[:2]  获取高宽
    # 3. for _ in range(random.randint(1, 3)):  随机擦除1-3个区域
    #    - eh = random.randint(h // 8, h // 4)  擦除区域高度
    #    - ew = random.randint(w // 8, w // 4)  擦除区域宽度
    #    - ey = random.randint(0, h - eh)  随机y位置
    #    - ex = random.randint(0, w - ew)  随机x位置
    #    - arr[ey:ey + eh, ex:ex + ew] = random.randint(0, 255)  用随机颜色填充
    # 4. return Image.fromarray(arr)
    arr = np.array(img).copy()
    h,w = arr.shape[:2]
    for _ in range(random.randint(1,3)):
        eh = random.randint(h // 8,h // 4)
        ew = random.randint(w // 8, w // 4)
        ey = random.randint(0, h - eh)
        ex = random.randint(0, w - ew) 
        arr[ey:ey + eh, ex:ex + ew] = random.randint(0, 255)
    img = Image.fromarray(arr)
    return img


def augment_scale(img):
    """
    随机缩放
    原理：随机缩放图片到 0.7~1.3 倍，然后中心裁剪/填充回原始尺寸
    作用：模拟不同拍摄距离，缺陷在图中大小变化
    """
    w, h = img.size
    scale = random.uniform(0.7, 1.3)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = img.resize((new_w, new_h), Image.LANCZOS)

    if scale < 1:
        # 缩小了，用灰色底图居中粘贴
        result = Image.new(img.mode, (w, h), 128)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        result.paste(scaled, (paste_x, paste_y))
    else:
        # 放大了，中心裁剪回原始尺寸
        crop_x = (new_w - w) // 2
        crop_y = (new_h - h) // 2
        result = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))
    return result


# ========================================================================
# 第2步：Mosaic 数据增强
# YOLO 最核心的增强技术：将4张不同的图拼成1张
# ========================================================================

def augment_mosaic(images_dir, exclude_name=None):
    """
    Mosaic 数据增强：随机选4张图拼成2x2的大图

    优点：
    1. 一张训练图包含4种不同场景，大幅增加数据多样性
    2. 增强 BatchNorm 效果（相当于变相增大 batch size）
    3. 小目标出现概率提高，提升小目标检测能力
    """
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    if exclude_name:
        all_images = [f for f in all_images if f != exclude_name]
    if len(all_images) < 4:
        return None

    selected = random.sample(all_images, 4)
    target_size = 200
    mosaic = Image.new('RGB', (target_size * 2, target_size * 2))

    for i, img_name in enumerate(selected):
        img = Image.open(os.path.join(images_dir, img_name)).convert('RGB')
        img = img.resize((target_size, target_size), Image.LANCZOS)
        row, col = divmod(i, 2)
        mosaic.paste(img, (col * target_size, row * target_size))
    return mosaic


# ========================================================================
# 第3步：可视化 - 单张图片的增强效果展示
# ========================================================================

def demo_single_augmentations(image_path, output_dir):
    """
    展示同一张图片经过12种不同增强后的效果
    输出一张 3行4列 的大图
    """
    img = Image.open(image_path).convert('RGB')
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 定义所有增强方法: (名称, 函数, 说明文字)
    augmentations = [
        ("原图", None, "原始钢铁表面图片"),
        ("水平翻转", augment_horizontal_flip, "缺陷左右位置不固定"),
        ("垂直翻转", augment_vertical_flip, "钢铁缺陷方向不固定"),
        ("旋转90度", augment_rotate_90, "划痕可以是水平/垂直/斜向"),
        ("随机旋转", augment_rotate_random, "模拟相机轻微偏转(-20~+20度)"),
        ("亮度调整", augment_brightness, "模拟光照变化"),
        ("对比度调整", augment_contrast, "增强缺陷可见性"),
        ("高斯噪声", augment_gaussian_noise, "模拟传感器噪声"),
        ("模糊", augment_blur, "模拟相机失焦"),
        ("锐化", augment_sharpen, "增强缺陷边缘"),
        ("随机擦除", augment_random_erasing, "遮挡部分区域防止过拟合"),
        ("缩放变换", augment_scale, "模拟不同拍摄距离"),
    ]

    # 创建 3行4列 的子图画布
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"数据增强效果展示 - {basename}", fontsize=16, fontweight='bold')

    # 用 for 循环遍历 augmentations，对每个增强：
    for idx, (name, aug_fn, desc) in enumerate(augmentations):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        if aug_fn is None:
            display_img = img
        else:
            display_img = aug_fn(img.copy())
        ax.imshow(display_img)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(desc, fontsize=8, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"augmentation_demo_{basename}.png"), dpi=150)
    plt.close()
    print(f"[OK] 单图增强展示 -> {output_dir}/augmentation_demo_{basename}.png")


# ========================================================================
# 第4步：可视化 - Mosaic 增强效果展示
# ========================================================================

def demo_mosaic(images_dir, output_dir):
    """
    展示3组 Mosaic 增强效果（每组都是随机选4张图拼成的）
    """
    # 创建 1行3列 的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Mosaic 数据增强 - YOLO核心增强技术", fontsize=16, fontweight='bold')

    # 循环3次，每次调用 augment_mosaic() 生成一组 Mosaic 图并显示
    for i in range(3):
        mosaic = augment_mosaic(images_dir)
        if mosaic is not None:
            axes[i].imshow(mosaic)
            axes[i].set_title(f"Mosaic 样本 {i + 1}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])

    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mosaic_demo.png"), dpi=150)
    plt.close()
    print(f"[OK] Mosaic增强展示 -> {output_dir}/mosaic_demo.png")


# ========================================================================
# 第5步：可视化 - 随机增强组合对比
# ========================================================================

def demo_augmentation_combination(images_dir, output_dir):
    """
    展示同一张图经过多次随机增强组合后的效果
    说明：每次运行结果不同，体现增强的随机性
    """
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    if not all_images:
        print("[!] 图片目录为空")
        return

    # 随机选一张图
    img_name = random.choice(all_images)
    img = Image.open(os.path.join(images_dir, img_name)).convert('RGB')
    basename = os.path.splitext(img_name)[0]

    # 创建 2行5列 的子图
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"随机增强组合 - 同一张图每次增强结果不同 ({img_name})", fontsize=14, fontweight='bold')

    # 所有可用的增强函数列表
    all_augments = [
        augment_horizontal_flip, augment_vertical_flip,
        augment_rotate_random, augment_brightness, augment_contrast,
        augment_gaussian_noise, augment_blur, augment_random_erasing,
        augment_scale,
    ]

    # 第一行 - 原图 + 4个单次增强
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("原图", fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for i in range(1, 5):
        aug_fn = random.choice(all_augments)
        augmented = aug_fn(img.copy())
        axes[0, i].imshow(augmented)
        axes[0, i].set_title(f"单次增强 {i}", fontsize=10)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

    # 第二行 - 5个组合增强（连续应用2-4个增强）
    for i in range(5):
        combined = img.copy()
        n_augs = random.randint(2, 4)
        selected = random.sample(all_augments, n_augs)
        for aug_fn in selected:
            combined = aug_fn(combined)
        axes[1, i].imshow(combined)
        axes[1, i].set_title(f"组合增强 {i + 1}", fontsize=10)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"augmentation_combination_{basename}.png"), dpi=150)
    plt.close()
    print(f"[OK] 增强组合对比 -> {output_dir}/augmentation_combination_{basename}.png")


# ========================================================================
# 主函数
# ========================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 随机选一张示例图片
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    if not all_images:
        print("[!] 未找到图片，请检查路径")
        exit()

    sample_image = os.path.join(IMAGES_DIR, random.choice(all_images))
    print(f"示例图片: {sample_image}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    print("[1/3] 生成单图增强效果展示...")
    demo_single_augmentations(sample_image, OUTPUT_DIR)

    print("[2/3] 生成 Mosaic 增强展示...")
    demo_mosaic(IMAGES_DIR, OUTPUT_DIR)

    print("[3/3] 生成随机增强组合对比...")
    demo_augmentation_combination(IMAGES_DIR, OUTPUT_DIR)

    print()
    print("=" * 60)
    print(f"全部完成! 增强可视化结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)
