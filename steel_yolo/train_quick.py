# -*- coding: utf-8 -*-
"""
快速验证训练脚本
用于在CPU上快速验证训练流程是否正常
"""

import os
import torch
from ultralytics import YOLO

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "dataset.yaml")
YOLOV12_YAML = os.path.join(BASE_DIR, "yolov12.yaml")


def train_quick():
    """快速验证训练"""
    print("=" * 60)
    print("YOLO 快速验证训练 (CPU)")
    print("=" * 60)
    
    # 检查设备
    if torch.cuda.is_available():
        device = "0"
        print(f"[✓] 检测到GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[!] 未检测到GPU, 将使用CPU训练 (速度较慢)")
    
    # 加载模型
    print("[INFO] 加载YOLOv12模型...")
    model = YOLO(YOLOV12_YAML)
    
    # 训练参数
    print("[INFO] 开始训练 (1个epoch, 验证流程)...")
    
    # TODO: 在这里写你的训练参数
    # 提示：根据你的需求调整以下参数
    # 1. epochs: 训练轮数，快速验证用1，正式训练用100+
    # 2. batch: 批次大小，CPU用4，GPU用16
    # 3. imgsz: 输入图像尺寸，你的图片是200x200
    # 4. patience: 早停轮数，防止过拟合
    results = model.train(
        data=DATASET_YAML,
        epochs=1,  # 只训练1个epoch
        batch=4,   # CPU模式下使用小batch
        imgsz=200, # 你的图片尺寸
        patience=10,
        device=device,
        workers=0,  # Windows系统建议设为0
        save=True,
        verbose=True,
        plots=False,
        project=os.path.join(BASE_DIR, "runs_quick"),
        name="quick_train",
        exist_ok=True,
        
        # 数据增强（使用基础增强）
        # TODO: 在这里调整数据增强参数
        # 提示：数据增强可以提高模型泛化能力
        # 1. mosaic: 马赛克增强，将4张图拼成1张
        # 2. mixup: 混合增强，将2张图混合
        # 3. fliplr: 水平翻转，增加方向多样性
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
    )
    
    print()
    print("=" * 60)
    print("快速验证训练完成!")
    print("=" * 60)
    print(f"  模型保存在: runs_quick/quick_train/weights/best.pt")
    print()
    print("下一步:")
    print("  1. 如果训练成功, 可以进行正式训练")
    print("  2. 正式训练命令: python train.py --device 0 --epochs 100 --batch 16 --pretrained")
    print()
    
    return results


if __name__ == "__main__":
    train_quick()