# -*- coding: utf-8 -*-
"""
模型评估脚本
评估训练好的YOLO模型性能
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "dataset.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "runs", "steel_train", "weights", "best.pt")


def evaluate_model():
    """评估模型性能"""
    print("=" * 60)
    print("YOLO 模型评估")
    print("=" * 60)
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先训练模型: python train.py --device 0 --epochs 100 --batch 16 --pretrained")
        return None
    
    print(f"[✓] 模型文件存在: {MODEL_PATH}")
    
    # 加载模型
    print("[INFO] 加载模型...")
    model = YOLO(MODEL_PATH)
    
    # 在验证集上评估
    print("[INFO] 在验证集上评估模型...")
    results = model.val(
        data=DATASET_YAML,
        split="val",  # 使用验证集
        verbose=True,
    )
    
    # 打印评估结果
    print()
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    
    # 基础指标
    print(f"mAP50:      {results.box.map50:.4f}")
    print(f"mAP50-95:   {results.box.map:.4f}")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    
    # 各类别指标
    print()
    print("各类别指标:")
    print(f"{'类别':<20} {'mAP50':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)
    
    # 获取类别名称
    names = results.names
    for i, (cls_name, map50, precision, recall) in enumerate(zip(
        names.values(),
        results.box.ap50,
        results.box.p,
        results.box.r,
    )):
        print(f"{cls_name:<20} {map50:>8.4f} {precision:>10.4f} {recall:>8.4f}")
    
    # 保存评估结果
    # TODO: 在这里构建评估结果字典
    # 提示：将评估指标保存到字典中
    # 1. 保存整体指标：mAP50, mAP50-95, Precision, Recall
    # 2. 保存各类别指标
    eval_results = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "Precision": float(results.box.mp),
        "Recall": float(results.box.mr),
        "per_class": {}
    }
    
    # TODO: 在这里保存各类别指标
    # 提示：遍历每个类别，保存其mAP50、Precision、Recall
    for i, (cls_name, map50, precision, recall) in enumerate(zip(
        names.values(),
        results.box.ap50,
        results.box.p,
        results.box.r,
    )):
        eval_results["per_class"][cls_name] = {
            "mAP50": float(map50),
            "Precision": float(precision),
            "Recall": float(recall),
        }
    
    # 保存到JSON文件
    eval_json_path = os.path.join(BASE_DIR, "evaluation_results.json")
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"[✓] 评估结果已保存到: {eval_json_path}")
    
    # 提供建议
    print()
    print("=" * 60)
    print("性能分析与建议")
    print("=" * 60)
    
    if results.box.map50 > 0.7:
        print("✓ 模型性能良好 (mAP50 > 0.7)")
    elif results.box.map50 > 0.5:
        print("△ 模型性能一般 (0.5 < mAP50 < 0.7)")
        print("  建议: 尝试增加训练轮数或使用更强的数据增强")
    else:
        print("✗ 模型性能较差 (mAP50 < 0.5)")
        print("  建议: 检查数据集质量，增加训练数据，或调整模型结构")
    
    if results.box.mp > results.box.mr:
        print("✓ 精度高于召回率，模型倾向于保守检测")
    else:
        print("△ 召回率高于精度，模型倾向于激进检测")
        print("  建议: 提高置信度阈值以减少误检")
    
    return results


if __name__ == "__main__":
    evaluate_model()