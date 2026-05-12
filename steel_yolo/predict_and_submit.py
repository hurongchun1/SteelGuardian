# -*- coding: utf-8 -*-
"""
预测和生成提交文件脚本
使用训练好的模型预测测试集，生成submission.csv
"""

import os
import csv
import time
from pathlib import Path
from ultralytics import YOLO

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "dataset.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "runs", "steel_train", "weights", "best.pt")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "data", "test", "images")

# 类别名称
CLASS_NAMES = {
    0: "crazing", 1: "inclusion", 2: "pitted_surface",
    3: "scratches", 4: "patches", 5: "rolled-in_scale",
}


def predict_and_submit():
    """预测测试集并生成提交文件"""
    print("=" * 60)
    print("YOLO 预测与生成提交文件")
    print("=" * 60)
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先训练模型: python train.py --device 0 --epochs 100 --batch 16 --pretrained")
        return None
    
    print(f"[✓] 模型文件存在: {MODEL_PATH}")
    
    # 检查测试集目录
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"[错误] 测试集目录不存在: {TEST_IMAGES_DIR}")
        return None
    
    # 获取测试图片列表
    test_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                         if f.endswith(('.jpg', '.png'))])
    
    if not test_files:
        print("[错误] 测试集目录为空")
        return None
    
    print(f"[✓] 测试集包含 {len(test_files)} 张图片")
    
    # 加载模型
    print("[INFO] 加载模型...")
    model = YOLO(MODEL_PATH)
    
    # 预测参数
    conf_threshold = 0.25  # 置信度阈值
    iou_threshold = 0.45   # NMS IoU阈值
    
    print(f"[INFO] 开始预测 (conf={conf_threshold}, iou={iou_threshold})...")
    
    all_rows = []
    start_time = time.time()
    
    for idx, img_file in enumerate(test_files):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        image_id = int(os.path.splitext(img_file)[0])
        
        # 预测
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        
        # 处理结果
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            # TODO: 在这里处理检测结果
            # 提示：将检测结果转换为提交格式
            # 1. 遍历每个检测框
            # 2. 提取边界框坐标（x1, y1, x2, y2）
            # 3. 提取置信度和类别ID
            # 4. 添加到结果列表
            for box, conf_val, cls_id in zip(xyxy, confs, classes):
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                all_rows.append({
                    "image_id": image_id,
                    "bbox": str(bbox),
                    "category_id": int(cls_id),
                    "confidence": round(float(conf_val), 6),
                })
        
        # 进度显示
        if (idx + 1) % 50 == 0 or idx == len(test_files) - 1:
            elapsed = time.time() - start_time
            print(f"  进度: {idx + 1}/{len(test_files)}  "
                  f"检测框: {len(all_rows)}  耗时: {elapsed:.1f}s")
    
    # 生成提交文件
    output_path = os.path.join(BASE_DIR, "submission.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "bbox", "category_id", "confidence"])
        writer.writeheader()
        writer.writerows(all_rows)
    
    elapsed = time.time() - start_time
    
    # 统计信息
    print()
    print("=" * 60)
    print("预测完成!")
    print("=" * 60)
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  检测框总数: {len(all_rows)}")
    print(f"  平均每张图: {len(all_rows) / max(len(test_files), 1):.1f} 个框")
    print(f"  输出文件: {output_path}")
    
    # 按类别统计
    class_counter = {}
    for row in all_rows:
        cid = row["category_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        class_counter[cname] = class_counter.get(cname, 0) + 1
    
    if class_counter:
        print("\n  各类别检测数量:")
        for cname, count in sorted(class_counter.items(), key=lambda x: -x[1]):
            print(f"    {cname}: {count}")
    
    print(f"\n  --> 提交文件: {output_path}")
    return output_path


if __name__ == "__main__":
    predict_and_submit()