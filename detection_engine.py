# -*- coding: utf-8 -*-
"""
钢铁缺陷检测引擎
================
封装YOLO推理接口，提供统一的检测结果格式

功能:
    1. 加载训练好的YOLO模型 (best.pt)
    2. 对钢铁表面图片进行缺陷检测
    3. 返回检测结果（缺陷位置、类别、置信度）
    4. 在图片上绘制检测框

使用方式:
    from detection_engine import create_detector

    # 创建检测器（自动加载模型）
    detector = create_detector()

    # 检测图片
    result = detector.detect("test.jpg")

    # 查看结果
    print(result.summary())

    # 获取标注后的图片
    annotated_img = result.annotated_image

依赖:
    - ultralytics (YOLO)
    - Pillow (图片处理)
    - numpy (数组操作)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ========================================================================
# YOLOv12 AAttn 兼容性补丁
# ========================================================================
# 
# 为什么需要这个补丁?
# -------------------
# YOLOv12 使用了一种叫 AAttn (Attention-centric Attention) 的注意力机制。
# 不同版本的 ultralytics 库对 AAttn 的实现有两种格式:
#   - pip官方版: 使用 self.qkv (合并的 QKV 矩阵)
#   - YOLOv12源码版: 使用 self.qk + self.v (分离的 QK 和 V 矩阵)
#
# 这个补丁的作用是: 在加载模型时，自动检测并兼容这两种格式，
# 确保无论用哪种方式训练的模型都能正常加载和推理。
#
def _patch_aattn_forward():
    """
    修补 AAttn.forward 方法，兼容 qkv 和 qk+v 两种权重格式
    
    工作原理:
    1. 尝试导入 ultralytics 的 AAttn 模块
    2. 保存原始的 forward 方法
    3. 用兼容版本替换 forward 方法
    4. 兼容版本会自动检测权重格式并选择正确的计算方式
    """
    try:
        from ultralytics.nn.modules.block import AAttn
    except ImportError:
        # 如果导入失败（版本不支持），直接返回，不影响后续使用
        return

    # 保存原始的 forward 方法
    _original_forward = AAttn.forward

    def _compatible_forward(self, x):
        """
        兼容版本的 forward 方法
        
        参数:
            self: AAttn 模块实例
            x: 输入特征图，形状为 [B, C, H, W]
                B: batch size (批次大小)
                C: 通道数
                H: 特征图高度
                W: 特征图宽度
        """
        # 检测是否是 qkv 格式（官方版）
        if hasattr(self, 'qkv'):
            # 如果有 qkv 属性，使用原始方法
            return _original_forward(self, x)

        # 否则是 qk+v 格式（源码版），手动实现计算
        import torch
        
        # 获取输入形状
        B, C, H, W = x.shape  # B=批次, C=通道, H=高度, W=宽度
        N = H * W  # N = 特征图的空间位置数

        # 计算 QK 和 V
        # qk: Query-Key 矩阵，用于计算注意力
        # v: Value 矩阵，包含实际的特征信息
        qk = self.qk(x).flatten(2).transpose(1, 2)  # [B, C*2, H, W] -> [B, N, C*2]
        v = self.v(x).flatten(2).transpose(1, 2)    # [B, C, H, W] -> [B, N, C]

        # 处理区域注意力（如果 area > 1）
        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape

        # 分离 Q 和 K
        # Q: Query (查询)，用于"询问"其他位置的信息
        # K: Key (键)，用于"回答"其他位置的询问
        q, k = (
            qk.view(B, N, self.num_heads, self.head_dim * 2)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        # 计算注意力分数
        # attn = softmax(Q @ K^T / sqrt(head_dim))
        # 这是标准的 Transformer 注意力计算公式
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)  # 归一化
        
        # 应用注意力到 V
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        # 恢复区域注意力的形状
        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        # 恢复原始的空间形状
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 添加位置编码 (pe = positional encoding)
        # 位置编码帮助模型理解特征图中不同位置的空间关系
        x = x + self.pe(v)
        
        # 输出投影
        return self.proj(x)

    # 用兼容版本替换原始方法
    AAttn.forward = _compatible_forward


# 应用兼容性补丁（模块加载时自动执行）
_patch_aattn_forward()


# ========================================================================
# 缺陷类别定义
# ========================================================================
# 
# 钢铁表面缺陷的6种类别:
# 1. crazing (龟裂): 表面呈现网状细小裂纹
# 2. inclusion (夹杂): 表面有异物嵌入或杂质
# 3. pitted_surface (点蚀): 表面有凹坑或麻点
# 4. scratches (划痕): 表面有线状刮痕
# 5. patches (斑块): 表面有不规则色斑或区域变色
# 6. rolled-in_scale (氧化铁皮压入): 表面有氧化皮被压入的痕迹
#

# 英文类别名称 (模型输出使用)
CLASS_NAMES = {
    0: "crazing",           # 龟裂
    1: "inclusion",         # 夹杂
    2: "pitted_surface",    # 点蚀
    3: "scratches",         # 划痕
    4: "patches",           # 斑块
    5: "rolled-in_scale",   # 氧化铁皮压入
}

# 中文类别名称 (界面显示使用)
CLASS_NAMES_CN = {
    0: "龟裂",
    1: "夹杂",
    2: "点蚀",
    3: "划痕",
    4: "斑块",
    5: "氧化铁皮压入",
}

# 每个类别的显示颜色 (RGB格式)
# 用于在检测结果图片上绘制不同颜色的边界框
CLASS_COLORS = {
    0: (255, 0, 0),      # 红色 - 龟裂
    1: (0, 255, 0),      # 绿色 - 夹杂
    2: (0, 0, 255),      # 蓝色 - 点蚀
    3: (255, 255, 0),    # 黄色 - 划痕
    4: (255, 0, 255),    # 紫色 - 斑块
    5: (0, 255, 255),    # 青色 - 氧化铁皮压入
}

# 英文类别名 -> 类别ID 的反向映射
# 用于将VLM等模型输出的文本标签转换为类别ID
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


# ========================================================================
# 数据类 (Data Classes)
# ========================================================================
# 
# 使用 Python 的 dataclass 装饰器定义数据类，用于存储检测结果。
# dataclass 的好处是自动生成 __init__、__repr__ 等方法，代码更简洁。
#

@dataclass
class DetectionBox:
    """
    单个检测框
    
    表示模型检测到的一个缺陷区域，包含:
    - 类别信息 (是什么缺陷)
    - 置信度 (模型有多确定)
    - 位置坐标 (缺陷在哪里)
    """
    class_id: int           # 类别ID (0-5)
    class_name: str         # 类别名称 (英文，如 "crazing")
    confidence: float       # 置信度 (0-1之间的浮点数，越大越确定)
    
    # 边界框坐标 (xyxy格式，像素坐标)
    # (x1, y1) 是左上角，(x2, y2) 是右下角
    x1: float               # 左上角 x 坐标
    y1: float               # 左上角 y 坐标
    x2: float               # 右下角 x 坐标
    y2: float               # 右下角 y 坐标

    @property
    def cn_name(self) -> str:
        """获取中文类别名称"""
        return CLASS_NAMES_CN.get(self.class_id, self.class_name)

    def to_dict(self) -> dict:
        """转换为字典格式（便于JSON序列化）"""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "cn_name": self.cn_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(self.x1, 1), round(self.y1, 1),
                     round(self.x2, 1), round(self.y2, 1)],
        }


@dataclass
class DetectionResult:
    """
    检测结果
    
    包含一张图片的所有检测信息:
    - 图片路径
    - 检测到的所有缺陷框
    - 推理时间
    - 标注后的图片
    """
    image_path: str                              # 被检测的图片路径
    boxes: List[DetectionBox] = field(default_factory=list)  # 检测框列表
    inference_time: float = 0.0                  # 推理耗时（秒）
    annotated_image: Optional[np.ndarray] = None # 标注后的图片（numpy数组）

    @property
    def has_defect(self) -> bool:
        """是否有检测到缺陷"""
        return len(self.boxes) > 0

    @property
    def defect_count(self) -> int:
        """检测到的缺陷数量"""
        return len(self.boxes)

    @property
    def defect_types(self) -> List[str]:
        """检测到的缺陷类型列表（去重）"""
        return list(set(b.class_name for b in self.boxes))

    def to_dict(self) -> dict:
        """转换为字典格式（便于JSON序列化和保存）"""
        return {
            "image_path": self.image_path,
            "has_defect": self.has_defect,
            "defect_count": self.defect_count,
            "defect_types": self.defect_types,
            "boxes": [b.to_dict() for b in self.boxes],
            "inference_time": round(self.inference_time, 3),
        }

    def summary(self) -> str:
        """
        生成检测结果的文字摘要
        
        返回示例:
            检测到 3 个缺陷区域:
              1. 龟裂 (crazing) 置信度: 95.23%
              2. 划痕 (scratches) 置信度: 87.65%
              3. 斑块 (patches) 置信度: 76.54%
        """
        if not self.has_defect:
            return "未检测到缺陷"
        
        lines = [f"检测到 {self.defect_count} 个缺陷区域:"]
        for i, box in enumerate(self.boxes, 1):
            lines.append(
                f"  {i}. {box.cn_name} ({box.class_name}) "
                f"置信度: {box.confidence:.2%}"
            )
        return "\n".join(lines)


# ========================================================================
# YOLO 检测器
# ========================================================================

class YOLODetector:
    """
    YOLO目标检测器
    
    这是检测引擎的核心类，负责:
    1. 加载YOLO模型
    2. 对图片进行推理
    3. 解析检测结果
    4. 绘制检测框
    """
    
    def __init__(self, model_path: str):
        """
        初始化检测器，加载YOLO模型
        
        参数:
            model_path: 模型文件路径，通常是 "models/best.pt"
        """
        from ultralytics import YOLO

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载YOLO模型
        print(f"[YOLO] 加载模型: {model_path}")
        self.model = YOLO(model_path)  # ultralytics 的 YOLO 类
        self.model_path = model_path
        print(f"[YOLO] 模型加载成功")

    def detect(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        draw_boxes: bool = True,
    ) -> DetectionResult:
        """
        检测单张图片
        
        参数:
            image_path: 图片文件路径
            conf: 置信度阈值，低于此值的检测框会被过滤 (默认0.25)
            iou: NMS的IoU阈值，用于去除重叠的检测框 (默认0.45)
            draw_boxes: 是否在图片上绘制检测框 (默认True)
        
        返回:
            DetectionResult: 检测结果对象
        
        异常:
            FileNotFoundError: 图片文件不存在时抛出
        """
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        # 记录开始时间
        start_time = time.time()

        # 调用YOLO模型进行预测
        # results 是一个列表，每个元素对应一张图片的检测结果
        results = self.model.predict(
            source=image_path,      # 输入图片路径
            conf=conf,              # 置信度阈值
            iou=iou,                # NMS IoU阈值
            verbose=False,          # 不打印详细信息
        )

        # 计算推理耗时
        elapsed = time.time() - start_time

        # 解析检测结果
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            # results[0].boxes 包含所有检测框的信息
            det_boxes = results[0].boxes
            
            # 提取各类信息
            xyxy = det_boxes.xyxy.cpu().numpy()      # 边界框坐标 [N, 4]
            confs = det_boxes.conf.cpu().numpy()      # 置信度 [N]
            classes = det_boxes.cls.cpu().numpy()     # 类别ID [N]

            # 遍历每个检测框，创建 DetectionBox 对象
            for box, c, cls_id in zip(xyxy, confs, classes):
                cls_id = int(cls_id)
                boxes.append(DetectionBox(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    confidence=float(c),
                    x1=float(box[0]),  # 左上角 x
                    y1=float(box[1]),  # 左上角 y
                    x2=float(box[2]),  # 右下角 x
                    y2=float(box[3]),  # 右下角 y
                ))

        # 创建检测结果对象
        result = DetectionResult(
            image_path=image_path,
            boxes=boxes,
            inference_time=elapsed,
        )

        # 如果需要，绘制检测框
        if draw_boxes:
            result.annotated_image = self._draw_boxes(image_path, boxes)

        return result

    def _draw_boxes(
        self,
        image_path: str,
        boxes: List[DetectionBox],
        font_size: int = 12,
    ) -> np.ndarray:
        """
        在图片上绘制检测框和标签
        
        参数:
            image_path: 原始图片路径
            boxes: 检测框列表
            font_size: 标签字体大小
        
        返回:
            np.ndarray: 绘制了检测框的图片（RGB格式的numpy数组）
        """
        # 打开图片并转换为RGB格式
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 尝试加载字体（优先使用中文字体）
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                # 尝试加载黑体（支持中文）
                font = ImageFont.truetype("simhei.ttf", font_size)
            except (OSError, IOError):
                # 使用默认字体
                font = ImageFont.load_default()

        # 绘制每个检测框
        for box in boxes:
            # 获取类别对应的颜色
            color = CLASS_COLORS.get(box.class_id, (255, 255, 255))

            # 绘制边界框（矩形）
            draw.rectangle(
                [box.x1, box.y1, box.x2, box.y2],
                outline=color,
                width=2,
            )

            # 绘制标签（类别名称 + 置信度）
            label = f"{box.cn_name} {box.confidence:.2f}"
            
            # 计算标签文字的尺寸
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]

            # 标签背景的位置（在边界框上方）
            label_y = max(0, box.y1 - label_h - 4)
            
            # 绘制标签背景（矩形填充）
            draw.rectangle(
                [box.x1, label_y, box.x1 + label_w + 4, label_y + label_h + 4],
                fill=color,
            )

            # 绘制标签文字
            draw.text(
                (box.x1 + 2, label_y + 2),
                label,
                fill=(255, 255, 255),  # 白色文字
                font=font,
            )

        # 返回numpy数组格式的图片
        return np.array(img)


# ========================================================================
# 工厂函数
# ========================================================================

def create_detector(model_path: str = None) -> YOLODetector:
    """
    创建检测器的工厂函数
    
    参数:
        model_path: 模型文件路径
                   如果为None，使用默认路径 "steel_yolo/models/best.pt"
    
    返回:
        YOLODetector: 检测器实例
    
    使用示例:
        # 使用默认模型
        detector = create_detector()
        
        # 使用指定模型
        detector = create_detector("path/to/custom_model.pt")
    """
    if model_path is None:
        # 使用默认路径：项目根目录下的 steel_yolo/models/best.pt
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "steel_yolo", "models", "best.pt")
    
    return YOLODetector(model_path)
