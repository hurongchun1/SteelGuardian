# -*- coding: utf-8 -*-
"""
钢铁缺陷检测系统 - Web界面
=========================
基于 Gradio 的可视化检测界面

功能:
    1. 上传钢铁表面图片
    2. 自动检测缺陷（龟裂、夹杂、点蚀、划痕、斑块、氧化铁皮压入）
    3. 显示检测结果和标注后的图片

使用方式:
    python app.py
    或
    gradio app.py

访问地址:
    http://localhost:7860
"""

import os
import gradio as gr
import numpy as np
from PIL import Image

from detection_engine import (
    create_detector,
    CLASS_NAMES,
    CLASS_NAMES_CN,
    CLASS_COLORS,
)


# ========================================================================
# 全局配置
# ========================================================================

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径（优先级从高到低）
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "steel_yolo", "models", "best.pt"),  # 推荐位置
    os.path.join(BASE_DIR, "models", "best.pt"),                # 备选位置
]

# 查找存在的模型文件
DEFAULT_MODEL = next(
    (p for p in MODEL_CANDIDATES if os.path.exists(p)),
    MODEL_CANDIDATES[0]  # 如果都不存在，使用第一个路径
)

# 全局检测器实例（延迟初始化）
detector = None


def get_detector():
    """获取或创建检测器实例（单例模式）"""
    global detector
    if detector is None:
        if os.path.exists(DEFAULT_MODEL):
            detector = create_detector(DEFAULT_MODEL)
        else:
            raise FileNotFoundError(
                f"模型文件不存在: {DEFAULT_MODEL}\n"
                "请先训练模型或下载预训练权重"
            )
    return detector


# ========================================================================
# 检测函数
# ========================================================================

def detect_defects(image, conf_threshold):
    """
    检测图片中的缺陷
    
    参数:
        image: 上传的图片（PIL Image 或 numpy array）
        conf_threshold: 置信度阈值（0-1）
    
    返回:
        annotated_image: 标注后的图片
        result_text: 检测结果文字
        json_result: JSON格式的详细结果
    """
    # 检查图片是否为空
    if image is None:
        return None, "请先上传图片", "{}"
    
    try:
        # 获取检测器
        detector = get_detector()
        
        # 如果是numpy数组，转换为PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 保存临时图片
        temp_path = os.path.join(BASE_DIR, "temp_image.jpg")
        image.save(temp_path, quality=95)
        
        # 执行检测
        result = detector.detect(
            temp_path,
            conf=conf_threshold,
            draw_boxes=True,
        )
        
        # 生成结果文字
        result_text = format_result_text(result)
        
        # 生成JSON结果
        json_result = str(result.to_dict())
        
        # 获取标注后的图片
        annotated_image = result.annotated_image
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return annotated_image, result_text, json_result
        
    except Exception as e:
        error_msg = f"检测失败: {str(e)}"
        return None, error_msg, "{}"


def format_result_text(result):
    """格式化检测结果为文字"""
    if not result.has_defect:
        return "✅ 未检测到缺陷\n\n钢材表面质量良好。"
    
    lines = [
        f"⚠️ 检测到 {result.defect_count} 个缺陷区域",
        f"📊 推理耗时: {result.inference_time:.3f} 秒",
        "",
        "【缺陷详情】",
    ]
    
    # 按置信度排序
    sorted_boxes = sorted(result.boxes, key=lambda b: b.confidence, reverse=True)
    
    for i, box in enumerate(sorted_boxes, 1):
        lines.append(
            f"{i}. {box.cn_name} ({box.class_name})\n"
            f"   置信度: {box.confidence:.2%}\n"
            f"   位置: ({box.x1:.0f}, {box.y1:.0f}) - ({box.x2:.0f}, {box.y2:.0f})"
        )
    
    # 统计各类别数量
    type_counts = {}
    for box in result.boxes:
        cn_name = box.cn_name
        type_counts[cn_name] = type_counts.get(cn_name, 0) + 1
    
    lines.append("")
    lines.append("【类别统计】")
    for name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  • {name}: {count} 个")
    
    return "\n".join(lines)


# ========================================================================
# 示例图片
# ========================================================================

def get_sample_images():
    """获取示例图片列表"""
    samples = []
    
    # 尝试从测试集获取示例
    test_dir = os.path.join(BASE_DIR, "steel_yolo", "data", "test", "images")
    if os.path.exists(test_dir):
        files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])[:5]
        for f in files:
            samples.append(os.path.join(test_dir, f))
    
    return samples if samples else None


# ========================================================================
# Gradio 界面
# ========================================================================

# 自定义CSS样式
CUSTOM_CSS = """
.result-text {
    font-family: monospace;
    white-space: pre-wrap;
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 5px;
}
"""

def create_ui():
    """创建Gradio界面"""
    
    # 创建界面
    with gr.Blocks(
        title="钢铁缺陷检测系统",
    ) as demo:
        
        # 标题
        gr.Markdown(
            """
            # 🔍 钢铁缺陷检测系统
            
            基于 YOLOv12 的钢铁表面缺陷自动检测工具
            
            **支持的缺陷类型:** 龟裂 | 夹杂 | 点蚀 | 划痕 | 斑块 | 氧化铁皮压入
            """
        )
        
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 📤 上传图片")
                
                # 图片上传
                input_image = gr.Image(
                    label="上传钢铁表面图片",
                    type="pil",
                    height=400,
                )
                
                # 示例图片
                sample_images = get_sample_images()
                if sample_images:
                    gr.Examples(
                        examples=sample_images,
                        inputs=input_image,
                        label="示例图片",
                    )
                
                # 置信度阈值
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.9,
                    value=0.1,
                    step=0.05,
                    label="置信度阈值",
                    info="低于此值的检测框将被过滤（建议从0.1开始）",
                )
                
                # 检测按钮
                detect_btn = gr.Button(
                    "🚀 开始检测",
                    variant="primary",
                    size="lg",
                )
            
            # 右侧：输出区域
            with gr.Column(scale=1):
                gr.Markdown("### 📊 检测结果")
                
                # 输出图片
                output_image = gr.Image(
                    label="检测结果",
                    height=400,
                )
                
                # 结果文字
                result_text = gr.Textbox(
                    label="检测详情",
                    lines=12,
                    interactive=False,
                    elem_classes=["result-text"],
                )
        
        # 详细JSON结果（可折叠）
        with gr.Accordion("📋 详细JSON结果", open=False):
            json_output = gr.Textbox(
                label="JSON",
                lines=10,
                interactive=False,
            )
        
        # 底部信息
        gr.Markdown(
            """
            ---
            **使用说明:**
            1. 上传钢铁表面图片或选择示例图片
            2. 调整置信度阈值（建议从0.1开始尝试）
            3. 点击"开始检测"
            4. 查看检测结果
            
            **模型信息:** YOLOv12 | 钢铁缺陷检测 | 设备: CPU
            **提示:** 如果未检测到缺陷，请尝试降低置信度阈值（如0.1-0.2）
            """
        )
        
        # 绑定事件
        detect_btn.click(
            fn=detect_defects,
            inputs=[input_image, conf_slider],
            outputs=[output_image, result_text, json_output],
        )
    
    return demo


# ========================================================================
# 主程序入口
# ========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("钢铁缺陷检测系统")
    print("=" * 50)
    print(f"模型路径: {DEFAULT_MODEL}")
    
    # 检查模型是否存在
    if not os.path.exists(DEFAULT_MODEL):
        print(f"[警告] 模型文件不存在: {DEFAULT_MODEL}")
        print("请先训练模型或下载预训练权重")
    else:
        print("[OK] 模型文件已就绪")
    
    print()
    print("启动Web界面...")
    print("访问地址: http://localhost:7860")
    print("=" * 50)
    
    # 创建并启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",    # 允许外部访问
        server_port=7860,          # 端口号
        share=False,               # 不创建公共链接
        inbrowser=True,            # 自动打开浏览器
        css=CUSTOM_CSS,            # 自定义CSS样式
        theme=gr.themes.Soft(),    # 主题设置
    )
