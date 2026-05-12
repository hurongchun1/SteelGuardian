# SteelGuardian - 钢铁缺陷检测系统

基于 YOLOv12 的钢铁表面缺陷自动检测系统，提供 Web 界面进行实时检测。

## 功能特点

- 支持 6 种钢铁表面缺陷检测：
  - 龟裂 (crazing)
  - 夹杂 (inclusion)
  - 点蚀 (pitted_surface)
  - 划痕 (scratches)
  - 斑块 (patches)
  - 氧化铁皮压入 (rolled-in_scale)
- 基于 YOLOv12 的高精度检测
- Gradio Web 界面，支持实时检测
- 可调节置信度阈值
- 生成检测结果标注图

## 项目结构

```
SteelGuardian/
├── app.py                  # Gradio Web 界面
├── detection_engine.py     # 检测引擎（核心推理模块）
├── steel_yolo/             # YOLO 训练相关文件
│   ├── data/               # 训练数据集
│   ├── models/             # 训练好的模型
│   ├── runs/               # 训练运行记录
│   ├── train.py            # 训练脚本
│   └── ...
└── README.md               # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics gradio pillow numpy
```

### 2. 启动 Web 界面

```bash
python app.py
```

启动后会自动打开浏览器，访问 http://localhost:7860

### 3. 使用检测功能

1. 上传钢铁表面图片（或选择示例图片）
2. 调整置信度阈值（可选，默认 0.25）
3. 点击"开始检测"
4. 查看检测结果

## 使用检测引擎（编程接口）

```python
from detection_engine import create_detector

# 创建检测器（自动加载模型）
detector = create_detector()

# 检测图片
result = detector.detect("test.jpg", conf=0.25)

# 查看结果
print(result.summary())

# 获取标注后的图片
annotated_img = result.annotated_image
```

## 模型文件

模型文件位于 `steel_yolo/models/best.pt`，如果该文件不存在，请先训练模型。

### 训练模型

```bash
cd steel_yolo
python train.py
```

## 技术栈

- **检测模型**: YOLOv12
- **深度学习框架**: PyTorch + ultralytics
- **Web 界面**: Gradio
- **图像处理**: Pillow + OpenCV

## 注意事项

1. 首次启动时会自动加载模型，可能需要几秒钟
2. 确保 `steel_yolo/models/best.pt` 模型文件存在
3. Web 界面默认监听 0.0.0.0:7860 端口