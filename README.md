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

## 项目流程

### 1. 分析钢铁材料的缺陷种类

分为 6 种缺陷类型：
- crazing（龟裂）
- inclusion（夹杂）
- pitted_surface（点蚀）
- scratches（划痕）
- patches（斑块）
- rolled-in_scale（氧化铁皮压入）

### 2. 收集钢铁材料缺陷图片

- 按 80/20 原则划分：80% 用于微调训练，20% 用于测试
- 即使数据量少也要预留测试集
- 使用 labelImg 进行数据标注，为每个缺陷画出边界框

### 3. EDA 数据分析

- 分析每种类别数量，每类不低于 300 张
- 根据标注的 txt/xml 文件，分析标注框的宽高与图片的占比
- 占比约等于 0 或约等于 1 说明标注可能有问题
- 检查类别是否均衡，若某个类别远少于其他类别，需通过过采样或增强补齐

### 4. 数据集增强

- 方向转换、亮度增强、锐化、随机翻转
- 多图层拼接、HSV 空间操作
- 增强后的图片和对应的标注标签需同步处理

### 5. YOLO 数据集准备

- 准备 `dataset.yaml`：配置数据集路径、类别数量和名称
- 准备 `yolov12.yaml`：配置 YOLO 模型神经网络参数

### 6. 验证测试数据集

- 验证训练集、测试集文件是否存在
- 验证类别配置是否正确

### 7. YOLO 模型微调训练

- 使用 YOLOv12 模型进行微调
- 监控三个损失函数：
  - box_loss：边界框损失（越小越好，说明定位越准确）
  - cls_loss：分类损失（越小越好，说明分类越准确）
  - dfl_loss：分布焦点损失（越小越好，说明边界框分布越集中）
- 使用早停策略（Early Stopping），验证集指标不再提升时自动停止，防止过拟合
- 保留微调后的模型：`best.pt`

### 8. 模型独立评估

- 用独立的测试集计算 precision、recall、mAP 等指标
- 不能只看训练损失，需评估模型在未见数据上的泛化能力

### 9. 制作检测引擎

- 将 YOLO 模型封装为易用的检测工具
- 处理异常情况：图片为空、格式错误、模型加载失败等兜底逻辑
- 后续调用工具即可检测图片

### 10. 编写 Web 页面

- 编写前端页面，便于用户上传图片
- 通过 YOLO 模型识别钢铁中是否存在缺陷
- 置信度阈值可调，让用户根据场景调整
- 支持批量检测，满足工业场景需求

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
2. 调整置信度阈值（可选，默认 0.1）
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
