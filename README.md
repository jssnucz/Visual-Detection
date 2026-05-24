# 🔍 智能视觉检测系统

基于 YOLOv8、EasyOCR 和 GPT-4V 构建的智能视觉检测平台，支持自然语言交互式目标检测与 OCR 文字识别，检测结果可一键导出为专业报告。

## ✨ 功能特性

### 🎯 智能目标检测
- **自然语言交互**：用日常语言描述检测需求（如"找出图片中的所有汽车"），系统自动理解并执行
- **双引擎检测**：自动选择 YOLOv8 快速检测（80 类常见物体）或视觉大模型深度识别
- **可视化标注**：检测结果自动在图像上绘制边界框和标签

### 📝 OCR 文字识别
- 支持中英文混合文字识别
- 自动标注文字位置与置信度
- 支持 EasyOCR 本地识别和视觉大模型云端识别双模式

### 📁 报告导出
- **DOCX 格式**：生成 Word 文档，包含检测图像和详细结果表格
- **PDF 格式**：生成 PDF 文档，支持中文字体，适合打印和分享

### 🌐 多模型支持
- 支持 OpenAI API 及兼容接口（API2D 等）
- 可自由切换 GPT-4V、GPT-4o 等视觉大模型
- Web 界面一键配置 API 密钥和模型

## 🖼️ 系统界面

系统启动后提供四个功能标签页：

| 标签页 | 功能 |
|--------|------|
| 🎯 智能目标检测 | 上传图片 + 自然语言描述，自动检测并标注目标 |
| 📝 OCR 文字识别 | 上传图片，提取其中所有文字内容 |
| ⚙️ API 设置 | 配置大语言模型 API 密钥、地址和模型 |
| 📖 使用指南 | 详细的功能说明和使用方法 |

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 推荐 GPU 环境（支持 CUDA 加速）

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/yourusername/vision-detection-system.git
cd vision-detection-system

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt
```

### 启动系统

```bash
python main.py
```

启动后浏览器自动打开 `http://localhost:7860`

### API 配置（可选）

如需使用视觉大模型检测功能：
1. 在 Web 界面点击"⚙️ API 设置"标签页
2. 输入 OpenAI 兼容 API 密钥
3. 配置 API 地址和模型名称
4. 点击"保存配置"即可启用

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 目标检测 | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| OCR 识别 | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| 视觉大模型 | OpenAI GPT-4V / GPT-4o |
| Web 界面 | [Gradio](https://gradio.app/) |
| 文档生成 | python-docx / ReportLab |
| 图像处理 | OpenCV / Pillow |

## 📂 项目结构

```
project/
├── main.py              # 主程序（包含检测系统核心逻辑和 Gradio 界面）
├── yolov8n.pt           # YOLOv8 预训练模型（首次运行自动下载）
├── requirements.txt     # Python 依赖
└── README.md            # 本文件
```

## 📋 使用示例

### 目标检测
1. 上传图片
2. 输入检测需求："帮我找出图片中的所有人"
3. 系统自动选择 YOLO 检测，标注所有人物位置
4. 点击"导出报告"生成 DOCX 或 PDF

### OCR 识别
1. 上传包含文字的图片
2. 点击"提取文字"
3. 查看标注图片和提取的文字内容

## 📄 License

MIT License
