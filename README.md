# YOLOv8 安全帽与人体综合检测项目 (端到端教学版)

![Project Banner](...) <!-- 建议在这里放一张您最终模型检测效果的酷炫截图或GIF -->

欢迎来到这个YOLOv8实战项目！本项目旨在通过一次完整的、端到端的实践，带领学习者从零开始，最终训练出一个能够实时、高精度地检测安全帽与人体的强大AI模型。

这个仓库不仅是代码的集合，更是我个人从遇到问题、分析问题到最终解决问题的完整学习路径的记录。希望能对你的AI学习之旅有所启发和帮助！

---

## 🌟 项目亮点

*   **端到端全流程**：涵盖了从建立性能基线、数据工程、硬件调试，到最终模型训练与实时应用开发的全过程。
*   **解决真实世界问题**：聚焦于解决`person`类别检测失败这一典型的数据不平衡问题。
*   **专业的工程实践**：项目采用了**数据融合**、**标签重映射**、**早停机制（Early Stopping）**等一系列行业标准技术。
*   **详尽的文档**：提供了完整的环境配置指南和代码说明，旨在让学习者能够轻松复现。

---

## 🚀 如何开始？

### 1. 克隆仓库
```bash
git clone https://github.com/xzyango1/YOLOv8-HumanDetection.git
cd YOLOv8-HumanDetection
```

### 2. 环境配置
推荐使用Conda进行环境管理。
```bash
# 创建并激活Conda环境
conda create -n yolo_env python=3.9
conda activate yolo_env

# 安装所有基础依赖
pip install -r requirements.txt

# 安装与您显卡匹配的GPU版PyTorch (关键！)
# 以下示例适用于CUDA 12.1，请根据您的NVIDIA驱动版本进行调整
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 下载数据集 (重要！)
本项目的数据集过于庞大，并未包含在仓库中。请从以下链接下载，并参考`merge_and_verify.py`脚本中的路径，将它们解压到`datasets/`目录下。

*   **数据集1 (头部安全)**: [Safety Helmet Dataset by andrewyolo](https://universe.roboflow.com/andrewyolo/safety-helmet-wqidg)
*   **数据集2 (头部安全)**: [Safety Helmet by mohammad-mehdi-tamehri](https://universe.roboflow.com/mohammad-mehdi-tamehri/safety-helmet-itjyo)
*   **数据集3 (人体)**: [Human by human-urngn](https://universe.roboflow.com/human-urngn/human-wg4jz)

### 4. 运行项目！
*   **数据准备**: 仔细阅读并运行 `remap_labels.py` 和 `merge_and_verify.py`，以创建最终的数据集。
*   **模型训练**: 修改并运行 `train.py` 来开始你自己的训练。
*   **模型测试**: 使用 `predict.py` 来测试已训练好的模型。
*   **实时应用**: 运行 `realtime_app.py`，调用你的摄像头，见证实时检测的威力！

---

## 📜 项目文件结构说明

```
.
├── datasets/               # (需手动创建) 存放所有数据集
├── runs/                   # (自动生成) 存放所有训练和预测结果
├── .gitignore              # Git忽略配置，防止上传大数据
├── merge_and_verify.py     # 自动化合并与验证数据集的脚本
├── predict.py              # 使用训练好的模型进行预测的脚本
├── README.md               # 项目说明文档 (就是你正在看的这个!)
├── realtime_app.py         # 实时摄像头检测的应用脚本
├── remap_labels.py         # 自动化重映射数据集标签的脚本
├── requirements.txt        # 项目Python依赖库
└── train.py                # 训练模型的脚本
```

## 致谢
*   感谢 **Ultralytics** 团队开发的YOLOv8框架。
*   感谢 **Roboflow Universe** 社区及所有无私分享数据集的贡献者。

---
*由 xzyango1 创建与维护*