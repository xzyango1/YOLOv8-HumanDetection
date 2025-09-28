# YOLOv8 安全帽与人体检测项目 (保姆级教学版)

<div align="center">
  <img src="assets/demo.gif" width="80%">
</div>

*⚠️【请在此处替换为您自己录制的、展示`realtime_app.py`运行效果的GIF动图！确保已将`demo.gif`文件放入`assets`目录。】*

---

## 📖 项目简介

欢迎来到这个YOLOv8实战项目！本项目旨在通过一次完整的、端到端的实践，带领学习者从零开始，最终训练出一个能够实时、高精度地检测安全帽与人体的强大AI模型。

这个仓库不仅是代码的集合，更是我个人从遇到问题、分析问题到最终解决问题的完整学习路径的记录。你将在这里学到：
*   如何从零开始搭建一个专业的深度学习环境。
*   如何融合多个不同来源的数据集，并进行清洗和预处理。
*   如何解决在模型训练中遇到的各种真实世界问题（如硬件限制、过拟合等）。
*   如何将训练好的模型，封装成一个有趣的实时摄像头应用。

希望这个项目能成为你AI学习路上的一个有趣且坚实的脚印！

---

## 🚀 项目学习路线图

本项目被精心设计为一条循序渐进的学习路径。请严格按照以下步骤进行，你将能100%复现本项目的全部成果：

| 步骤 | 阶段             | 核心任务                                                     | 涉及文件/工具                                    |
| :--- | :--------------- | :----------------------------------------------------------- | :----------------------------------------------- |
| **0**  | **环境搭建**     | 配置一个稳定、高效的深度学习环境。                           | `Conda`, `pip`, `requirements.txt`, `PyTorch`        |
| **1**  | **数据准备**     | 融合三个数据集，创建一个庞大的、高质量的训练集。             | `data_preparation/` 目录下的脚本, `Roboflow`     |
| **2**  | **模型训练**     | 使用准备好的数据集，从零开始训练一个强大的`yolov8x`模型。      | `train.py`                                       |
| **3**  | **模型测试**     | 评估你训练出的模型，并用它来分析图片和视频。                 | `predict.py`                                     |
| **4**  | **实时应用**     | 将模型封装成一个可以调用电脑摄像头的实时检测程序。           | `realtime_app.py`                                |
| **5**  | **(选修)模型压缩**| 学习如何将巨大的`.pt`模型压缩为轻量级的`.onnx`模型。        | `quantize_model.py`                              |

---

## 🛠️ 第零步：环境搭建

在开始之前，我们需要为项目配置一个独立的Python环境，以避免与你电脑上其他的项目产生冲突。

### 1. 安装Conda
如果你还没有安装Conda，请先从官网下载并安装 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

### 2. 创建并激活Conda环境
打开你的终端（对于Windows用户，推荐使用Anaconda Prompt），然后逐行运行以下命令：

```bash
# 创建一个名为 yolo_env 的、使用Python 3.9 的新环境
conda create -n yolo_env python=3.9

# 激活这个新创建的环境
conda activate yolo_env
```
*成功激活后，你的终端提示符前面会出现`(yolo_env)`的字样。*

### 3. 克隆本项目仓库
```bash
git clone https://github.com/xzyango1/YOLOv8-HumanDetection.git
cd YOLOv8-HumanDetection
```

### 4. 安装项目依赖
本项目的所有依赖库都记录在`requirements.txt`中。

```bash
# 安装所有基础依赖库
pip install -r requirements.txt
```

### 5. 安装PyTorch (最关键的一步！)
深度学习的核心计算库PyTorch需要根据你的硬件（有无NVIDIA显卡）进行单独安装。

*   **如果你有NVIDIA显卡 (推荐)**：
    请访问 [PyTorch官网](https://pytorch.org/get-started/locally/)，根据你电脑的CUDA版本，选择并复制对应的安装命令。以下示例适用于CUDA 12.1：
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

*   **如果你的电脑只有CPU**：
    ```bash
    pip3 install torch torchvision torchaudio
    ```

至此，你的开发环境已经完美配置完毕！

---

## 📦 第一步：数据准备

一个强大的模型，离不开高质量、大规模的数据。在这个阶段，我们将学习如何将三个不同的数据集融合为一个终极数据集。

### 1. 下载原始数据集
本项目的数据集托管在[Roboflow Universe](https://universe.roboflow.com/)上。请分别从以下链接下载这三个数据集（选择`YOLOv8`格式导出）：

*   **数据集1 (头部安全)**: [Safety Helmet Dataset by andrewyolo](https://universe.roboflow.com/andrewyolo/safety-helmet-wqidg)
*   **数据集2 (头部安全)**: [Safety Helmet by mohammad-mehdi-tamehri](https://universe.roboflow.com/mohammad-mehdi-tamehri/safety-helmet-itjyo)
*   **数据集3 (人体)**: [Human by human-urngn](https://universe.roboflow.com/human-urngn/human-wg4jz)

### 2. 组织文件结构
在你的项目根目录下，新建一个`datasets`文件夹。将下载好的三个数据集解压到其中，并为它们起清晰的英文名（例如`DS1_Helmet`, `DS2_Helmet`, `DS3_Human`）。

### 3. 运行自动化准备脚本
为了将不同来源的数据进行统一和合并，你需要**依次**运行`data_preparation/`目录下的两个脚本：

```bash
# 第一步：统一所有数据集的类别标签
# (请先根据你自己的文件夹名，修改脚本内的SOURCE_DATASETS列表)
python data_preparation/remap_labels.py

# 第二步：将所有数据集的文件合并，并进行数量验证
python data_preparation/merge_and_verify.py
```
*运行成功后，你会在`datasets/`目录下得到一个名为`ULTIMATE_DATASET`的最终数据集文件夹。*

---

## 🧠 第二步：模型训练

现在，我们将使用准备好的终极数据集，来训练一个强大的`yolov8x`模型。

1.  **打开 `train.py` 文件。**
2.  在文件顶部的“配置区”，你可以根据自己的需求调整训练参数（如`EPOCHS`训练轮次等）。对于初次尝试，建议保持默认设置。
3.  **运行脚本开始训练：**
    ```bash
    python train.py
    ```
*这是一个漫长的过程，根据硬件情况可能需要1-2天。你可以随时按`Ctrl+C`提前中断，程序会自动保存当前最好的结果。*

---

## 📊 第三步：模型测试

训练完成后，是时候检验我们成果了！

1.  **找到你最好的模型**：训练结束后，你最好的模型权重会被保存在类似 `runs/detect/yolov8x_.../weights/best.pt` 的路径下。
2.  **打开 `predict.py` 文件。**
3.  在顶部的“配置区”，将`MODEL_PATH`的值，**精确地修改为你自己 `best.pt` 文件的完整路径**。
4.  将`SOURCE_PATH`修改为你想要测试的一张图片或一段视频的路径（项目中`assets/`文件夹内已提供一个测试视频）。
5.  **运行脚本进行预测：**
    ```bash
    python predict.py
    ```
*预测结果（带标注的图片/视频）会自动保存在一个新的`runs/detect/predict...`文件夹中，快去看看效果吧！*

---

## 🎥 第四步：实时应用

最后，让我们把模型变成一个能与你实时互动的应用程序！

1.  **打开 `realtime_app.py` 文件。**
2.  和上一步一样，在顶部的“配置区”，**将`MODEL_PATH`的值，修改为你自己`best.pt`文件的路径**。
3.  确保你的电脑摄像头没有被其他程序占用。
4.  **运行实时检测程序：**
    ```bash
    python realtime_app.py
    ```
*一个窗口将会弹出，显示你摄像头的实时画面，并开始进行智能检测！按键盘上的`q`键可以退出程序。*

---

## 💡 (选修) 第五步：模型压缩

如果你想将模型部署到手机或边缘设备，或者想更方便地分享模型，可以学习如何进行模型压缩。

1.  **打开 `quantize_model.py` 文件。**
2.  在“配置区”设置好正确的模型和数据路径。
3.  **运行脚本：**
    ```bash
    python quantize_model.py
    ```
*这将生成一个体积更小、速度更快的`.onnx`格式的量化模型。*

---

## 🤝 致谢

*   感谢 **Ultralytics** 团队开发的YOLOv8框架。
*   感谢 **Roboflow Universe** 社区及所有无私分享数据集的贡献者。

---
*由 [xzyango1](https://github.com/xzyango1) 创建与维护*
*如果你觉得这个项目对你有帮助，请给一个 Star ⭐ 吧！*