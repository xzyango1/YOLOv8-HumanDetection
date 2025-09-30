# YOLOv8 安全帽与人体检测项目


**一个由本项目V2.0模型驱动的实时检测效果演示。**

<div align="center">
  <img src="assets/demo.gif" width="80%">
</div>

*原视频：https://www.bilibili.com/video/BV1sdKizVE2g*

## 📖 项目简介

欢迎来到这个YOLOv8实战项目！本项目旨在通过一次完整的、端到端的实践，带领学习者从零开始，最终训练出一个能够实时、高精度地检测安全帽与人体的强大AI模型。

这个仓库不仅是代码的集合，更是我个人从遇到问题、分析问题到最终解决问题的完整学习路径的记录。你将在这里学到：
*   如何从零开始搭建一个专业的深度学习环境。
*   如何融合多个不同来源的数据集，并进行清洗和预处理。
*   如何解决在模型训练中遇到的各种真实世界问题（如硬件限制、过拟合等）。
*   如何将训练好的模型，封装成一个有趣的实时摄像头应用。

希望这个项目能成为你AI学习路上的一个有趣且坚实的脚印！

---

## 🚀 “开箱即用”快速体验指南

**如果你想立即看到效果，可以暂时跳过复杂的训练步骤！** 我已经将两个训练好的成品模型打包在了项目里，你可以按照以下步骤，在5分钟内直接体验：

1.  **完成 [第零步：环境搭建](#️-第零步环境搭建)**，确保你的环境已经配置好。
2.  **打开 `predict.py` 文件。**
3.  在顶部的“配置区”，将`EXPERIMENT_FOLDER`这个变量注释掉或删除。
4.  **取消你想测试的模型的`MODEL_PATH`一行的注释**。例如，要测试轻量级模型：
    ```python
    # EXPERIMENT_FOLDER = '...' # 注释掉这一行

    # --- 模型路径二选一 (取消你想要测试的那一行的注释) ---
    # 选项1: 轻量级模型 (速度快, 仅擅长头部/安全帽)
    MODEL_PATH = 'best-light.pt'

    # 选项2: 终极版ONNX模型 (精度高, 全能型)
    # MODEL_PATH = 'best-complete.int8.onnx' 
    ```
5.  **运行脚本**，即可看到预训练模型对`assets/test_video.mp4`的检测效果！
    ```bash
    python predict.py
    ```

---

## 📦 已训练好的模型介绍

本仓库包含两个可以直接使用的成品模型，它们分别代表了项目的两个关键阶段。

### 1. `best-light.pt` (V1.0 轻量级头部安全检测器)
*   **训练背景**: 使用`YOLOv8n`（Nano）这个最小的模型，在**一个**约5000张图片的数据集上，训练了100轮。
*   **性能数据**:
    *   **`mAP50-95` (总分)**: `0.398`
    *   **`mAP50` (分项)**: `head`: **92.0%**, `helmet`: **94.2%**, `person`: **1.5%**
*   **特点**:
    *   ✅ **极快的速度**：非常适合需要高帧率的实时应用。
    *   ✅ **头部/安全帽专家**：在识别是否佩戴安全帽这个核心任务上，精度极高。
    *   ❌ **无法识别完整的人**：由于训练数据问题，它几乎不认识`person`这个类别。
*   **格式**: `.pt`，这是PyTorch的原生格式，训练完的模型都会以`.pt`文件的形式出现。

### 2. `best-complete.int8.onnx` (V2.0 完整版综合安全检测器)
*   **训练背景**: 使用`YOLOv8x`（Extra-Large）这个最强大的模型，在一个由**三个**数据集融合而成的、近**3万张**图片的终极数据集上，通过智能早停机制，训练了43轮（最佳效果出现在第18轮）。
*   **性能数据**:
    *   **`mAP50-95` (总分)**: **`0.482`** (比V1.0提升了**21%**！)
    *   **`mAP50` (分项)**: `head`: **91.2%**, `helmet`: **87.4%**, `person`: **71.3%**
*   **特点**:
    *   ✅ **顶级精度与泛化性**：在所有类别上都表现出色，尤其是在`person`检测上实现了从无到有的质变。
    *   ✅ **体积小，速度快**：虽然原始模型`best-complete.py`巨大且不便上传，但这个版本经过了**INT8量化压缩**转化为`.onnx`文件，体积减小了约70%，推理速度也得到了极大提升（在RTX 4070上可达57 FPS）。
    *   ✅ **跨平台**：这是一个`.onnx`格式的模型，意味着它不依赖PyTorch，可以在更多不同的环境中被轻松部署。
*   **如何使用ONNX模型？**
    *   你**无需做任何特殊操作**！`ultralytics`库非常智能，`predict.py`和`realtime_app.py`中的`YOLO()`函数可以直接加载并运行`.onnx`文件，就像加载`.pt`文件一样简单。

---

## 📖 项目完整学习路线图

如果你想从头到尾完整地体验整个项目，请严格按照以下步骤进行。

本项目被精心设计为一条循序渐进的学习路径。严格按照以下步骤进行，你将能100%复现本项目的全部成果：

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

### 5. 安装FFmpeg以实现音画同步

为了让最终生成的预测视频能够包含原始音轨，本项目使用了一个强大的外部工具`FFmpeg`。

**最简单的安装方式 (Windows 10/11):**
打开你的终端，运行以下命令。Windows包管理器`winget`会自动帮你完成所有安装和配置：
```bash
winget install Gyan.FFmpeg
```
安装完成后，请**重启你的终端或VS Code**，然后运行`ffmpeg -version`。如果你能看到版本信息，就证明安装成功了！

*如果`winget`命令无效，请参考[这篇教程](https://www.wikihow.com/Install-FFmpeg-on-Windows)进行手动安装和环境变量配置。*

### 6. 安装PyTorch (最关键的一步！)

深度学习的核心计算库PyTorch需要根据你的硬件进行单独安装。

*   **选项A：如果你有NVIDIA显卡 (极推荐)**：
    这是释放项目全部性能的关键。请访问 [PyTorch官网](https://pytorch.org/get-started/locally/)，网站会自动检测你的系统，并为你生成最适合的安装命令。通常你只需在网站上选择 `Stable` -> `Windows` -> `Pip` -> `Python` -> `CUDA ...`，然后复制粘贴网站生成的命令即可。
    
    *一个常见的适用于CUDA 12.1的命令示例：*
    ```bash
    pip3 install torch torchvision torchiudio --index-url https://download.pytorch.org/whl/cu121
    ```

*   **选项B：如果你的电脑只有CPU，或GPU安装失败 (仅保证运行)**：
    如果你不确定自己的显卡配置，或者GPU版本安装失败，不用担心！CPU版本虽然速度较慢，但**保证可以运行**本项目的所有代码。
    ```bash
    pip3 install torch torchvision torchiudio
    ```

至此，你的开发环境已经完美配置完毕！



## 📦 第一步：数据准备

一个强大的模型，离不开高质量、大规模的数据。在这个阶段，我们将学习如何将三个不同的数据集融合为一个终极数据集。

### 1. 下载原始数据集
本项目的数据集托管在[Roboflow Universe](https://universe.roboflow.com/)上。请分别从以下链接下载这三个数据集（选择`YOLOv8`格式导出）：

*   **数据集1 (头部安全)**: [Safety Helmet Dataset by andrewyolo](https://universe.roboflow.com/andrewyolo/safety-helmet-wqidg)
*   **数据集2 (头部安全)**: [Safety Helmet by mohammad-mehdi-tamehri](https://universe.roboflow.com/mohammad-mehdi-tamehri/safety-helmet-itjyo)
*   **数据集3 (人体)**: [Human by human-urngn](https://universe.roboflow.com/human-urngn/human-wg4jz)

合并这三个数据集进行训练将会得到一个同时识别头部与人体的物体检测模型。
你也可以在[Roboflow Universe](https://universe.roboflow.com/)上搜索自己想要的数据集，实现其他效果。

### 2. 组织文件结构
在你的项目根目录下，新建一个`datasets`文件夹。将下载好的三个数据集解压到其中，并为它们起清晰的英文名（例如`DS1_Helmet`, `DS2_Helmet`, `DS3_Human`）。

### 3. 运行自动化准备脚本
为了将不同来源的数据进行统一和合并，你需要**依次**运行`data_preparation/`目录下的两个脚本：

```bash
# 第一步：统一所有数据集的类别标签
# ⚠️ 在运行前，请务必用VS Code打开 data_preparation/remap_labels.py 文件，
# 找到顶部的“配置区”，将其中的文件夹名修改为你自己解压后得到的名字。
python data_preparation/remap_labels.py

# 第二步：将所有数据集的文件合并，并进行数量验证
python data_preparation/merge_and_verify.py
```
*运行成功后，你会在`datasets/`目录下得到一个名为`ULTIMATE_DATASET`的最终数据集文件夹。*

---

## 🧠 第二步：模型训练

现在，我们将使用准备好的数据集，来训练一个强大的`yolov8x`（默认）模型。

1.  **打开 `train.py` 文件。**
2.  在文件顶部的“配置区”，你可以根据自己的需求调整训练参数（如`EPOCHS`训练轮次等）。对于初次尝试，如果没有完全理解各个配置，建议保持默认设置。
3.  **运行脚本开始训练：**
    ```bash
    python train.py
    ```
这是一个漫长的“挂机”任务，你可以启动训练后，就去睡觉或者做别的事情。根据你的硬件，完整的100轮训练大约需要1-2天。通常模型会在这之前完成收敛，并在设置的第"PATIENCE"次训练无提升后结束训练，因此实际训练时长通常短于这一预测值。

**非常重要**：你不需要一次性跑完！可以随时在终端按`Ctrl+C`来中断训练，程序会自动保存当前为止最好的模型。即使只训练几个小时，你也能得到一个效果不错的模型！*

第一次训练也可以使用`yolov8n`小模型，`5000`张图片的数据集（如只加载第一个链接所述数据集，这样可以暂时跳过数据集合并步骤），以快速验证配置可行性并得到一个效果尚可的安全帽识别模型，训练时长将在约1小时以内，适合快速见到成果。

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
*一个窗口将会弹出，显示你摄像头的实时画面，并开始进行智能检测！按键盘上的`q`键可以退出程序。（英文键盘在窗口里输入）*

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

在运行此脚本前，请确保已安装所有必要的转换库。可以运行以下命令：
    ```bash
    pip install tensorflow-cpu onnx onnx-tf onnxruntime
    ```


# 📋 常见问题 FAQ

## 🔧 环境配置问题

### Q1: conda命令无法识别怎么办？
**问题表现**：
```
'conda' 不是内部或外部命令，也不是可运行的程序
```

**解决方案**：
1. **方案A - 初始化PowerShell**（推荐）
   ```bash
   # 打开Anaconda Prompt（管理员模式）
   conda init powershell
   # 重启终端
   ```

2. **方案B - 使用完整路径**
   ```bash
   # 找到你的Anaconda安装路径，通常在：
   C:\ProgramData\Anaconda3\Scripts\conda.exe activate yolo_env
   ```

3. **方案C - 添加环境变量**
   - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
   - 在Path中添加：`C:\ProgramData\Anaconda3\Scripts`（根据实际路径调整）

---

### Q2: PyTorch安装失败或CUDA版本不匹配
**问题表现**：
```
RuntimeError: CUDA out of memory
或
torch.cuda.is_available() 返回 False
```

**解决方案**：
1. **检查CUDA版本**
   ```bash
   nvidia-smi  # 查看显卡驱动支持的最高CUDA版本
   ```

2. **安装匹配版本的PyTorch**
   - 访问 [PyTorch官网](https://pytorch.org/get-started/locally/)
   - 选择对应的CUDA版本
   - 如果不确定，安装CPU版本保证可用：
   ```bash
   pip3 install torch torchvision torchaudio
   ```

3. **验证安装**
   ```python
   import torch
   print(torch.__version__)
   print(f"CUDA可用: {torch.cuda.is_available()}")
   print(f"CUDA版本: {torch.version.cuda}")
   ```

---

### Q3: requirements.txt 安装失败
**问题表现**：
```
ERROR: Could not find a version that satisfies the requirement...
```

**解决方案**：
1. **更新pip**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **分步安装**
   ```bash
   # 先安装核心依赖
   pip install ultralytics
   pip install opencv-python
   pip install numpy pandas matplotlib
   
   # 再安装其他依赖
   pip install -r requirements.txt
   ```

3. **使用国内镜像**（中国用户）
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

---

## 📊 数据集问题

### Q4: 数据集下载失败或无法访问Roboflow
**解决方案**：
1. **使用VPN**：Roboflow在某些地区可能访问受限
2. **备用数据集**：
   - 在[Kaggle](https://www.kaggle.com/)搜索"helmet detection"
   - 在[COCO Dataset](https://cocodataset.org/)下载人体数据
   - 使用开源数据集如[CrowdHuman](https://www.crowdhuman.org/)

3. **自己标注数据**：
   - 使用[LabelImg](https://github.com/heartexlabs/labelImg)
   - 使用[CVAT](https://www.cvat.ai/)在线标注工具

---

### Q5: remap_labels.py 运行出错
**问题表现**：
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**：
1. **检查文件夹路径**
   ```python
   # 在 remap_labels.py 顶部配置区，确保路径正确
   DATASET_FOLDERS = [
       'datasets/DS1_Helmet',  # 改成你实际的文件夹名
       'datasets/DS2_Helmet',
       'datasets/DS3_Human'
   ]
   ```

2. **验证数据集结构**
   ```
   datasets/
   ├── DS1_Helmet/
   │   ├── train/
   │   │   ├── images/
   │   │   └── labels/
   │   └── valid/
   └── ...
   ```

3. **手动检查标签文件**
   - 打开`.txt`标签文件，确保格式为：`class_id x_center y_center width height`

---

### Q6: 合并数据集后标签混乱
**解决方案**：
1. **运行验证脚本**
   ```bash
   python data_preparation/merge_and_verify.py
   ```
   查看输出的统计信息，确认类别数量正确

2. **手动检查映射**
   ```python
   # 确保类别映射正确
   CLASS_MAPPING = {
       'head': 0,
       'helmet': 1, 
       'person': 2
   }
   ```

---

## 🏋️ 模型训练问题

### Q7: CUDA out of memory 内存不足
**问题表现**：
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**解决方案**：
1. **减小batch size**（最有效）
   ```python
   # 在 train.py 配置区
   BATCH_SIZE = 8  # 从16降到8，甚至4或2
   ```

2. **使用更小的模型**
   ```python
   MODEL_SIZE = 'yolov8n'  # 从yolov8x改为yolov8n或yolov8s
   ```

3. **减小图像尺寸**
   ```python
   IMAGE_SIZE = 416  # 从640降到416
   ```

4. **启用混合精度训练**
   ```python
   # 在train()函数中添加
   amp=True  # 自动混合精度
   ```

5. **清理GPU缓存**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### Q8: 训练速度非常慢

**优化方案**：
1. **确认使用GPU**
   ```python
   import torch
   print(f"使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
   ```

2. **增大batch size**（如果显存允许）
   ```python
   BATCH_SIZE = 32  # GPU显存充足时可以增大
   ```

3. **使用更快的数据加载**
   ```python
   WORKERS = 8  # 增加数据加载线程数（根据CPU核心数调整）
   ```

4. **关闭不必要的验证**
   ```python
   # 降低验证频率
   val=True  # 如果训练慢，可以每几个epoch验证一次
   ```

---

### Q9: 训练loss不下降或震荡严重
**解决方案**：
1. **调整学习率**
   ```python
   LR0 = 0.001  # 降低初始学习率（默认0.01）
   ```

2. **检查数据质量**
   - 查看`runs/detect/train/`中的训练图像
   - 确认标注框是否准确

3. **增加训练轮次**
   ```python
   EPOCHS = 200  # 从100增加到200
   PATIENCE = 100  # 提高早停耐心值
   ```

4. **使用预训练权重**
   ```python
   # 在train.py中确保使用预训练模型
   model = YOLO('yolov8x.pt')  # 而不是从头训练
   ```

---

### Q10: 训练中断后如何继续
**解决方案**：
```python
# 在 train.py 中修改
model = YOLO('runs/detect/yolov8x_<timestamp>/weights/last.pt')  # 加载上次的检查点
results = model.train(
    resume=True,  # 设置为True继续训练
    # ... 其他参数保持不变
)
```

---

## 🎯 模型预测问题

### Q11: 预测结果中文乱码
**问题表现**：图像标注中中文显示为方框

**解决方案**：
```python
# 在 predict.py 或 realtime_app.py 开头添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

---

### Q12: 实时检测帧率太低
**问题表现**：摄像头画面卡顿，FPS < 10

**解决方案**：
1. **使用量化模型**
   ```python
   MODEL_PATH = 'best-complete.int8.onnx'  # ONNX模型更快
   ```

2. **降低分辨率**
   ```python
   # 在 realtime_app.py 中
   results = model(frame, imgsz=416)  # 从640降到416
   ```

3. **跳帧处理**
   ```python
   frame_count = 0
   while True:
       frame_count += 1
       if frame_count % 2 == 0:  # 每2帧处理一次
           results = model(frame)
   ```

4. **使用更快的模型**
   ```python
   MODEL_PATH = 'best-light.pt'  # 使用YOLOv8n轻量模型
   ```

---

### Q13: 模型检测效果差
**可能原因和解决方案**：

1. **数据集质量问题**
   - 检查训练图像是否与测试场景相似
   - 确认标注是否准确

2. **置信度阈值过高**
   ```python
   # 在 predict.py 中调整
   results = model(source, conf=0.25)  # 降低置信度阈值（默认0.5）
   ```

3. **NMS阈值调整**
   ```python
   results = model(source, iou=0.5)  # 调整IoU阈值
   ```

4. **需要更多训练**
   - 增加训练轮次
   - 使用数据增强

---

## 🚀 模型部署问题

### Q14: ONNX模型转换失败
**解决方案**：
```bash
# 确保安装所有依赖
pip install onnx onnxruntime onnx-tf tensorflow-cpu

# 如果还是失败，尝试单独导出
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.export(format='onnx')"
```

---

### Q15: 摄像头无法打开
**问题表现**：
```
Error: Could not open camera
```

**解决方案**：
1. **检查摄像头索引**
   ```python
   # 在 realtime_app.py 中尝试不同索引
   cap = cv2.VideoCapture(0)  # 尝试0, 1, 2...
   ```

2. **检查权限**
   - Windows：设置 → 隐私 → 相机 → 允许应用访问相机

3. **关闭其他占用摄像头的程序**
   - 关闭视频会议软件、其他Python脚本等

---

## 💾 文件和路径问题

### Q16: 找不到模型文件
**解决方案**：
```python
# 使用绝对路径
import os
MODEL_PATH = os.path.abspath('runs/detect/yolov8x_20240315/weights/best.pt')

# 或者使用相对路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'runs/detect/.../weights/best.pt')
```

---

### Q17: 路径中包含中文导致错误
**解决方案**：
- 避免在项目路径中使用中文
- 将项目移动到纯英文路径，如：`D:/Projects/YOLOv8-HumanDetection`

---

## 📚 学习建议

### Q18: 我是完全的新手，该如何开始？
**推荐学习路径**：

1. **第1天**：环境搭建 + 快速体验预训练模型
2. **第2-3天**：理解YOLOv8原理，运行predict.py测试
3. **第4-5天**：下载和整理数据集
4. **第6-7天**：使用小数据集（1000张）训练YOLOv8n模型
5. **第8-10天**：完整训练YOLOv8x模型
6. **第11天**：部署实时应用
7. **第12天+**：尝试自定义数据集

---

### Q19: 如何使用自己的数据集？
**步骤**：
1. 准备图像文件
2. 使用LabelImg或CVAT标注（YOLO格式）
3. 组织成以下结构：
   ```
   my_dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```
4. 创建`data.yaml`：
   ```yaml
   path: ./my_dataset
   train: images/train
   val: images/val
   names:
     0: class1
     1: class2
   ```
5. 修改`train.py`中的`DATA_PATH`

---

## 🆘 仍然无法解决？

如果以上方案都无法解决你的问题：

1. **提交Issue**：在[GitHub Issues](https://github.com/xzyango1/YOLOv8-HumanDetection/issues)详细描述问题
2. **查看官方文档**：[Ultralytics YOLOv8文档](https://docs.ultralytics.com/)
3. **加入社区**：Ultralytics Discord或相关论坛寻求帮助

**提问时请包含**：
- 操作系统和Python版本
- 完整的错误信息
- 你已经尝试过的解决方案
- 相关配置文件和代码片段

---

## 🤝 致谢

*   感谢 **Ultralytics** 团队开发的YOLOv8框架。
*   感谢 **Roboflow Universe** 社区及所有无私分享数据集的贡献者。

---
*由 [xzyango1](https://github.com/xzyango1) 创建与维护*

*如果你觉得这个项目对你有帮助，请给一个 Star ⭐ 吧！秋梨膏！*