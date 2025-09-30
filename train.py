# train.py (简约版)
from ultralytics import YOLO

# --- 配置区 ---
# 在这里修改所有训练参数，而无需使用命令行

# 1. 数据集配置文件路径 (YAML)
#    这就像是训练的“地图”，告诉程序去哪里找数据。通常你不需要修改它。
DATA_CONFIG_PATH = 'datasets/ULTIMATE_DATASET/data.yaml'

# 2. 训练参数 (这里是你“炼丹”的地方)
EPOCHS = 100         # 你想让模型学习多少轮？100轮是一个很好的起点。
BATCH_SIZE = 4       # 一次“喂”给显卡几张图片。如果显存小，就调低这个值，显存大可适当调大。
PATIENCE = 25        # 如果连续25轮都没进步，就智能地提前停止，为你节省时间，通常设为总轮数的20%-30%。
WORKERS = 8          # “工人”数量，帮你准备数据。通常设为CPU核心数的一半即可。
DEVICE = 1          # 使用哪块显卡。0代表第一块。如果想用CPU，就写 'cpu'。用GPU训练会快很多，一般电脑GPU区分为核显（性能弱）和独显（性能强），注意选择性能强的显卡，这里默认第二块为独立显卡。
PROJECT_NAME = 'My_YOLOv8_Journey' # 所有训练的总项目名
EXPERIMENT_NAME = 'exp_01_yolov8x_100e' # 本次训练的具体名称
LR0 = 0.01 # 初始学习率，默认0.01，通常不需要修改

# --- 核心执行区 ---

def main():
    """主函数：执行模型训练"""
    print("--- 开始模型训练 ---")
    
    # 加载预训练模型
    model = YOLO('yolov8x.pt') # 你也可以换成 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt' 从'n'到'x'逐渐变大变强, 但也更占显存和更慢
    print(f"模型 '{model.model_name}' 已加载。")

    # 开始训练
    model.train(
        data=DATA_CONFIG_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        cache=False # 建议保持False以确保稳定
    )
    
    print("\n--- 训练完成！---")

if __name__ == '__main__':
    main()