# train.py (简约版)
from ultralytics import YOLO

# --- 配置区 ---
# 在这里修改所有训练参数，而无需使用命令行

# 1. 数据集配置文件路径 (YAML)
#    这个文件里定义了数据集的位置、类别等信息。
DATA_CONFIG_PATH = 'datasets/ULTIMATE_DATASET/data.yaml' # 假设您的数据集yaml在此

# 2. 训练参数
EPOCHS = 100 # 训练轮数
BATCH_SIZE = 4 # 每个批次的图片数量 (根据显存调整，显存大可适当调大)
PATIENCE = 25 # 早停法：如果在这么多轮内验证集指标没有提升，则提前结束训练
WORKERS = 8 # 数据加载的线程数 (根据CPU核数调整)
DEVICE = 1  # 0代表使用第一个GPU,以此类推； 如果想用CPU, 则写 'cpu'。 通常电脑有核显（性能弱）和独立显卡（性能强），请确保选择性能强的显卡。
PROJECT_NAME = 'YOLOv8-Safety-Helmet-and-Person' # 训练结果将保存在 runs/detect/PROJECT_NAME 目录下
EXPERIMENT_NAME = 'yolov8x_ultimate_data_100e_balanced' # 本次训练的具体名称

# --- 核心执行区 ---

def main():
    """主函数：执行模型训练"""
    print("--- 开始模型训练 ---")
    
    # 加载预训练模型
    model = YOLO('yolov8x.pt')

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