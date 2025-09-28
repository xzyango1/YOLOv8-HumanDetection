import argparse
from ultralytics import YOLO

def train_model(data_config_path):
    """
    根据指定的数据集配置文件加载预训练模型并开始训练。
    训练参数（如epochs, batch等）也应定义在YAML文件中。
    """
    # 使用yolov8x作为预训练模型起点
    model = YOLO('yolov8x.pt') 
    
    print(f"📄 使用配置文件 '{data_config_path}' 开始训练...")
    results = model.train(data=data_config_path)
    print(f"✅ 训练完成！结果保存在: {results.save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 模型训练脚本")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/datasets/ultimate_dataset.yaml', 
        help="指向数据集和训练参数的.yaml配置文件路径"
    )
    args = parser.parse_args()
    
    train_model(args.config)