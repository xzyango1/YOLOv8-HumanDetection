import argparse
from pathlib import Path
from ultralytics import YOLO

def predict(model_path, source_path, conf_threshold):
    """
    使用指定的模型对源文件进行预测。
    """
    model_file = Path(model_path)
    source_file = Path(source_path)

    if not model_file.exists():
        print(f"❌ 错误: 模型文件不存在 -> {model_path}")
        return
    if not source_file.exists():
        print(f"❌ 错误: 源文件/目录不存在 -> {source_path}")
        return
        
    print(f"🔍 加载模型 '{model_path}'...")
    model = YOLO(model_path)
    
    print(f"🚀 开始对 '{source_path}' 进行预测...")
    results = model.predict(source=source_path, save=True, conf=conf_threshold)
    
    # 打印结果的保存路径
    if isinstance(results, list):
        print(f"✅ 预测完成！结果保存在: {results[0].save_dir}")
    else:
        print("✅ 预测完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 模型预测脚本")
    parser.add_argument('--model', type=str, required=True, help="指向.pt模型文件的路径")
    parser.add_argument('--source', type=str, required=True, help="指向待预测的图片或视频文件")
    parser.add_argument('--conf', type=float, default=0.5, help="检测结果的置信度阈值")
    args = parser.parse_args()
    
    predict(args.model, args.source, args.conf)