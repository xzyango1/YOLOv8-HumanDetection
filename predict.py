# predict.py (简约版)
from ultralytics import YOLO
from pathlib import Path

# --- 配置区 ---
# 在这里修改所有预测参数

# 1. 指向您训练好的、最好的模型权重文件 (.pt)
# 例如: 'runs/detect/yolov8x_ultimate_data_100e_balanced/weights/best.pt'
MODEL_PATH = 'runs/detect/YOUR_PROJECT_NAME/YOUR_EXPERIMENT_NAME/weights/best.pt' # ⚠️ 请务必改为您真实的模型路径！

# 2. 指定您要进行预测的图片或视频文件路径
SOURCE_PATH = 'assets/test_video.mp4'

# 3. 设置置信度阈值 (只显示高于此分数的检测结果，范围0-1，建议0.25-0.5之间)
CONFIDENCE_THRESHOLD = 0.3

# --- 核心执行区 ---

def main():
    """主函数：执行预测"""
    model_file = Path(MODEL_PATH)
    source_file = Path(SOURCE_PATH)

    if not model_file.exists():
        print(f"❌ 错误: 模型文件不存在 -> {MODEL_PATH}")
        return
    if not source_file.exists():
        print(f"❌ 错误: 源文件/目录不存在 -> {SOURCE_PATH}")
        return

    print(f"🔍 加载模型 '{model_file.name}'...")
    model = YOLO(model_file)

    print(f"🚀 开始对 '{source_file.name}' 进行预测...")
    results = model.predict(source=source_file, save=True, conf=CONFIDENCE_THRESHOLD)
    
    if results and hasattr(results[0], 'save_dir') and results[0].save_dir:
        print(f"✅ 预测完成！结果已保存在: {results[0].save_dir}")
    else:
        print("✅ 预测完成！但未能获取到保存路径。请检查 'runs/detect/' 目录。")

if __name__ == '__main__':
    main()