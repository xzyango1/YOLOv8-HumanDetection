import os
import shutil
from pathlib import Path

# ==============================================================================
# --- ⚠️ 警告：请仔细阅读！ ---
#
# 本脚本将对您的项目结构进行一次性的、不可逆的重构操作。
# 它会自动创建新目录、移动文件、并用新的模板代码覆盖您现有的核心脚本。
#
# 在运行前，强烈建议您：
# 1. 确保您当前的工作已经通过 `git commit` 保存。
# 2. 备份您整个项目文件夹，以防万一。
#
# 如果您已阅读并理解以上内容，请在下面的终端提示中输入 "yes" 来继续。
# ==============================================================================

# --- 配置区 ---
PROJECT_ROOT = Path(__file__).resolve().parent

# 定义新的目录结构
DIRECTORIES = [
    "assets",
    "configs",
    "configs/datasets",
    "data_preparation",
    "notebooks",
]

# 定义需要移动的文件及其目标位置
FILES_TO_MOVE = {
    "merge_and_verify.py": "data_preparation/",
    "remap_labels.py": "data_preparation/",
}

# 定义需要移动的探索性文件和媒体文件
NOTEBOOK_EXTENSIONS = ['.ipynb']
ASSET_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']

# --- 新文件内容定义区 ---

# 内容：main.py
MAIN_PY_CONTENT = """
import argparse
from train import train_model
from predict import predict
from realtime import run_realtime_detection

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 安全帽与人体检测项目主入口")
    subparsers = parser.add_subparsers(dest='command', required=True, help="可用的命令")

    # --- 'train' 命令 ---
    parser_train = subparsers.add_parser('train', help='根据YAML配置文件训练一个新的模型')
    parser_train.add_argument('--config', type=str, required=True, help="指向数据集和训练参数的.yaml配置文件路径")

    # --- 'predict' 命令 ---
    parser_predict = subparsers.add_parser('predict', help='使用已训练好的模型进行图片或视频预测')
    parser_predict.add_argument('--model', type=str, required=True, help="指向.pt模型文件的路径")
    parser_predict.add_argument('--source', type=str, required=True, help="指向待预测的图片或视频文件")
    parser_predict.add_argument('--conf', type=float, default=0.5, help="检测结果的置信度阈值")

    # --- 'realtime' 命令 ---
    parser_realtime = subparsers.add_parser('realtime', help='启动摄像头进行实时检测')
    parser_realtime.add_argument('--model', type=str, required=True, help="指向.pt模型文件的路径")
    parser_realtime.add_argument('--camera-id', type=int, default=0, help="要使用的摄像头ID (通常为0)")
    parser_realtime.add_argument('--conf', type=float, default=0.5, help="检测结果的置信度阈值")

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args.config)
    elif args.command == 'predict':
        predict(args.model, args.source, args.conf)
    elif args.command == 'realtime':
        run_realtime_detection(args.model, args.camera_id, args.conf)

if __name__ == '__main__':
    main()
"""

# 内容：train.py
TRAIN_PY_CONTENT = """
import argparse
from ultralytics import YOLO

def train_model(data_config_path):
    \"\"\"
    根据指定的数据集配置文件加载预训练模型并开始训练。
    训练参数（如epochs, batch等）也应定义在YAML文件中。
    \"\"\"
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
"""

# 内容：predict.py
PREDICT_PY_CONTENT = """
import argparse
from pathlib import Path
from ultralytics import YOLO

def predict(model_path, source_path, conf_threshold):
    \"\"\"
    使用指定的模型对源文件进行预测。
    \"\"\"
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
"""

# 内容：realtime_app.py 被重命名为 realtime.py
REALTIME_PY_CONTENT = """
import cv2
from ultralytics import YOLO

def run_realtime_detection(model_path, camera_id=0, conf_threshold=0.5):
    \"\"\"
    启动摄像头，使用指定模型进行实时目标检测。
    \"\"\"
    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"❌ 错误：无法打开ID为 {camera_id} 的摄像头。")
        return

    print("🎥 摄像头已启动，请按 'q' 键退出...")

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ 无法从摄像头读取画面。")
            break

        # 对当前帧运行YOLOv8推理 (verbose=False关闭了冗长的输出)
        results = model.predict(frame, stream=True, conf=conf_threshold, verbose=False)

        # 处理并显示结果
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow(f"实时安全检测 (模型: {Path(model_path).name}) - 按 'q' 退出", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 程序已成功退出。")
"""

# 内容：configs/datasets/ultimate_dataset.yaml
YAML_CONFIG_CONTENT = """
# 数据集配置文件
# 路径相对于项目根目录

path: ../datasets/ULTIMATE_DATASET
train: train/images
val: valid/images
test: test/images

# 类别定义
nc: 3
names: ['head', 'helmet', 'person']

# --- 训练参数 ---
# 在这里调整训练配置，而无需修改.py文件
epochs: 100
batch: 4
patience: 25
workers: 8
device: 0
name: yolov8x_ultimate_data_100e_balanced
cache: false
"""

# 内容：requirements.txt
REQUIREMENTS_TXT_CONTENT = """
# 项目核心依赖
# 请注意：torch需要根据您的CUDA版本手动安装

ultralytics
opencv-python
"""

# --- 脚本执行区 ---

def create_directories():
    print("--- 1. 创建新目录结构 ---")
    for directory in DIRECTORIES:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 已创建或确认存在: {dir_path}")

def move_files():
    print("\n--- 2. 移动和整理文件 ---")
    
    # 移动指定的工具脚本
    for src, dst_dir in FILES_TO_MOVE.items():
        src_path = PROJECT_ROOT / src
        dst_path = PROJECT_ROOT / dst_dir / src
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"🚚 已移动: {src} -> {dst_dir}")

    # 移动探索性文件和媒体文件
    for item in PROJECT_ROOT.iterdir():
        if item.is_file():
            if item.suffix in NOTEBOOK_EXTENSIONS:
                shutil.move(str(item), str(PROJECT_ROOT / "notebooks" / item.name))
                print(f"🚚 已移动Notebook: {item.name} -> notebooks/")
            elif item.suffix in ASSET_EXTENSIONS:
                shutil.move(str(item), str(PROJECT_ROOT / "assets" / item.name))
                print(f"🚚 已移动资源文件: {item.name} -> assets/")

def create_and_overwrite_files():
    print("\n--- 3. 创建和重构核心文件 ---")

    files_to_create = {
        "main.py": MAIN_PY_CONTENT,
        "train.py": TRAIN_PY_CONTENT,
        "predict.py": PREDICT_PY_CONTENT,
        "realtime.py": REALTIME_PY_CONTENT, # 将realtime_app重命名
        "configs/datasets/ultimate_dataset.yaml": YAML_CONFIG_CONTENT,
        "requirements.txt": REQUIREMENTS_TXT_CONTENT,
        "data_preparation/__init__.py": "" # 创建一个空的__init__.py使之成为一个包
    }

    for file_path_str, content in files_to_create.items():
        file_path = PROJECT_ROOT / file_path_str
        file_path.write_text(content.strip(), encoding='utf-8')
        print(f"📄 已创建/覆盖: {file_path_str}")

    # 删除旧的realtime_app.py
    old_realtime_app = PROJECT_ROOT / "realtime_app.py"
    if old_realtime_app.exists():
        old_realtime_app.unlink()
        print(f"🗑️ 已删除旧文件: realtime_app.py")


def main():
    """主执行函数"""
    print("=" * 60)
    print("🚀 开始执行项目专业级重构... 🚀")
    print("=" * 60)
    
    # 安全确认
    confirm = input("⚠️ 本操作将修改您的项目结构，是否继续? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ 操作已取消。")
        return

    create_directories()
    move_files()
    create_and_overwrite_files()

    print("\n" + "=" * 60)
    print("🎉 重构完成！您的项目现在拥有一个更专业、更清晰的结构。")
    print("下一步建议:")
    print("1. 仔细检查 `README.md` 并根据您的项目故事进行个性化修改。")
    print("2. 运行 `git status` 查看所有更改，然后进行一次新的提交。")
    print("3. 尝试使用新的命令行入口: `python main.py --help`")
    print("=" * 60)


if __name__ == "__main__":
    main()