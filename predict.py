# predict.py (音画同步版 - WMV2兼容修复)
import os
import subprocess
from pathlib import Path
from ultralytics import YOLO
import shutil

# --- 配置区 ---
# 在这里修改所有预测参数

# 1. 指定模型路径
# 默认使用下载的预训练模型。请确保 'best-complete.int8.onnx' 文件已放置在项目根目录。
MODEL_PATH = 'best-complete.int8.onnx'

# [备选] 如果你想使用自己训练的模型，请注释掉上面一行，并使用下面这段：
# EXPERIMENT_FOLDER = 'your_experiment_folder_name' # ⚠️ 请改为您真实的实验文件夹名称！
# MODEL_PATH = f'runs/detect/{EXPERIMENT_FOLDER}/weights/best.pt'

# 2. 指定您要进行预测的原始视频文件路径
SOURCE_PATH = 'assets/test_video.mp4'

# 3. 设置置信度阈值
CONFIDENCE_THRESHOLD = 0.5

# 4. 视频编码设置
# 'fast' - 快速编码，质量稍低但速度快 (推荐)
# 'high_quality' - 高质量编码，速度较慢
ENCODE_MODE = 'fast'

# --- 核心执行区 ---

def get_next_predict_dir_name():
    """
    计算下一个预测文件夹的名称，例如 predict, predict2, predict3...
    """
    base_dir = Path('runs/detect')
    if not base_dir.exists():
        return 'predict'
    
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('predict')]
    
    if not existing_dirs:
        return 'predict'

    max_num = 0
    # 检查 'predict' 文件夹是否存在
    predict_exists = any(d.name == 'predict' for d in existing_dirs)
    if predict_exists:
        max_num = 1 # 如果 'predict' 存在，我们至少从 'predict2' 开始

    # 检查 'predictX' 格式的文件夹
    for d in existing_dirs:
        if d.name.startswith('predict') and d.name[7:].isdigit():
            num = int(d.name[7:])
            if num > max_num:
                max_num = num
    
    if max_num == 0 and not predict_exists:
        return 'predict'
    else:
        return f'predict{max_num + 1}'

def main():
    """
    主函数：执行YOLOv8预测，并使用FFmpeg自动合并原始音轨。
    """
    model_file = Path(MODEL_PATH)
    source_file = Path(SOURCE_PATH)

    if not model_file.exists():
        print(f"❌ 错误: 找不到模型文件 -> {MODEL_PATH}")
        return
    if not source_file.is_file():
        print(f"❌ 错误: 找不到源视频文件 -> {SOURCE_PATH}")
        return

    # --- 阶段一: 使用YOLOv8进行无声视频预测 ---
    print("="*60 + "\n🚀 阶段一: 开始进行YOLOv8视频预测 (此过程将生成无声视频)...\n" + "="*60)
    
    model = YOLO(model_file)

    predict_name = get_next_predict_dir_name()

    results = model.predict(
        source=str(source_file), 
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        project='runs/detect', # 指定根目录
        name=predict_name,    # 指定基础名称，YOLO会自动处理增量
    )
    # 尝试定位生成的预测视频文件
    try:
        # ultralytics 8.1.0+ 版本后，推荐使用 results[0].save_dir
        if hasattr(results[0], 'save_dir'):
            output_dir = Path(results[0].save_dir)
        else:
            # 兼容旧版本，手动构造路径
            # 注意：这需要你知道 YOLO 的确切命名规则
            print("⚠️警告：无法从结果对象中直接找到 save_dir，尝试手动构造路径。")
            # 这里我们直接使用我们计算出的 predict_name
            output_dir = Path(f'runs/detect/{predict_name}')

        if not output_dir.exists():
            raise FileNotFoundError(f"预测目录 '{output_dir}' 未被创建。")
            
        print(f"💡 预测结果已保存至: {output_dir.resolve()}")

        # 在新的输出目录中查找生成的视频文件
        possible_extensions = ['.avi', '.mp4', '.mkv']
        processed_video_path = None
        
        for ext in possible_extensions:
            candidate = output_dir / (source_file.stem + ext)
            if candidate.exists():
                processed_video_path = candidate
                break
        
        if processed_video_path is None:
            if output_dir.exists():
                files = list(output_dir.iterdir())
                for f in files:
                    if f.suffix.lower() in ['.avi', '.mp4', '.mkv']:
                        processed_video_path = f
                        break
        
        if processed_video_path is None or not processed_video_path.exists():
            raise FileNotFoundError("YOLOv8未能成功生成预测视频。")
            
        print(f"✅ 阶段一完成！无声预测视频已生成: {processed_video_path}")

    except Exception as e:
        print(f"❌ 错误：在定位预测视频时出错。 {e}")
        # 在出错时，可选：清理可能已创建的文件夹
        if 'output_dir' in locals() and output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"🗑️ 已清理不完整的预测文件夹: {output_dir}")
        return

    # --- 阶段二: 使用FFmpeg合并音轨 ---
    print("\n" + "="*60 + "\n🚀 阶段二: 开始使用FFmpeg合并原始音轨...\n" + "="*60)

    final_output_path = Path(f"{source_file.stem}_{predict_name}_processed.mp4")

    if ENCODE_MODE == 'fast':
        video_codec = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
        print("   - 使用模式: 快速编码 (推荐)")
    else:
        video_codec = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '18']
        print("   - 使用模式: 高质量编码 (较慢)")

    ffmpeg_command = [
        'ffmpeg',
        '-i', str(processed_video_path),
        '-i', str(source_file),
        *video_codec,
        '-c:a', 'aac',
        '-b:a', '128k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y',
        str(final_output_path)
    ]
    
    try:
        print(f"   - 正在执行FFmpeg转码...")
        print(f"   - 这可能需要一些时间，请耐心等待...")
        
        result = subprocess.run(
            ffmpeg_command, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        print("\n" + "="*60)
        print("🎉 恭喜！音画同步的最终视频已生成！")
        print(f"   - 文件保存在: {final_output_path.resolve()}")
        
        file_size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
        print(f"   - 文件大小: {file_size_mb:.2f} MB")
        print("="*60)

    except FileNotFoundError:
        print("❌ 错误: FFmpeg未找到。")
        print("\n📥 安装FFmpeg的方法:")
        print("   1. 访问 https://ffmpeg.org/download.html")
        print("   2. 下载Windows版本")
        print("   3. 解压到C:\\ffmpeg")
        print("   4. 将C:\\ffmpeg\\bin添加到系统环境变量Path中")
        
    except subprocess.CalledProcessError as e:
        print("❌ 错误: FFmpeg在执行过程中出错。")
        print(f"\n详细错误信息:")
        print(f"{e.stderr}")
        
        print("\n💡 可能的解决方案:")
        print("   1. 尝试更改ENCODE_MODE为'high_quality'")
        print("   2. 检查输入视频是否完整")
        print("   3. 确保有足够的磁盘空间")
        print("   4. 确保FFmpeg已正确安装并配置在系统Path中")

if __name__ == '__main__':
    print("="*60)
    print("🔧 YOLOv8视频预测 + 音频合并工具")
    print("="*60)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        version_line = result.stdout.split('\n')[0]
        print(f"✅ FFmpeg已安装: {version_line}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️ 警告: 未检测到FFmpeg，音频合并功能将无法使用。")
        print("   视频预测仍可正常进行，但最终视频将没有声音。")
        response = input("\n是否继续？(y/n): ")
        if response.lower() != 'y':
            print("程序已取消。")
            exit(0)
    
    print("")
    main()