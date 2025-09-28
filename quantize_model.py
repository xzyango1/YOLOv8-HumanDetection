# quantize_model.py (最终、最稳固版 - 修正路径问题)

import os # <--- 新增导入
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO

try:
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
except ImportError:
    print("错误：无法从 onnxruntime.quantization 导入模块。")
    print("请确保您的 onnxruntime 版本较新 (建议 >= 1.16.0)。")
    print("您可以尝试运行: pip install --upgrade onnxruntime")
    exit()

# --- 配置区 ---
FP32_MODEL_PATH = 'best-complete.pt'

# --- 核心修正：使用os.path.join构建绝对正确的路径 ---
CALIBRATION_DATA_DIR = os.path.join('datasets', 'calibration_data', 'images')

NUM_CALIBRATION_IMAGES = 500

# --- 核心执行区 (后续代码无需任何修改) ---
class ONNXCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir, num_images=100, input_name='images'):
        # 确保路径存在
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"校准数据目录不存在: {data_dir}")
        
        self.image_paths = list(Path(data_dir).glob('*.*'))[:num_images]
        self.input_name = input_name
        self.data_iter = iter(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在目录下没有找到任何图片文件: {data_dir}")
            
        print(f"INFO: 成功找到 {len(self.image_paths)} 张图片用于校准。")

    def get_next(self):
        try:
            image_path = next(self.data_iter)
            img = cv2.imread(str(image_path))
            img = cv2.resize(img, (640, 640))
            img = img.astype(np.float32) / 255.0
            img_tensor = np.transpose(img[np.newaxis, ...], (0, 3, 1, 2))
            return {self.input_name: img_tensor}
        except StopIteration:
            return None

def main():
    model_path = Path(FP32_MODEL_PATH)
    if not model_path.exists():
        print(f"❌ 错误: 原始模型文件不存在 -> {model_path}")
        return

    onnx_path = model_path.with_suffix('.onnx')
    quantized_onnx_path = model_path.with_suffix('.int8.onnx')

    # ... (后续main函数代码保持不变) ...
    # --- 阶段一: 导出为 ONNX 格式 ---
    print("="*60 + f"\n🚀 阶段一: 导出为 ONNX 格式...\n" + "="*60)
    try:
        model = YOLO(model_path)
        model.export(format='onnx', opset=12)
        print(f"✅ ONNX 导出成功！文件保存在: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        return

    # --- 阶段二: 使用 ONNXRuntime 进行 INT8 量化 ---
    print("\n" + "="*60 + f"\n🚀 阶段二: 开始 INT8 静态量化...\n" + "="*60)
    try:
        calibration_data_reader = ONNXCalibrationDataReader(CALIBRATION_DATA_DIR, NUM_CALIBRATION_IMAGES)
        
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(quantized_onnx_path),
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        print("\n" + "="*60 + "\n🎉 恭喜！模型量化成功！\n" + "="*60)
        print(f"   - 原始模型 ({model_path.name}): {model_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"   - 量化后模型 ({quantized_onnx_path.name}): {quantized_onnx_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"   - 文件保存在: {quantized_onnx_path}")

    except Exception as e:
        print(f"❌ ONNX 量化失败: {e}")

if __name__ == '__main__':
    main()