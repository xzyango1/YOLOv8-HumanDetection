# realtime.py (简约版)
import cv2
from ultralytics import YOLO
from pathlib import Path

# --- 配置区 ---
# 在这里修改所有实时检测参数

# 1. 指向您训练好的、最好的模型权重文件 (.pt)
MODEL_PATH = 'runs/detect/ultimate_model_aggressive_v28/weights/best.pt' # ⚠️ 请务必改为您真实的模型路径！

# 2. 要使用的摄像头ID (0 通常是您电脑的默认内置摄像头)
CAMERA_ID = 0

# 3. 设置置信度阈值
CONFIDENCE_THRESHOLD = 0.25

# --- 核心执行区 ---
def main():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f"❌ 错误: 模型文件不存在 -> {MODEL_PATH}")
        return
        
    model = YOLO(model_file)
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"❌ 错误：无法打开ID为 {CAMERA_ID} 的摄像头。")
        return

    print("🎥 摄像头已启动，请按 'q' 键退出...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("实时安全检测 - 按 'q' 退出", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 程序已成功退出。")

if __name__ == '__main__':
    main()