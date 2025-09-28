import cv2
from ultralytics import YOLO

# --- 核心配置区 ---

# 1. 加载我们训练好的、最强大的V2.0终极模型
# 确保这个路径是正确的！
model_path = 'D:/VScode Project/YOLOv8-HumanDetection/runs/detect/ultimate_model_aggressive_v28/weights/best.pt'
model = YOLO(model_path)

# 2. 打开电脑的默认摄像头 (摄像头ID通常为0)
# 如果您有多个摄像头，可以尝试更改为 1, 2, ...
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头。请检查摄像头是否连接或被其他程序占用。")
    exit()

print("摄像头已启动，按 'q' 键退出...")

# --- 主循环：逐帧处理 ---

while True:
    # 1. 从摄像头读取一帧画面
    success, frame = cap.read()

    # 如果成功读取到画面
    if success:
        # 2. 对当前帧运行YOLOv8推理
        # stream=True 建议用于视频或实时流，以获得更好的性能
        results = model.predict(frame, stream=True, conf=0.3, verbose=False)

        # 3. 处理检测结果并进行可视化
        for r in results:
            # 使用YOLOv8内置的 plot() 方法，它会自动绘制边界框和标签
            # 这比我们手动用cv2.rectangle()画图要方便得多
            annotated_frame = r.plot()

            # 4. 在窗口中显示带标注的画面
            cv2.imshow("实时安全检测 - 按 'q' 退出", annotated_frame)

        # 5. 检测按键，如果按下 'q' 键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频流结束或读取失败，也退出循环
        break

# --- 收尾工作 ---

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV创建的窗口
cv2.destroyAllWindows()

print("程序已成功退出。")