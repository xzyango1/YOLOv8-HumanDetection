import cv2
from ultralytics import YOLO

def run_realtime_detection(model_path, camera_id=0, conf_threshold=0.5):
    """
    启动摄像头，使用指定模型进行实时目标检测。
    """
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