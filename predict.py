from ultralytics import YOLO

if __name__ == "__main__":
    # --- 核心：加载我们最新、最强大的V2.0终极模型 ---
    model_path = "D:/VScode Project/YOLOv8-HumanDetection/runs/detect/ultimate_model_aggressive_v28/weights/best.pt"
    model = YOLO(model_path)

    # --- 指定一个您之前测试过的、V1.0模型表现不佳的视频 ---
    # 最好是一个包含很多全身人物的视频
    source_path = "videos/广州街拍.mp4"

    # --- 运行预测 ---
    # 我们可以稍微提高置信度，因为这个模型更强大
    results = model.predict(source=source_path, save=True, conf=0.3)

    print(f"\n终极模型预测完成！结果已保存到: {results[0].save_dir}")
