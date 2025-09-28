from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8x.pt")

    results = model.train(
        data="datasets/ULTIMATE_DATASET/data.yaml",
        # --- 激进的性能参数 ---
        epochs=100,  # 总轮次
        patience=25,  # 耐心，多轮次无提升则停止训练，一般在总论次的1/4到1/3之间
        imgsz=640,  # 暂时保持640
        # --- 内存与显存配置 ---
        batch=4,  # 8GB显存对散热系统负载过大，改为4
        workers=8,  # 线程数量
        cache=False,  # 尝试将数据集缓存到内存中以加速训练！
        # --- 其他高级参数 ---
        optimizer="AdamW",  # 显式指定一个优秀的优化器
        name="ultimate_model_aggressive_v2",  # 为这次豪华版训练起个新名字
    )

    print("训练已完成！")
