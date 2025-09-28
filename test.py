from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 加载模型（使用训练好的模型或预训练模型）
model = YOLO('best.pt')  # 或 'best.pt'（如果你训练过）

# 设置输入源
source = 'test_video.mp4' 

# 运行预测
results = model.predict(source, save=True, conf=0.5)

# 显示结果
for i, r in enumerate(results):
    # 方法1：使用matplotlib显示
    im_bgr = r.plot()  # BGR格式图片
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.title(f'Detection Result {i+1}')
    plt.show()
    
    # 方法2：打印检测信息
    print(f"Image {i+1}:")
    for box in r.boxes:
        print(f"  Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}")
        print(f"  Coordinates: {box.xyxy[0].tolist()}")