import sys
import cv2
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import threading
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt,QTimer

# 加载预训练的模型和特征提取器
model_name = "./handmodels"  # 确保路径正确
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# 使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# 共享变量和锁
frame = None
frame_lock = threading.Lock()
stop_processing = False
prediction = "未检测到手势"

class HandGestureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Hand Gesture Recognition")
        self.setGeometry(100, 100, 300, 100)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_label(self, text):
        self.label.setText(text)

def process_frame():
    global frame, prediction
    while not stop_processing:
        with frame_lock:
            if frame is None:
                continue
            # 复制帧以防止并发修改
            local_frame = frame.copy()

        # 将图像转换为模型输入格式
        inputs = processor(images=local_frame, return_tensors="pt").to(device)

        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取预测结果
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # 更新预测结果
        prediction = f"Prediction: {predicted_class}"

        # 限制处理速度
        time.sleep(0.1)  # 每秒处理10帧

app = QApplication(sys.argv)
window = HandGestureWindow()
window.show()

# 启动处理线程
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()

frame_counter = 0

def update_gui():
    global prediction
    if frame_counter % 5 == 0:
        window.update_label(prediction)
    # 继续调用自己以更新GUI
    QTimer.singleShot(100, update_gui)

# 启动GUI更新
QTimer.singleShot(100, update_gui)

while True:
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret:
        break

    # 每隔5帧处理一次
    frame_counter += 1
    if frame_counter % 5 == 0:
        with frame_lock:
            frame_copy = frame.copy()

    # 显示原始帧
    cv2.imshow('Hand Gesture Recognition', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_processing = True
        break

# 等待处理线程结束
processing_thread.join()

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
sys.exit(app.exec_())
