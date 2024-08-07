# 手势识别 Hand Gesture Recognition

本项目实现了一个基于视觉Transformer（ViT）模型的手势识别系统，结合OpenCV进行视频处理，并使用PyQt5创建图形用户界面（GUI）。  
This project implements a hand gesture recognition system based on the Vision Transformer (ViT) model, using OpenCV for video processing and PyQt5 for creating a graphical user interface (GUI).

## 特点 Features

- **实时手势识别**：通过摄像头捕获视频帧，并使用视觉Transformer模型实时分类手势。  
  **Real-time Hand Gesture Recognition**: Capture video frames through the camera and use the Vision Transformer model to classify gestures in real-time.
- **GPU加速**：如果可用，使用GPU进行更快的推理。  
  **GPU Acceleration**: Use GPU for faster inference if available.
- **图形用户界面**：使用PyQt5显示识别结果，并通过OpenCV显示实时视频流。  
  **Graphical User Interface**: Display recognition results using PyQt5 and show real-time video stream via OpenCV.

## 先决条件 Prerequisites

- Python 3.x
- PyTorch
- Transformers
- OpenCV
- PyQt5

## 安装 Installation

1. **克隆仓库** Clone the repository:
   ```bash
   git clone https://github.com/ice-lemon/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **安装所需的包** Install required packages:
   ```bash
   pip install torch torchvision transformers opencv-python PyQt5
   ```

3. **下载预训练模型** Download the pre-trained model:
   请确保预训练的手势识别模型位于`./hand-gestures`目录中。你可以从[Hugging Face模型库](https://huggingface.co/dima806/hand_gestures_image_detection/tree/main)下载该模型。  
   Ensure the pre-trained hand gesture recognition model is located in the `./hand-gestures` directory. You can download the model from the [Hugging Face Model Hub](https://huggingface.co/dima806/hand_gestures_image_detection/tree/main).
     ```bash
     config.json
     model.safetensors
     preprocessor_config.json
   ```

## 使用方法 Usage

1. **运行脚本** Run the script:
   ```bash
   python hand_gesture_recognition.py
   ```

2. **与GUI交互** Interact with the GUI:
   - GUI将显示识别的手势。  
     The GUI will display the recognized gestures.
   - 实时视频流将在一个单独的OpenCV窗口中显示。  
     The real-time video stream will be displayed in a separate OpenCV window.
   - 按下 'q' 键退出应用程序。  
     Press 'q' to exit the application.

## 代码解释 Code Explanation

### 导入必要的库 Import necessary libraries

```python
import sys
import cv2
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import threading
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
```

### 加载预训练的模型和处理器 Load the pre-trained model and processor

```python
model_name = "./hand-gestures"  # 确保路径正确 Ensure the path is correct
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
```

### 使用GPU加速 Use GPU for acceleration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 打开摄像头 Open the camera

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
```

### 共享变量和锁 Shared variables and lock

```python
frame = None
frame_lock = threading.Lock()
stop_processing = False
prediction = "未检测到手势"  # No gesture detected
```

### 定义手势识别窗口类 Define the hand gesture recognition window class

```python
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
```

### 定义处理帧的函数 Define the function to process frames

```python
def process_frame():
    global frame, prediction
    while not stop_processing:
        with frame_lock:
            if frame is None:
                continue
            local_frame = frame.copy()

        inputs = processor(images=local_frame, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        prediction = f"Prediction: {predicted_class}"

        time.sleep(0.1)
```

### 启动PyQt5应用程序 Start the PyQt5 application

```python
app = QApplication(sys.argv)
window = HandGestureWindow()
window.show()
```

### 启动处理线程 Start the processing thread

```python
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()
```

### 更新GUI Update the GUI

```python
frame_counter = 0

def update_gui():
    global prediction
    window.update_label(prediction)
    QTimer.singleShot(100, update_gui)

QTimer.singleShot(100, update_gui)
```

### 主循环 Main loop

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 5 == 0:
        with frame_lock:
            frame_copy = frame.copy()

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_processing = True
        break

processing_thread.join()

cap.release()
cv2.destroyAllWindows()
sys.exit(app.exec_())
```

## 许可证 License

本项目采用MIT许可证。  
This project is licensed under the MIT License.

---

欢迎通过提交问题或拉取请求来为本项目做出贡献。如有重大更改，请先打开一个问题以讨论您想要更改的内容。  
Contributions are welcome through issue submissions or pull requests. For major changes, please open an issue first to discuss what you would like to change.