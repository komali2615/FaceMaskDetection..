# 🧠 Face Mask Detection Using Deep Learning

This project uses a **Convolutional Neural Network (CNN)** and **OpenCV** to detect whether a person is wearing a mask — in real-time via webcam.

## 📂 Project Structure
FaceMaskDetection/
├── dataset/
│ ├── with_mask/
│ └── without_mask/
├── train_mask_detector.py
├── real_time_detection.py
├── requirements.txt
├── setup.py
└── README.md
## 📦 Requirements

Install all dependencies using:
```bash
pip install -r requirements.txt

 How to Use
1.Add images to dataset/with_mask and dataset/without_mask

2.Run:
      python train_mask_detector.py
3.After training, run real-time detection:
     python real_time_detection.py
(It will use your webcam to detect if you're wearing a mask or not. Press Q to quit the webcam window)
Features:
1 Real-time detection via webcam

2 Uses MobileNetV2 for efficient training

3 Alerts whether a mask is detected or not

4 Easily extendable to security systems or IoT
Developed by Komali Koppaka
