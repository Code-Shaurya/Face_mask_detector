# 😷 Face Mask Detection with Live Alert System

A real-time face mask detection system using OpenCV, TensorFlow/Keras, and a webcam. The system identifies whether a person is wearing a mask or not and displays visual alerts accordingly.

---

## 📌 Features
- Real-time face mask detection from webcam video
- Deep learning-based classification using MobileNetV2
- Audio alert for "No Mask" detections (optional)
- Easy to extend with additional alert systems or datasets

---

## 🛠️ Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- MobileNetV2
- NumPy
- imutils

---

## 📁 Project Structure
Face-Mask-Detection/
├── dataset/ # Dataset folder with masked/unmasked images
├── face_detector/ # Face detector using Caffe model
│ ├── deploy.prototxt
│ └── res10_300x300_ssd_iter_140000.caffemodel
├── mask_detector.h5 # Trained face mask classification model
├── detect_mask_video.py # Main real-time detection script
├── train_mask_detector.py # CNN model training script
├── audio_alert.wav # Optional audio alert for no-mask cases
└── README.md # Project documentation

