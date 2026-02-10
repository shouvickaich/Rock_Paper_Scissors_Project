✋✌️✊ Rock–Paper–Scissors Detection using YOLO26n (GPU) & Streamlit

This project implements a GPU-accelerated object detection system for identifying Rock, Paper, and Scissors hand gestures using a custom YOLO26n model, with an interactive Streamlit web application for inference.

🚀 Key Features

Custom YOLO26n object detection model

GPU-based training and inference (CUDA enabled)

Detects Rock / Paper / Scissors hand gestures

Streamlit-based web UI for image inference

Trained on Roboflow dataset

Can be tested using Google Colab (GPU runtime)

🧠 Model Information

Model Name: YOLO26n

Framework: Ultralytics YOLO (custom YAML architecture)

Task: Object Detection

Acceleration: NVIDIA GPU (CUDA)

Input Size: 640 × 640

Classes
0 → rock
1 → paper
2 → scissors

📁 Project Structure
.
├── app.py                 # Streamlit application
├── best.pt                # Trained YOLO26n model (GPU trained)
├── requirements.txt       # Dependencies
├── README.md              # Documentation

🖥️ Hardware & Runtime

Training: GPU (Google Colab / CUDA-enabled system)

Inference:

GPU (Colab / local CUDA)

CPU (Streamlit Cloud – slower)

⚙️ Training Details

Dataset sourced from Roboflow

Data format: YOLO

Training epochs: 50+

Optimizer: Default Ultralytics optimizer

Best model saved automatically as best.pt

Saved at:

runs/detect/trainX/weights/best.pt

▶️ Run Streamlit App (GPU / Local)
pip install -r requirements.txt
streamlit run app.py

☁️ Run on Google Colab (GPU)

Enable GPU runtime

Upload:

app.py

best.pt

Install dependencies

Run Streamlit using ngrok

Access via generated public URL

⚠️ Colab hosting is temporary and session-based.

📦 requirements.txt
streamlit
ultralytics
opencv-python-headless
numpy
Pillow
torch
torchvision

🖼️ Output

Bounding boxes drawn on detected hands

Class label (rock / paper / scissors)

Confidence score per detection

🚧 Known Limitations

Streamlit Cloud does not support GPU

CPU inference is slower

Performance depends on lighting and hand orientation

🔮 Future Enhancements

Real-time video & webcam detection

Confidence threshold slider

Model optimization (TensorRT / ONNX)

Mobile deployment

REST API (FastAPI)

👤 Author

Shouvick Aich
AI | Computer Vision | Deep Learning

⭐ Acknowledgements

Ultralytics YOLO

Roboflow

Streamlit

Google Colab
