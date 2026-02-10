import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Rock Paper Scissors Detector", layout="centered")

st.title("✋✌️✊ Rock–Paper–Scissors Detector")
st.write("Upload an image and the YOLO model will detect Rock, Paper, or Scissors.")

# Load model (cached so it loads once)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Running detection..."):
            results = model(
                np.array(image),
                conf=0.25
            )

            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Prediction", use_container_width=True)

            # Show detections
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                st.success("Detections found!")
            else:
                st.warning("No objects detected.")