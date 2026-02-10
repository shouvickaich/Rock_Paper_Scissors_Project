import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Rock Paper Scissors Detector", layout="centered")

st.title("✊✋✌️ Rock Paper Scissors Detection")
st.write("Use your camera or upload an image to detect Rock / Paper / Scissors")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)

mode = st.radio("Choose input method", ["📸 Camera", "🖼️ Upload Image"])

image = None

if mode == "📸 Camera":
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        image = Image.open(camera_image)

else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Running detection..."):
        results = model.predict(
            source=np.array(image),
            conf=conf_threshold,
            device="cpu"
        )

    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_container_width=True)

    if len(results[0].boxes) == 0:
        st.warning("No gesture detected 😕")
    else:
        st.success("Detection complete ✅")
