"""DeepFER web app. Run: streamlit run app.py"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from common import CLASSES
from emotion import EmotionRecognizer

st.set_page_config(page_title="DeepFER", page_icon="🙂")
st.title("DeepFER: Facial Emotion Recognition")
st.caption("ResNet-18 transfer-learning model trained on FER2013 (7 emotions).")


@st.cache_resource
def get_model():
    return EmotionRecognizer("runs/resnet18.pt")


rec = get_model()
src = st.radio("Input", ["Webcam snapshot", "Upload image"], horizontal=True)
file = st.camera_input("Take a photo") if src == "Webcam snapshot" else st.file_uploader("Image", type=["jpg", "jpeg", "png"])
if file:
    bgr = cv2.cvtColor(np.array(Image.open(file).convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes = rec.detect(gray)
    if len(boxes) == 0:  # e.g. an already-cropped 48x48 face: classify the whole image
        boxes = np.array([[0, 0, gray.shape[1], gray.shape[0]]])
        st.info("No face detected - classifying the whole image.")
    probs = rec.predict_faces(gray, boxes)
    out = bgr.copy()
    for (x, y, w, h), p in zip(boxes, probs):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 3)
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
    for i, p in enumerate(probs):
        st.subheader(f"Face {i + 1}: {CLASSES[int(p.argmax())]} ({p.max():.0%})")
        st.bar_chart({c: float(v) for c, v in zip(CLASSES, p)})
