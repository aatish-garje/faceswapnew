import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ==============================
# 🧠 Model Loader
# ==============================

@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"

    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.isfile(model_path):
        with st.spinner("🔽 Downloading FaceSwap model..."):
            r = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("✅ Model downloaded!")

    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])

    return face_analyzer, swapper

# ==============================
# 🔄 Face Swapper
# ==============================

def swap_face(src_img, tgt_img, app, swapper):
    src_faces = app.get(src_img)
    tgt_faces = app.get(tgt_img)

    if len(src_faces) == 0:
        return None, "❌ No face found in source image."
    if len(tgt_faces) == 0:
        return None, "❌ No face found in target image."

    swapped = tgt_img.copy()
    for face in tgt_faces:
        swapped = swapper.get(swapped, face, src_faces[0], paste_back=True)

    return swapped, None

# ==============================
# 🎛️ Streamlit App UI
# ==============================

st.set_page_config(layout="wide")
st.title("🧑‍🎤 Realistic Face Swap App (SimSwap - ONNX)")

st.sidebar.header("📤 Upload Images")
src_file = st.sidebar.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"])
tgt_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if src_file and tgt_file:
    src_img = np.array(Image.open(src_file).convert("RGB"))
    tgt_img = np.array(Image.open(tgt_file).convert("RGB"))

    st.subheader("📷 Preview")
    col1, col2 = st.columns(2)
    col1.image(src_img, caption="Source Face", use_column_width=True)
    col2.image(tgt_img, caption="Target Image", use_column_width=True)

    if st.button("🔄 Swap Face"):
        with st.spinner("Running Face Swap..."):
            fa, swapper = load_models()
            result, error = swap_face(src_img, tgt_img, fa, swapper)
            if error:
                st.error(error)
            else:
                st.success("✅ Face Swapped Successfully!")
                st.image(result, caption="🎯 Final Output", use_column_width=True)
