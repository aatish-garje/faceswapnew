import streamlit as st
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2
import numpy as np
from PIL import Image

# Set Streamlit page layout
st.set_page_config(layout="wide")
st.title("🧑‍🎭 InsightFace Face Swap App")

# Load InsightFace face analyzer
@st.cache_resource
def load_models():
    app = FaceAnalysis(allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)
    return app, swapper

fa, swapper = load_models()

# Face swapping function
def swap_face(src_img, tgt_img):
    src_faces = fa.get(src_img)
    tgt_faces = fa.get(tgt_img)

    if not src_faces or not tgt_faces:
        return None, "No faces found in source or target."

    src_face = src_faces[0]
    tgt_face = tgt_faces[0]

    # Apply swapper
    result = swapper.get(tgt_img, tgt_face, src_face)
    return result, None

# Upload images
src_file = st.sidebar.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"])
tgt_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if src_file and tgt_file:
    src = np.array(Image.open(src_file).convert("RGB"))
    tgt = np.array(Image.open(tgt_file).convert("RGB"))

    if st.sidebar.button("Swap Face"):
        with st.spinner("Swapping..."):
            swapped_img, err = swap_face(src, tgt)
            if err:
                st.error(err)
            else:
                st.image(swapped_img, caption="🎭 Swapped Output", use_column_width=True)
