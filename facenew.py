import streamlit as st
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from PIL import Image
import onnxruntime

st.title("🖼️ Face Swap (via InsightFace InSwapper)")

# Load face analyzer
fa = FaceAnalysis(allowed_modules=['detection', 'inswapper'])
fa.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode

def swap_face(src_img, tgt_img):
    results = fa.get(src_img)
    if len(results) == 0:
        return None, "No face found in source or target."

    src_obj = results[0]
    tgt_obj = fa.get(tgt_img)[0]

    swapped = fa.inswap(src_img, tgt_img, tgt_obj, src_obj)
    return swapped, None

src_file = st.file_uploader("Upload Source Face Image", type=['jpg','png'])
tgt_file = st.file_uploader("Upload Target Image", type=['jpg','png'])

if src_file and tgt_file:
    src = np.array(Image.open(src_file).convert("RGB"))
    tgt = np.array(Image.open(tgt_file).convert("RGB"))
    swapped, err = swap_face(src, tgt)
    if err:
        st.error(err)
    else:
        st.image(swapped, caption="Swapped Output", use_column_width=True)
