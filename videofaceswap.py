import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from PIL import Image
import tempfile
import os
import requests
import shutil

st.set_page_config(layout="wide")
st.title("🎥 Multi-Face Video Face Swap (InsightFace)")

# ==============================
# Load models
# ==============================

@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.isfile(model_path):
        with st.spinner("🔽 Downloading face swap model..."):
            r = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        st.success("✅ Model downloaded!")

    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])
    return face_analyzer, swapper

# ==============================
# Extract frames from video
# ==============================

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def write_video(frames, output_path, fps=25):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

# ==============================
# Swap faces in a frame
# ==============================

def swap_faces_in_frame(frame, fa, swapper, source_faces, mapping_dict):
    detected_faces = fa.get(frame)
    swapped = frame.copy()

    for idx, face in enumerate(detected_faces):
        if str(idx) in mapping_dict:
            src_index = int(mapping_dict[str(idx)])
            if src_index >= 0 and src_index < len(source_faces):
                swapped = swapper.get(swapped, face, source_faces[src_index], paste_back=True)

    return swapped

# ==============================
# Streamlit UI
# ==============================

st.sidebar.header("🎯 Upload Media")

video_file = st.sidebar.file_uploader("Upload Target Video", type=["mp4", "mov", "avi"])
src_files = st.sidebar.file_uploader("Upload Source Face Images (multiple allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if video_file and src_files:
    fa, swapper = load_models()

    # Load and analyze source faces
    source_faces = []
    st.subheader("🧑‍🎤 Source Faces Preview")
    src_col = st.columns(len(src_files))
    for i, src in enumerate(src_files):
        img = np.array(Image.open(src).convert("RGB"))
        faces = fa.get(img)
        if len(faces) > 0:
            source_faces.append(faces[0])
            src_col[i].image(img, caption=f"Source ID {i}", use_column_width=True)
        else:
            st.warning(f"No face found in Source Image {i}")

    if len(source_faces) == 0:
        st.error("❌ No valid faces found in uploaded source images.")
        st.stop()

    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Extract sample frame to preview target faces
    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    cap.release()
    if not ret:
        st.error("❌ Could not read video file.")
        st.stop()

    sample_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
    tgt_faces = fa.get(sample_rgb)

    if len(tgt_faces) == 0:
        st.error("❌ No faces detected in video.")
        st.stop()

    st.subheader("🎞️ Target Face IDs in Sample Frame")
    tgt_col = st.columns(len(tgt_faces))
    face_map = {}
    for i, face in enumerate(tgt_faces):
        x1, y1, x2, y2 = list(map(int, face.bbox))
        face_crop = sample_rgb[y1:y2, x1:x2]
        tgt_col[i].image(face_crop, caption=f"Target ID {i}", use_column_width=True)
        face_map[str(i)] = st.sidebar.selectbox(
            f"Swap Target Face {i} with Source Face ID:",
            options=["-1"] + [str(j) for j in range(len(source_faces))],
            key=f"map{i}",
            format_func=lambda x: "❌ Skip" if x == "-1" else f"Source ID {x}"
        )

    if st.button("🎬 Start Face Swap"):
        with st.spinner("Processing video..."):
            frames = extract_frames(video_path)
            swapped_frames = []

            for frame in frames:
                out = swap_faces_in_frame(frame, fa, swapper, source_faces, face_map)
                swapped_frames.append(out)

            output_video = "swapped_output.mp4"
            write_video(swapped_frames, output_video)

            st.success("✅ Face Swap Completed!")
            st.video(output_video)

            with open(output_video, "rb") as f:
                st.download_button("⬇️ Download Result Video", f, file_name="swapped_output.mp4", mime="video/mp4")
