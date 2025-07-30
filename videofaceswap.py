import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import requests
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# -------------------------------
# 🔄 Download ONNX Model
# -------------------------------
def download_model():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"
    os.makedirs("models", exist_ok=True)

    if not os.path.isfile(model_path):
        st.info("📥 Downloading face swap model...")
        r = requests.get(model_url, stream=True)
        if r.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Model downloaded!")
        else:
            raise Exception(f"Download failed with status {r.status_code}")
    return model_path

# -------------------------------
# 🔧 Load InsightFace Models
# -------------------------------
@st.cache_resource
def load_models():
    model_path = download_model()
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])
    return face_analyzer, swapper

# -------------------------------
# 🧠 Face Swap Logic on Frame
# -------------------------------
def process_video(src_faces, video_path, face_analyzer, swapper, face_mapping):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_analyzer.get(frame)
        for idx, target_face in enumerate(faces):
            if idx in face_mapping:
                src_face_idx = face_mapping[idx]
                if src_face_idx < len(src_faces):
                    try:
                        frame = swapper.get(frame, target_face, src_faces[src_face_idx], paste_back=True)
                    except:
                        continue
        out.write(frame)
    cap.release()
    out.release()
    return output_path

# -------------------------------
# 🎛️ Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🎬 Face Swap Video App (Multi-face Logic)")

st.sidebar.header("🧠 Upload Source Face Images")
src_files = st.sidebar.file_uploader("Upload source faces (multiple allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.header("🎞️ Upload Video File")
video_file = st.sidebar.file_uploader("Upload target video", type=["mp4", "mov", "avi"])

if src_files and video_file:
    st.info("⌛ Loading Models...")
    fa, swapper = load_models()

    # Load and show source faces
    src_faces_all = []
    st.subheader("🧑 Source Faces")
    for i, src_file in enumerate(src_files):
        src_img = np.array(cv2.imdecode(np.frombuffer(src_file.read(), np.uint8), 1))
        faces = fa.get(src_img)
        if faces:
            src_faces_all.append(faces[0])
            st.image(src_img, caption=f"Source Face ID {i}", width=150)
        else:
            st.warning(f"No face found in source image {i+1}")

    # Load video and extract first frame
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("❌ Failed to read video")
    else:
        tgt_faces = fa.get(first_frame)
        st.subheader("🎯 Target Faces in First Frame")
        face_mapping = {}
        cols = st.columns(len(tgt_faces))
        for idx, face in enumerate(tgt_faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            crop = first_frame[y1:y2, x1:x2]
            with cols[idx]:
                st.image(crop, caption=f"Target Face ID {idx}", width=150)
                face_id = st.selectbox(f"Swap with source face?", options=["None"] + list(range(len(src_faces_all))), key=f"map_{idx}")
                if face_id != "None":
                    face_mapping[idx] = int(face_id)

        if st.button("🔄 Start Face Swap on Video"):
            with st.spinner("⏳ Processing video..."):
                output_path = process_video(src_faces_all, video_path, fa, swapper, face_mapping)
                st.success("✅ Video processed successfully!")
                st.video(output_path)
                with open(output_path, "rb") as f:
                    st.download_button("📥 Download Result Video", f, file_name="swapped_video.mp4")
