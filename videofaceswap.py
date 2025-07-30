import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import requests
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from PIL import Image

# -------------------------------
# Download Model if Not Present
# -------------------------------
@st.cache_resource
def load_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"

    os.makedirs("models", exist_ok=True)
    if not os.path.exists(model_path):
        os.remove(model_path)
    with st.spinner("🔽 Downloading FaceSwap model..."):
        r = requests.get(model_url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model downloaded!")

    face_analyzer = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, providers=["CPUExecutionProvider"])
    return face_analyzer, swapper

# -------------------------------
# Extract Frames from Video
# -------------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# -------------------------------
# Swap Faces in Frame
# -------------------------------
def swap_faces_in_frame(frame, src_faces, tgt_faces, swapper, face_map):
    output = frame.copy()
    for target_id, source_id in face_map.items():
        if target_id < len(tgt_faces) and source_id < len(src_faces):
            output = swapper.get(output, tgt_faces[target_id], src_faces[source_id], paste_back=True)
    return output

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🎥 Video-Based Face Swap App (Multi-Face ID Mapping)")

src_files = st.sidebar.file_uploader("Upload Source Image(s)", type=["jpg", "png"], accept_multiple_files=True)
video_file = st.sidebar.file_uploader("Upload Target Video", type=["mp4", "mov", "avi"])

if src_files and video_file:
    st.info("⌛ Loading Models...")
    fa, swapper = load_models()

    # Load and show source faces
    src_faces_all = []
    src_face_labels = []
    for idx, file in enumerate(src_files):
        img = np.array(Image.open(file).convert("RGB"))
        faces = fa.get(img)
        for fid, f in enumerate(faces):
            src_faces_all.append(f)
            src_face_labels.append(f"Source {idx}-Face {fid}")
            st.sidebar.image(img[int(f.bbox[1]):int(f.bbox[3]), int(f.bbox[0]):int(f.bbox[2])], caption=f"Src {idx} - Face {fid}", width=100)

    # Read video and extract target faces from first frame
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    frames = extract_frames(tfile.name)
    tgt_faces = fa.get(frames[0])

    st.subheader("🎭 Face ID Mapping")
    face_map = {}
    for i, face in enumerate(tgt_faces):
        st.image(frames[0][int(face.bbox[1]):int(face.bbox[3]), int(face.bbox[0]):int(face.bbox[2])],
                 caption=f"Target Face {i}", width=100)
        sel = st.selectbox(f"🟢 Map Target Face {i} to:", options=["None"] + src_face_labels, key=f"map_{i}")
        if sel != "None":
            sel_index = src_face_labels.index(sel)
            face_map[i] = sel_index

    if st.button("🎬 Start Face Swap"):
        with st.spinner("Processing video..."):
            out_frames = []
            for frame in frames:
                tgt_in_frame = fa.get(frame)
                swapped = swap_faces_in_frame(frame, src_faces_all, tgt_in_frame, swapper, face_map)
                out_frames.append(swapped)

            # Save video
            height, width, _ = out_frames[0].shape
            out_path = "output_swapped_video.mp4"
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))
            for f in out_frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()

        st.success("✅ Done! Download your video below.")
        with open(out_path, "rb") as file:
            st.download_button("📥 Download Swapped Video", data=file, file_name="face_swapped_output.mp4", mime="video/mp4")
