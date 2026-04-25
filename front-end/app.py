import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import base64
import io
import tempfile
import time
from collections import defaultdict
from pathlib import Path
import serial

import cv2
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import trimesh
from PIL import Image
from streamlit.components.v1 import html
from ultralytics import YOLO
import serial.tools.list_ports

from cut_optimizer import optimize_cut_shape

BACKEND_URL = "http://127.0.0.1:5000"

@st.cache_resource
def get_arduino():
    ports = list(serial.tools.list_ports.comports())
      
    arduino_port = None
    
    for p in ports:
        device_name = p.device
        # On Mac, Arduinos almost always contain 'usbmodem' or 'usbserial'
        if "usbmodem" in device_name.lower() or "usbserial" in device_name.lower():
            arduino_port = device_name
            break
            
    if arduino_port:
        try:
            # Reconnect logic
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2) # Vital for Arduino: it resets when the port opens
            st.sidebar.success(f"✅ Connected: {arduino_port}")
            return ser
        except Exception as e:
            st.sidebar.error(f"❌ Found {arduino_port} but failed: {e}")
            return None
    else:
        st.sidebar.warning("No Arduino detected. Check USB cable.")
        return None

arduino = get_arduino()

# --- Fix for macOS M1/M2 mutex crash ---
torch.multiprocessing.set_start_method("spawn", force=True)
torch.set_num_threads(1)

st.set_page_config(page_title="Gem Dimensions + Inclusions", layout="wide")

if "show_3d_ui" not in st.session_state:
    st.session_state.show_3d_ui = False

# --------- weights path ----------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())
WEIGHTS_PATH = str((SCRIPT_DIR / "best.pt").resolve())

# --------- Styling ----------
st.markdown(
    """
    <style>
      .gradient-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white; padding: 18px 24px; border-radius: 18px;
        box-shadow: 0 10px 24px rgba(99,102,241,0.3);
      }
      .thumb {
        border-radius: 14px; box-shadow: 0 10px 24px rgba(0,0,0,0.12);
        overflow: hidden; background: #0b0b0b;
      }
      .subtle { color: #6b7280; font-size: 0.9rem; }
      .block-container { padding-top: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gradient-header">
      <h2 style="margin:0;">💎 Gem Dimensions + Inclusions</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")


# --------- Model loader ----------
@st.cache_resource(show_spinner=False)
def load_model_cached(wpath: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(wpath)
    return model.to(device)


def get_name_map(model: YOLO):
    if isinstance(model.names, dict):
        return model.names
    return {i: n for i, n in enumerate(model.names)}


def predict(model: YOLO, pil_img: Image.Image, conf: float):
    results = model.predict(pil_img, conf=conf, verbose=False)
    return results[0]


def _ensure_mask_size(mask_bool: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Ensure mask is (H,W) == target_hw."""
    Ht, Wt = target_hw
    if mask_bool.shape[:2] == (Ht, Wt):
        return mask_bool
    try:
        resized = cv2.resize(
            mask_bool.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        return resized
    except Exception:
        return (
            np.array(
                Image.fromarray(mask_bool.astype(np.uint8) * 255).resize(
                    (Wt, Ht), resample=Image.NEAREST
                )
            )
            > 0
        )


def combine_masks_for_ids(r, wanted_ids: set[int], target_hw: tuple[int, int]):
    """Return combined boolean mask (H,W) for instances with cls in wanted_ids."""
    if r.masks is None or r.boxes is None or len(r.boxes) == 0 or not wanted_ids:
        return None
    cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
    masks = r.masks.data.detach().cpu().numpy()
    selected = []
    for j, cid in enumerate(cls_ids):
        if cid in wanted_ids:
            m = masks[j] > 0.5
            m = _ensure_mask_size(m, target_hw)
            selected.append(m)
    if not selected:
        return None
    return np.any(np.stack(selected, axis=0), axis=0)


def mask_edge(mask_bool: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Compute mask outline (edges) as boolean array."""
    if thickness < 1:
        thickness = 1
    try:
        m = mask_bool.astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        er = cv2.erode(m, k, iterations=thickness)
        edge = (m > 0) & (er == 0)
        return edge
    except Exception:
        m = mask_bool
        edge = np.zeros_like(m, dtype=bool)
        edge[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(
            m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        )
        return edge


def overlay_masks(
    base_rgb: np.ndarray,
    inclusion_mask: np.ndarray | None,
    gem_mask: np.ndarray | None,
    alpha: float = 0.35,
    show_inclusion_fill: bool = True,
    show_inclusion_outline: bool = True,
    show_gem_outline: bool = True,
):
    """
    Clean overlay:
    - inclusion fill (red)
    - inclusion outline (yellow)
    - gem outline (green)
    """
    out = base_rgb.copy().astype(np.float32)

    if inclusion_mask is not None and show_inclusion_fill:
        color = np.array([255, 0, 0], dtype=np.float32)
        out[inclusion_mask] = (1 - alpha) * out[inclusion_mask] + alpha * color

    if inclusion_mask is not None and show_inclusion_outline:
        e = mask_edge(inclusion_mask, thickness=2)
        out[e] = np.array([255, 255, 0], dtype=np.float32)

    if gem_mask is not None and show_gem_outline:
        e = mask_edge(gem_mask, thickness=2)
        out[e] = np.array([0, 255, 0], dtype=np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def axis_aligned_bbox_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1, (x1 - x0 + 1), (y1 - y0 + 1)


def pca_sizes_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) < 10:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0, keepdims=True)
    X = pts - mean
    cov = (X.T @ X) / max(1, len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    v1 = eigvecs[:, order[0]]
    v2 = eigvecs[:, order[1]]
    p1 = X @ v1
    p2 = X @ v2
    return float(p1.max() - p1.min()), float(p2.max() - p2.min())


def mm_per_pixel_from_distance(distance_mm: float, img_w: int, img_h: int, hfov_deg: float):
    hfov = np.deg2rad(hfov_deg)
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * (img_h / img_w))
    physical_w_mm = 2.0 * distance_mm * np.tan(hfov / 2.0)
    physical_h_mm = 2.0 * distance_mm * np.tan(vfov / 2.0)
    return physical_w_mm / max(1, img_w), physical_h_mm / max(1, img_h)


def render_glb_bytes(glb_bytes: bytes, title: str, height: int = 650):
    b64 = base64.b64encode(glb_bytes).decode()

    st.subheader(title)

    viewer = f"""
    <script type="module"
    src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js">
    </script>

    <model-viewer
        src="data:model/gltf-binary;base64,{b64}"
        alt="{title}"
        auto-rotate
        camera-controls
        shadow-intensity="1"
        style="width:100%; height:{height-50}px; background:#111;">
    </model-viewer>
    """

    html(viewer, height=height)


def load_trimesh_from_glb_bytes(glb_bytes: bytes) -> trimesh.Trimesh:
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp.write(glb_bytes)
        tmp_path = tmp.name

    try:
        scene = trimesh.load(tmp_path, file_type="glb")
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No mesh geometry found in GLB.")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        return mesh
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def plot_meshes(rough: trimesh.Trimesh, cut: trimesh.Trimesh):
    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=rough.vertices[:, 0],
            y=rough.vertices[:, 1],
            z=rough.vertices[:, 2],
            i=rough.faces[:, 0],
            j=rough.faces[:, 1],
            k=rough.faces[:, 2],
            opacity=0.20,
            color="blue",
            name="Rough Gem",
        )
    )

    fig.add_trace(
        go.Mesh3d(
            x=cut.vertices[:, 0],
            y=cut.vertices[:, 1],
            z=cut.vertices[:, 2],
            i=cut.faces[:, 0],
            j=cut.faces[:, 1],
            k=cut.faces[:, 2],
            opacity=0.88,
            color="red",
            name="Optimized Cut",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        ),
        showlegend=True,
    )
    return fig


with right:
    st.markdown("### Upload & Settings")
    st.caption(f"Using weights at: `{WEIGHTS_PATH}`")

    if not Path(WEIGHTS_PATH).exists():
        st.error(f"Couldn't find `best.pt` at `{WEIGHTS_PATH}`. Place it next to this script and refresh.")
        st.stop()

    model = load_model_cached(WEIGHTS_PATH)
    name_map = get_name_map(model)
    all_class_names = [name_map[i] for i in sorted(name_map.keys())]

    kws = ["inclusion", "fracture", "crack", "feather", "cavity", "cloud", "needle"]
    suggested_inclusions = [n for n in all_class_names if any(k in n.lower() for k in kws)]

    uploaded = st.file_uploader(
        "Upload 1 gem image (JPG/PNG).",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.01)

    inclusion_labels = suggested_inclusions
    gem_pick = "Auto (largest non-inclusion)"

    st.markdown("### Camera distance")
    distance_cm = st.number_input("Distance camera → gem (cm)", min_value=0.1, value=30.0, step=0.5)

    with st.expander("Advanced: size calibration (optional)"):
        hfov_deg = st.number_input(
            "Camera Horizontal FOV (deg)",
            min_value=20.0,
            max_value=120.0,
            value=60.0,
            step=1.0,
        )

    st.markdown("### Overlay display")
    show_inclusion_fill = st.checkbox("Show inclusion fill", value=True)
    show_inclusion_outline = st.checkbox("Show inclusion outline", value=True)
    show_gem_outline = st.checkbox("Show gem outline", value=True)
    alpha = st.slider("Inclusion fill opacity", 0.05, 0.85, 0.35, 0.01)

    run = st.button("▶️ Run", type="primary", use_container_width=True)

    st.markdown("### Webcam Capture")
    capture_webcam = st.button("📸 Capture 36 Images from Webcam", use_container_width=True)

    st.divider()
    if st.button("📦 3D Model Generation", use_container_width=True):
        st.session_state.show_3d_ui = True

with left:
    st.markdown("### Segmentation Result (clean overlay)")

    if not run:
        st.caption("Upload an image → set inclusions → distance → click **Run**.")
    else:
        if uploaded is None:
            st.warning("Please upload an image.")
            st.stop()

        pil_img = Image.open(uploaded).convert("RGB")
        base = np.array(pil_img)
        H, W = base.shape[:2]

        with st.spinner("Running YOLO segmentation..."):
            r = predict(model, pil_img, conf=conf)

        class_area_sum = defaultdict(int)
        if r.masks is not None and r.boxes is not None and len(r.boxes.cls) > 0:
            cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
            masks = r.masks.data.detach().cpu().numpy()
            for j, cid in enumerate(cls_ids):
                m = _ensure_mask_size((masks[j] > 0.5), (H, W))
                class_area_sum[cid] += int(m.sum())

        inclusion_id_set = {i for i, n in name_map.items() if n in inclusion_labels}

        if gem_pick != "Auto (largest non-inclusion)":
            gem_class_id = next((i for i, n in name_map.items() if n == gem_pick), None)
        else:
            candidates = [(area, cid) for cid, area in class_area_sum.items() if cid not in inclusion_id_set]
            if not candidates and class_area_sum:
                candidates = [(area, cid) for cid, area in class_area_sum.items()]
            gem_class_id = max(candidates, key=lambda x: x[0])[1] if candidates else None

        gem_name = name_map.get(gem_class_id, "Unknown")

        gem_mask = combine_masks_for_ids(r, {gem_class_id} if gem_class_id is not None else set(), (H, W))
        inclusion_mask = combine_masks_for_ids(r, inclusion_id_set, (H, W))

        if gem_mask is None:
            st.error("No gem mask found. Lower confidence or choose the correct gem class.")
            st.stop()

        overlay = overlay_masks(
            base_rgb=base,
            inclusion_mask=inclusion_mask,
            gem_mask=gem_mask,
            alpha=alpha,
            show_inclusion_fill=show_inclusion_fill,
            show_inclusion_outline=show_inclusion_outline,
            show_gem_outline=show_gem_outline,
        )

        st.markdown('<div class="thumb">', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        gem_area_px = int(gem_mask.sum())
        inc_area_px = int(inclusion_mask.sum()) if inclusion_mask is not None else 0
        inc_frac = (inc_area_px / gem_area_px) if gem_area_px > 0 else 0.0

        bbox = axis_aligned_bbox_from_mask(gem_mask)
        x0, y0, x1, y1, w_px, h_px = bbox if bbox else (0, 0, 0, 0, 0, 0)

        pca = pca_sizes_from_mask(gem_mask)
        major_px, minor_px = pca if pca else (float(w_px), float(h_px))

        mm_w = mm_h = major_mm = minor_mm = None
        try:
            distance_mm = float(distance_cm) * 10.0
            mm_per_px_x, mm_per_px_y = mm_per_pixel_from_distance(distance_mm, W, H, float(hfov_deg))
            mm_per_px_avg = 0.5 * (mm_per_px_x + mm_per_px_y)
            mm_w = w_px * mm_per_px_x
            mm_h = h_px * mm_per_px_y
            major_mm = major_px * mm_per_px_avg
            minor_mm = minor_px * mm_per_px_avg
        except Exception:
            pass

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Gem class", gem_name)
        with m2:
            st.metric("Inclusion %", f"{inc_frac * 100:.2f}%")
        with m3:
            st.metric("Width (bbox)", f"{w_px}px")
        with m4:
            st.metric("Height (bbox)", f"{h_px}px")

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            st.metric("Major (PCA)", f"{major_px:.1f}px")
        with m6:
            st.metric("Minor (PCA)", f"{minor_px:.1f}px")
        with m7:
            st.metric("Inclusion area", f"{inc_area_px:,} px")
        with m8:
            st.metric("Gem area", f"{gem_area_px:,} px")

        if mm_w is not None:
            st.caption("mm values are approximate (distance + HFOV). Pixels are always correct.")
            a, b, c, d = st.columns(4)
            with a:
                st.metric("Width (bbox)", f"{mm_w:.2f} mm")
            with b:
                st.metric("Height (bbox)", f"{mm_h:.2f} mm")
            with c:
                st.metric("Major (PCA)", f"{major_mm:.2f} mm")
            with d:
                st.metric("Minor (PCA)", f"{minor_mm:.2f} mm")

    if capture_webcam:
        st.markdown("## 📸 Webcam Capture Running")

        if arduino:
            arduino.write(b'1')
            st.toast("Motor Started", icon="⚙️")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not available")
            st.stop()

        frame_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_text = st.empty()

        captured_frames = []
        total_images = 36
        interval = 2.05
        progress_bar = progress_placeholder.progress(0)

        try:
            for i in range(total_images):
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read webcam frame")
                        cap.release()
                        st.stop()

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    if time.time() - start_time >= interval:
                        break

                captured_frames.append(frame_rgb)

                flash = np.ones_like(frame_rgb) * 255
                frame_placeholder.image(flash, channels="RGB", use_container_width=True)
                time.sleep(0.1)

                progress_bar.progress((i + 1) / total_images)
                status_text.write(f"Captured {i + 1}/{total_images}")

            cap.release()
        finally:
            # --- STOP MOTOR ---
            if arduino:
                arduino.write(b'0')
                st.toast("Motor Stopped", icon="🛑")
            cap.release()

        st.success("Capture Complete!")
        st.session_state["captured_frames"] = captured_frames

        st.markdown("### 🔎 YOLO Predictions")
        cols = st.columns(3)

        for idx, frame in enumerate(captured_frames):
            pil_img = Image.fromarray(frame)
            r = predict(model, pil_img, conf)
            H, W = frame.shape[:2]

            inclusion_id_set = {i for i, n in name_map.items() if n in inclusion_labels}
            gem_mask = combine_masks_for_ids(r, set(name_map.keys()), (H, W))
            inclusion_mask = combine_masks_for_ids(r, inclusion_id_set, (H, W))

            overlay = overlay_masks(
                frame,
                inclusion_mask,
                gem_mask,
                alpha,
                show_inclusion_fill,
                show_inclusion_outline,
                show_gem_outline,
            )

            with cols[idx % 3]:
                st.image(overlay, caption=f"Image {idx + 1}", use_container_width=True)


# -----------------------------
# 3D MODEL GENERATION
# -----------------------------
if st.session_state.show_3d_ui:
    st.divider()
    st.subheader("📦 3D Model Generation")

    uploaded_file = st.file_uploader(
        "Upload Gem Image for 3D Reconstruction",
        type=["png", "jpg", "jpeg"],
        key="3d_upload",
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate 3D Model", use_container_width=True):
            st.info("Sending image to backend...")

            image_bytes = uploaded_file.getvalue()
            files = {
                "image": (
                    uploaded_file.name,
                    io.BytesIO(image_bytes),
                    uploaded_file.type,
                )
            }

            try:
                res = requests.post(
                    f"{BACKEND_URL}/generate_3d",
                    files=files,
                    proxies={"http": None, "https": None},
                    timeout=120,
                )
            except Exception:
                st.error("Cannot connect to backend")
                st.stop()

            if res.status_code != 200:
                st.error("Backend error")
                st.write(res.text)
                st.stop()

            data = res.json()
            task_id = data["task_id"]
            st.success(f"Task Created: {task_id}")

            progress = st.progress(0)
            progress_value = 0
            glb_data = None

            st.info("Generating 3D model...")

            for _ in range(120):
                r = requests.get(
                    f"{BACKEND_URL}/get_glb/{task_id}",
                    proxies={"http": None, "https": None},
                )

                content_type = r.headers.get("Content-Type", "")

                if "application/json" in content_type:
                    status = r.json().get("status", "UNKNOWN")
                    st.write("Status:", status)
                    progress_value = min(progress_value + 5, 95)
                    progress.progress(progress_value)
                    time.sleep(5)
                elif "model/gltf-binary" in content_type:
                    glb_data = r.content
                    progress.progress(100)
                    break

            if glb_data is None:
                st.error("Model generation timeout")
                st.stop()

            st.success("✅ 3D Model Ready!")

            render_glb_bytes(glb_data, "Generated 3D Gem")

            target_shape_path = SCRIPT_DIR / "shapes" / "oval.glb"
            if not target_shape_path.exists():
                st.error(f"Missing target shape file: {target_shape_path}")
                st.stop()

            with st.spinner("Computing optimized cut..."):
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                    tmp.write(glb_data)
                    rough_glb_path = tmp.name

                try:
                    result = optimize_cut_shape(
                        rough_gem=rough_glb_path,
                        target_shape=target_shape_path,
                        rough_sample_n=8000,
                        target_sample_n=1500,
                        n_slices=60,
                        spin_step=15,
                        axis_positions=11,
                        transverse_steps=3,
                    )
                    rough_mesh = load_trimesh_from_glb_bytes(glb_data)
                except Exception as e:
                    st.error(f"Cut optimization failed: {e}")
                    st.stop()
                finally:
                    if os.path.exists(rough_glb_path):
                        os.remove(rough_glb_path)

            if result.fitted_mesh is None:
                st.error("No valid cut placement found.")
            else:
                st.success(f"Optimized cut found. Score: {result.score:.5f}")
                st.subheader("Optimized Cut Shape")
                fig = plot_meshes(rough_mesh, result.fitted_mesh)
                st.plotly_chart(fig, use_container_width=True)
