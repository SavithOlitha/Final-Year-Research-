import streamlit as st
import requests
import time
import base64
import io
from streamlit.components.v1 import html

BACKEND_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Gem Image → 3D Model", layout="wide")

st.title("💎 Gem Image → 3D Model Generator")

uploaded_file = st.file_uploader(
    "Upload Gem Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    st.image(uploaded_file, caption="Uploaded Image", width="stretch")

    if st.button("Generate 3D Model", width="stretch"):

        st.info("Sending image to backend...")

        image_bytes = uploaded_file.getvalue()

        files = {
            "image": (
                uploaded_file.name,
                io.BytesIO(image_bytes),
                uploaded_file.type
            )
        }

        res = requests.post(
            f"{BACKEND_URL}/generate_3d",
            files=files,
            proxies={"http": None, "https": None},
            timeout=120
        )

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

        # Poll backend
        for _ in range(120):

            r = requests.get(
                f"{BACKEND_URL}/get_glb/{task_id}",
                proxies={"http": None, "https": None}
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

        # Convert GLB to base64
        b64 = base64.b64encode(glb_data).decode()

        st.subheader("Interactive 3D Model")

        viewer = f"""
        <script type="module"
        src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js">
        </script>

        <model-viewer
            src="data:model/gltf-binary;base64,{b64}"
            alt="3D Gem Model"
            auto-rotate
            camera-controls
            shadow-intensity="1"
            style="width:100%; height:600px; background:#111;">
        </model-viewer>
        """

        html(viewer, height=650)