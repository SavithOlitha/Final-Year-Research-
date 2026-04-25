import os
import numpy as np
import streamlit as st
import trimesh
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("💎 Fast Pear Cut Optimizer")

uploaded_file = st.file_uploader("Upload Rough Gem (.glb)", type=["glb"])

with st.sidebar:
    st.header("Speed / Quality")
    rough_sample_n = st.slider("Rough sample points", 3000, 20000, 8000, 1000)
    pear_sample_n = st.slider("Pear sample points", 500, 5000, 1500, 250)
    n_slices = st.slider("Axis slices", 30, 120, 60, 10)
    spin_step = st.slider("Spin step (degrees)", 5, 30, 15, 5)
    axis_positions = st.slider("Axis positions", 5, 21, 11, 2)
    transverse_steps = st.slider("Transverse offsets", 1, 5, 3, 1)


def load_mesh(file_or_path, max_faces=20000):
    if isinstance(file_or_path, str):
        ext = os.path.splitext(file_or_path)[1].lower().replace(".", "")
        scene = trimesh.load(file_or_path, file_type=ext if ext else None)
    else:
        name = getattr(file_or_path, "name", "model.glb")
        ext = os.path.splitext(name)[1].lower().replace(".", "")
        scene = trimesh.load(file_or_path, file_type=ext if ext else "glb")

    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No mesh geometry found.")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()

    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(max_faces)
        except Exception:
            pass

    return mesh


def principal_axes(points):
    pts = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    basis = vecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return basis


def normalize_mesh(mesh):
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    max_extent = np.max(m.bounding_box.extents)
    if max_extent > 1e-9:
        m.apply_scale(1.0 / max_extent)
    return m


def sample_points(mesh, n_points):
    verts = mesh.vertices
    if len(verts) > min(400, len(verts)):
        idx = np.random.choice(len(verts), min(400, len(verts)), replace=False)
        verts = verts[idx]

    surf = np.empty((0, 3))
    try:
        surf, _ = trimesh.sample.sample_surface(mesh, n_points)
    except Exception:
        pass

    return np.vstack([verts, surf])


def find_rough_direction(rough_pts):
    basis = principal_axes(rough_pts)
    axis = basis[:, 0]
    center = rough_pts.mean(axis=0)

    rel = rough_pts - center
    s = rel @ axis
    s_min = np.percentile(s, 2)
    s_max = np.percentile(s, 98)
    axis_len = s_max - s_min
    band = 0.12 * axis_len

    proj = np.outer(s, axis)
    radial = np.linalg.norm(rel - proj, axis=1)

    neg_mask = s < (s_min + band)
    pos_mask = s > (s_max - band)

    neg_width = np.median(radial[neg_mask]) if np.any(neg_mask) else np.inf
    pos_width = np.median(radial[pos_mask]) if np.any(pos_mask) else np.inf

    if neg_width < pos_width:
        long_axis = axis
        narrow_s = s_min
        wide_s = s_max
    else:
        long_axis = -axis
        narrow_s = -s_max
        wide_s = -s_min
        s = -s

    long_axis = long_axis / np.linalg.norm(long_axis)

    temp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(temp, long_axis)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])

    u = temp - np.dot(temp, long_axis) * long_axis
    u /= np.linalg.norm(u)
    v = np.cross(long_axis, u)
    v /= np.linalg.norm(v)

    return {
        "center": center,
        "axis": long_axis,
        "u": u,
        "v": v,
        "s_coords": s,
        "narrow_s": narrow_s,
        "wide_s": wide_s,
        "basis": np.column_stack([long_axis, u, v]),
    }


def find_pear_tip_direction(pear_pts):
    basis = principal_axes(pear_pts)
    axis = basis[:, 0]
    center = pear_pts.mean(axis=0)

    rel = pear_pts - center
    s = rel @ axis
    s_min = np.min(s)
    s_max = np.max(s)
    band = 0.10 * (s_max - s_min)

    proj = np.outer(s, axis)
    radial = np.linalg.norm(rel - proj, axis=1)

    neg_mask = s < (s_min + band)
    pos_mask = s > (s_max - band)

    neg_width = np.median(radial[neg_mask]) if np.any(neg_mask) else np.inf
    pos_width = np.median(radial[pos_mask]) if np.any(pos_mask) else np.inf

    if neg_width < pos_width:
        tip_dir = -axis
    else:
        tip_dir = axis

    return tip_dir / np.linalg.norm(tip_dir)


def rotation_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    cross = np.cross(a, b)
    norm_cross = np.linalg.norm(cross)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)

    if norm_cross < 1e-9:
        if dot > 0:
            return np.eye(3)
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = tmp - np.dot(tmp, a) * a
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()

    axis = cross / norm_cross
    angle = np.arccos(dot)
    return Rotation.from_rotvec(angle * axis).as_matrix()


def build_rough_radius_profile(rough_pts, rough_info, n_slices=60):
    center = rough_info["center"]
    axis = rough_info["axis"]
    s = (rough_pts - center) @ axis

    proj = np.outer(s, axis)
    radial_vec = (rough_pts - center) - proj
    radial = np.linalg.norm(radial_vec, axis=1)

    s_min = np.percentile(s, 1)
    s_max = np.percentile(s, 99)
    bins = np.linspace(s_min, s_max, n_slices + 1)
    mids = 0.5 * (bins[:-1] + bins[1:])

    radius = np.zeros_like(mids)
    for i in range(n_slices):
        mask = (s >= bins[i]) & (s < bins[i + 1])
        if np.any(mask):
            radius[i] = np.percentile(radial[mask], 90)
        else:
            radius[i] = 0.0

    # Smooth small gaps
    valid = radius > 0
    if valid.sum() >= 2:
        radius = np.interp(mids, mids[valid], radius[valid])

    return interp1d(
        mids,
        radius,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    ), s_min, s_max


def transform_points(points, scale, R, t):
    return (points @ R.T) * scale + t


def transform_mesh(mesh, scale, R, t):
    m = mesh.copy()
    m.apply_scale(scale)
    T = np.eye(4)
    T[:3, :3] = R
    m.apply_transform(T)
    m.apply_translation(t)
    return m


def validate_by_profile(points, rough_info, radius_fn, s_min, s_max):
    center = rough_info["center"]
    axis = rough_info["axis"]

    rel = points - center
    s = rel @ axis
    proj = np.outer(s, axis)
    radial_vec = rel - proj
    radial = np.linalg.norm(radial_vec, axis=1)

    allowed = radius_fn(s)

    valid_s = (s >= s_min) & (s <= s_max)
    valid_r = radial <= allowed

    ok = valid_s & valid_r
    return float(ok.mean())


def binary_search_scale(points_rot, t, rough_info, radius_fn, s_min, s_max, max_scale, thresh=0.985, steps=8):
    lo, hi = 0.0, max_scale
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        pts = points_rot * mid + t
        frac = validate_by_profile(pts, rough_info, radius_fn, s_min, s_max)
        if frac >= thresh:
            lo = mid
        else:
            hi = mid
    return lo


def optimize_pear(rough_mesh, pear_mesh, rough_pts, pear_pts):
    rough_info = find_rough_direction(rough_pts)
    radius_fn, s_min, s_max = build_rough_radius_profile(rough_pts, rough_info, n_slices=n_slices)

    pear_mesh_n = normalize_mesh(pear_mesh)
    pear_pts_n = sample_points(pear_mesh_n, pear_sample_n)

    pear_tip_dir = find_pear_tip_direction(pear_pts_n)

    # Align pear tip to narrow direction of rough
    base_R = rotation_from_vectors(pear_tip_dir, rough_info["axis"])

    rough_center = rough_info["center"]
    axis = rough_info["axis"]
    u = rough_info["u"]
    v = rough_info["v"]

    s_positions = np.linspace(
        s_min + 0.2 * (s_max - s_min),
        s_min + 0.8 * (s_max - s_min),
        axis_positions,
    )

    shift_levels = np.linspace(-0.15, 0.15, transverse_steps)
    centers = []
    rough_span = s_max - s_min
    side_scale = 0.08 * rough_span

    for s0 in s_positions:
        base_center = rough_center + axis * s0
        for a in shift_levels:
            for b in shift_levels:
                centers.append(base_center + side_scale * (a * u + b * v))

    rough_box = rough_mesh.bounding_box.extents

    best = {
        "score": -1.0,
        "mesh": None,
        "scale": 0.0,
        "R": None,
        "t": None,
        "rough_info": rough_info,
    }

    spin_angles = list(range(0, 360, spin_step))

    progress = st.progress(0.0)
    total = max(len(spin_angles), 1)

    for i, spin in enumerate(spin_angles, start=1):
        progress.progress(i / total)

        R_spin = Rotation.from_rotvec(np.radians(spin) * rough_info["axis"]).as_matrix()
        R = R_spin @ base_R

        rotated_pts = pear_pts_n @ R.T
        extents = rotated_pts.max(axis=0) - rotated_pts.min(axis=0)
        extents = np.maximum(extents, 1e-8)
        max_scale = 0.98 * float(np.min(rough_box / extents))

        for t in centers:
            scale = binary_search_scale(
                rotated_pts,
                t,
                rough_info,
                radius_fn,
                s_min,
                s_max,
                max_scale=max_scale,
                thresh=0.985,
                steps=8,
            )

            if scale <= 1e-6:
                continue

            cut_mesh = transform_mesh(pear_mesh_n, scale, R, t)
            score = cut_mesh.volume / max(rough_mesh.volume, 1e-9)

            if score > best["score"]:
                best.update({
                    "score": score,
                    "mesh": cut_mesh,
                    "scale": scale,
                    "R": R,
                    "t": t,
                })

    return best


def plot_meshes(rough, cut):
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=rough.vertices[:, 0],
        y=rough.vertices[:, 1],
        z=rough.vertices[:, 2],
        i=rough.faces[:, 0],
        j=rough.faces[:, 1],
        k=rough.faces[:, 2],
        opacity=0.20,
        color="blue",
        name="Rough Gem"
    ))

    if cut is not None:
        fig.add_trace(go.Mesh3d(
            x=cut.vertices[:, 0],
            y=cut.vertices[:, 1],
            z=cut.vertices[:, 2],
            i=cut.faces[:, 0],
            j=cut.faces[:, 1],
            k=cut.faces[:, 2],
            opacity=0.88,
            color="red",
            name="Pear Cut"
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data"
        ),
        showlegend=True,
    )
    return fig


if uploaded_file is not None:
    pear_path = os.path.join("shapes", "oval.glb")
    if not os.path.exists(pear_path):
        st.error("Missing shapes/pear.glb")
        st.stop()

    with st.spinner("Loading meshes..."):
        rough_mesh = load_mesh(uploaded_file, max_faces=18000)
        pear_mesh = load_mesh(pear_path, max_faces=8000)

    st.success("Meshes loaded")

    with st.spinner("Sampling geometry..."):
        rough_pts = sample_points(rough_mesh, rough_sample_n)
        pear_pts = sample_points(pear_mesh, pear_sample_n)

    with st.spinner("Optimizing pear placement..."):
        best = optimize_pear(rough_mesh, pear_mesh, rough_pts, pear_pts)

    if best["mesh"] is None:
        st.error("No valid pear placement found.")
        st.stop()

    st.subheader("Best pear cut")
    st.write(f"Volume retention score: {best['score']:.5f}")
    st.write("Pear point is aligned toward the narrower end of the rough gem.")

    fig = plot_meshes(rough_mesh, best["mesh"])
    st.plotly_chart(fig, use_container_width=True)