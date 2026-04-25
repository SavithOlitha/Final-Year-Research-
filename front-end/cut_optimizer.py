import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Optional, Union

import numpy as np
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


MeshInput = Union[str, os.PathLike, BinaryIO, trimesh.Trimesh]


@dataclass
class CutOptimizationResult:
    fitted_mesh: Optional[trimesh.Trimesh]
    score: float
    scale: float
    rotation: Optional[np.ndarray]
    translation: Optional[np.ndarray]
    rough_info: Dict[str, Any]


def load_mesh(mesh_input: MeshInput, max_faces: int = 20000) -> trimesh.Trimesh:
    """
    Load a mesh from:
    - file path
    - file-like object
    - trimesh.Trimesh
    """
    if isinstance(mesh_input, trimesh.Trimesh):
        mesh = mesh_input.copy()
    else:
        if isinstance(mesh_input, (str, os.PathLike)):
            path = str(mesh_input)
            ext = os.path.splitext(path)[1].lower().replace(".", "")
            scene = trimesh.load(path, file_type=ext if ext else None)
        else:
            name = getattr(mesh_input, "name", "model.glb")
            ext = os.path.splitext(name)[1].lower().replace(".", "")
            scene = trimesh.load(mesh_input, file_type=ext if ext else "glb")

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


def principal_axes(points: np.ndarray) -> np.ndarray:
    pts = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    basis = vecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return basis


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    max_extent = np.max(m.bounding_box.extents)
    if max_extent > 1e-9:
        m.apply_scale(1.0 / max_extent)
    return m


def sample_points(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
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


def find_rough_direction(rough_pts: np.ndarray) -> Dict[str, Any]:
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


def find_target_tip_direction(target_pts: np.ndarray) -> np.ndarray:
    basis = principal_axes(target_pts)
    axis = basis[:, 0]
    center = target_pts.mean(axis=0)

    rel = target_pts - center
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


def rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def build_rough_radius_profile(
    rough_pts: np.ndarray,
    rough_info: Dict[str, Any],
    n_slices: int = 60,
):
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

    valid = radius > 0
    if valid.sum() >= 2:
        radius = np.interp(mids, mids[valid], radius[valid])

    return (
        interp1d(
            mids,
            radius,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        ),
        s_min,
        s_max,
    )


def transform_mesh(mesh: trimesh.Trimesh, scale: float, R: np.ndarray, t: np.ndarray) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_scale(scale)

    T = np.eye(4)
    T[:3, :3] = R
    m.apply_transform(T)
    m.apply_translation(t)
    return m


def validate_by_profile(
    points: np.ndarray,
    rough_info: Dict[str, Any],
    radius_fn,
    s_min: float,
    s_max: float,
) -> float:
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


def binary_search_scale(
    points_rot: np.ndarray,
    t: np.ndarray,
    rough_info: Dict[str, Any],
    radius_fn,
    s_min: float,
    s_max: float,
    max_scale: float,
    thresh: float = 0.985,
    steps: int = 8,
) -> float:
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


def optimize_cut_shape(
    rough_gem: MeshInput,
    target_shape: MeshInput,
    rough_sample_n: int = 8000,
    target_sample_n: int = 1500,
    n_slices: int = 60,
    spin_step: int = 15,
    axis_positions: int = 11,
    transverse_steps: int = 3,
    rough_max_faces: int = 18000,
    target_max_faces: int = 8000,
    fit_threshold: float = 0.985,
    binary_steps: int = 8,
) -> CutOptimizationResult:
    """
    Fit a target cut shape inside a rough gem.

    Parameters
    ----------
    rough_gem:
        Rough gem mesh/path/file-like object.
    target_shape:
        Target cut shape mesh/path/file-like object.
    rough_sample_n:
        Number of rough gem sample points.
    target_sample_n:
        Number of target shape sample points.
    n_slices:
        Number of slices used to build the rough radius profile.
    spin_step:
        Rotation step around the main axis, in degrees.
    axis_positions:
        Number of candidate positions along the main axis.
    transverse_steps:
        Number of candidate offsets in the transverse directions.
    rough_max_faces:
        Max faces for rough mesh simplification.
    target_max_faces:
        Max faces for target shape simplification.
    fit_threshold:
        Fraction of target sample points that must lie inside the rough profile.
    binary_steps:
        Binary search iterations for scale fitting.

    Returns
    -------
    CutOptimizationResult
    """
    rough_mesh = load_mesh(rough_gem, max_faces=rough_max_faces)
    target_mesh = load_mesh(target_shape, max_faces=target_max_faces)

    rough_pts = sample_points(rough_mesh, rough_sample_n)
    rough_info = find_rough_direction(rough_pts)
    radius_fn, s_min, s_max = build_rough_radius_profile(rough_pts, rough_info, n_slices=n_slices)

    target_mesh_n = normalize_mesh(target_mesh)
    target_pts_n = sample_points(target_mesh_n, target_sample_n)
    target_tip_dir = find_target_tip_direction(target_pts_n)

    base_R = rotation_from_vectors(target_tip_dir, rough_info["axis"])

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

    best_score = -1.0
    best_mesh = None
    best_scale = 0.0
    best_R = None
    best_t = None

    for spin in range(0, 360, spin_step):
        R_spin = Rotation.from_rotvec(np.radians(spin) * axis).as_matrix()
        R = R_spin @ base_R

        rotated_pts = target_pts_n @ R.T
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
                thresh=fit_threshold,
                steps=binary_steps,
            )

            if scale <= 1e-6:
                continue

            cut_mesh = transform_mesh(target_mesh_n, scale, R, t)
            score = cut_mesh.volume / max(rough_mesh.volume, 1e-9)

            if score > best_score:
                best_score = score
                best_mesh = cut_mesh
                best_scale = scale
                best_R = R
                best_t = t

    return CutOptimizationResult(
        fitted_mesh=best_mesh,
        score=best_score if best_mesh is not None else 0.0,
        scale=best_scale,
        rotation=best_R,
        translation=best_t,
        rough_info=rough_info,
    )