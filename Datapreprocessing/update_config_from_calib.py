#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update config/config.toml from a Pose2Sim calibration file (Calib_scene.toml).

Usage (Windows cmd):
    python Datapreprocessing\\update_config_from_calib.py \
        --calib "d:\\ARCS2\\Project Code\\Data\\20251028\\calibration\\Calib_scene.toml" \
        --camera int_cam01_img \
        --config "d:\\ARCS2\\Project Code\\config\\config.toml"

If --calib is omitted, the script will try to auto-detect Calib_scene.toml in this order:
    1) Using project_dir or project_name from config/app_config.toml → <project_dir>/calibration/Calib_scene.toml
    2) Latest matching file under Data/*/calibration/Calib_scene.toml (by modification time)

What it does:
  - Read the specified camera section from Calib_scene.toml (e.g., [int_cam01_img])
  - Extract fx, fy from the 3x3 intrinsic matrix
  - Convert Rodrigues rotation vector to a 3x3 rotation matrix (T_cw)
    - Detect FPS from corresponding videos/camXX.mp4 and update estimation.other_parameters.hz_image
  - Write values into config.toml under:
      [estimation.camera_parameters]
        fx = <fx>
        fy = <fy>
      [estimation.transformation_matrix]
        T_cw = [[...],[...],[...]]
            [estimation.other_parameters]
                hz_image = <fps>

Camera selection sources (precedence):
  1) CLI: --camera and/or --camera-index
  2) config/config.toml: [estimation.camera_selection] camera_key="int_cam01_img", camera_index=1
      (fallback: [estimation.camera_parameters] camera_key/camera_index if present)
  3) Default camera_key="int_cam01_img"; camera index inferred from key (no camera 0)

Notes:
  - If the target sections/keys are missing in config.toml, they will be created.
  - T_cw here is the 3x3 rotation matrix computed from the 'rotation' vector in the calibration file.
    If your downstream expects another convention, adjust accordingly.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import toml  # Requires 'toml' package
import numpy as np
 

try:
    import cv2 as cv
except Exception as e:
    cv = None


def rodrigues_to_R(rvec):
    v = np.array(rvec, dtype=float).reshape(3)
    if cv is not None:
        R, _ = cv.Rodrigues(v)
        return R
    # Fallback using scipy if available
    try:
        from scipy.spatial.transform import Rotation as Rsc
        return Rsc.from_rotvec(v).as_matrix()
    except Exception:
        raise RuntimeError("No backend available to convert Rodrigues vector to rotation matrix (cv2/scipy missing)")


def load_toml(path: Path):
    return toml.load(str(path))

def save_toml(data, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(toml.dumps(data))


def find_repo_root() -> Path:
    # Repo root assumed to be the parent of this script's directory
    return Path(__file__).resolve().parent.parent


def load_app_config(repo_root: Path):
    app_cfg_path = repo_root / 'config' / 'app_config.toml'
    if app_cfg_path.exists():
        try:
            return load_toml(app_cfg_path)
        except Exception:
            return {}
    return {}


def resolve_project_dir(repo_root: Path, app_cfg):
    # Prefer explicit project_dir
    proj = app_cfg.get('project', {}) if isinstance(app_cfg, dict) else {}
    project_dir = proj.get('project_dir') if isinstance(proj, dict) else None
    project_name = proj.get('project_name') if isinstance(proj, dict) else None

    if isinstance(project_dir, str) and project_dir.strip():
        pd = Path(project_dir)
        return pd if pd.is_absolute() else (repo_root / pd)

    # Fallback to Data/<project_name>
    if isinstance(project_name, str) and project_name.strip():
        cand = repo_root / 'Data' / project_name
        if cand.exists():
            return cand

    # Last resort: latest directory under Data/* that contains calibration/Calib_scene.toml
    data_root = repo_root / 'Data'
    if data_root.exists():
        # Collect candidate calibration files and pick the newest by mtime
        candidates = list(data_root.glob('*/calibration/Calib_scene.toml'))
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            return newest.parent.parent
    return None


def find_calib_file(repo_root: Path, provided: str | None = None) -> Path:
    # If provided explicitly, trust it
    if provided:
        p = Path(provided)
        return p if p.is_absolute() else (Path.cwd() / p)

    # Try via app_config
    app_cfg = load_app_config(repo_root)
    proj_dir = resolve_project_dir(repo_root, app_cfg)
    tried = []
    if proj_dir is not None:
        cand = proj_dir / 'calibration' / 'Calib_scene.toml'
        tried.append(cand)
        if cand.exists():
            return cand
        # Try to find any Calib_scene.toml under project_dir
        sub_candidates = list((proj_dir).glob('**/Calib_scene.toml'))
        if sub_candidates:
            return max(sub_candidates, key=lambda p: p.stat().st_mtime)

    # Fallback: search under Data/*/calibration
    data_root = repo_root / 'Data'
    if data_root.exists():
        candidates = list(data_root.glob('*/calibration/Calib_scene.toml'))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

    # If nothing found, raise a descriptive error
    hint = "; tried: " + ", ".join(str(p) for p in tried) if tried else ""
    raise FileNotFoundError(f"Could not auto-detect Calib_scene.toml{hint}. Provide --calib explicitly.")


def extract_cam_index(camera_key: str) -> int:
    """Try to infer camera index from key like 'int_cam01_img' -> 1. Defaults to 1 if not found.
    Note: There is no camera 0; indices start from 1.
    """
    import re
    m = re.search(r'cam(\d+)', camera_key)
    if m:
        idx = int(m.group(1))
        if idx >= 1:
            return idx
    # Fallback to 1
    print(f"[warn] Could not parse camera index from key '{camera_key}', defaulting to 1")
    return 1


def get_video_fps(project_dir: Path, cam_index: int) -> float | None:
    """Return FPS from videos/camXX.mp4 if possible; None if unavailable."""
    cam_name = f"cam{cam_index:02d}.mp4"
    vid_path = project_dir / 'videos' / cam_name
    if not vid_path.exists():
        print(f"[warn] Video not found: {vid_path}")
        return None
    if cv is None:
        print("[warn] OpenCV (cv2) not available; cannot read FPS")
        return None
    cap = cv.VideoCapture(str(vid_path))
    try:
        if not cap.isOpened():
            print(f"[warn] Failed to open video: {vid_path}")
            return None
        fps = cap.get(cv.CAP_PROP_FPS)
        try:
            fps = float(fps)
        except Exception:
            fps = 0.0
        if fps and fps > 0.1 and fps < 10000:
            return fps
        print(f"[warn] Read invalid FPS ({fps}) from {vid_path}")
        return None
    finally:
        cap.release()


def read_camera_selection(config_path: Path):
    """Read camera_key and camera_index from config if available.
    Returns (camera_key or None, camera_index or None)."""
    try:
        cfg = load_toml(config_path)
    except Exception:
        return None, None
    camera_key = None
    camera_index = None
    est = cfg.get('estimation', {}) if isinstance(cfg, dict) else {}
    if isinstance(est, dict):
        sel = est.get('camera_selection', {}) if isinstance(est, dict) else {}
        if isinstance(sel, dict):
            if 'camera_key' in sel and isinstance(sel['camera_key'], str):
                camera_key = sel['camera_key']
            if 'camera' in sel and isinstance(sel['camera'], str) and not camera_key:
                camera_key = sel['camera']
            if 'camera_index' in sel:
                try:
                    camera_index = int(sel['camera_index'])
                except Exception:
                    camera_index = None
        # fallback to camera_parameters if present
        params = est.get('camera_parameters', {}) if isinstance(est, dict) else {}
        if isinstance(params, dict):
            if camera_key is None and 'camera_key' in params and isinstance(params['camera_key'], str):
                camera_key = params['camera_key']
            if camera_index is None and 'camera_index' in params:
                try:
                    camera_index = int(params['camera_index'])
                except Exception:
                    camera_index = None
    return camera_key, camera_index


def update_config(calib_path: Path, camera_key: str, config_path: Path, camera_index: int | None = None) -> None:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    calib = load_toml(calib_path)
    if camera_key not in calib:
        # Try to find by case-insensitive match
        lower_map = {k.lower(): k for k in calib.keys()}
        if camera_key.lower() in lower_map:
            camera_key = lower_map[camera_key.lower()]
        else:
            raise KeyError(f"Camera section '{camera_key}' not found in {calib_path}")

    cam = calib[camera_key]
    # Extract fx, fy from 3x3 matrix
    try:
        K = np.array(cam['matrix'], dtype=float)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
    except Exception as e:
        raise ValueError(f"Invalid or missing 'matrix' for {camera_key}: {e}")

    # Build 3x3 T_cw from Rodrigues 'rotation'
    try:
        rvec = cam['rotation']
        R = rodrigues_to_R(rvec)
    except Exception as e:
        raise ValueError(f"Invalid or missing 'rotation' for {camera_key}: {e}")

    cfg = load_toml(config_path)
    # Ensure nested structure exists
    if 'estimation' not in cfg or not isinstance(cfg['estimation'], dict):
        cfg['estimation'] = {}
    est = cfg['estimation']
    if 'camera_parameters' not in est or not isinstance(est['camera_parameters'], dict):
        est['camera_parameters'] = {}
    if 'transformation_matrix' not in est or not isinstance(est['transformation_matrix'], dict):
        est['transformation_matrix'] = {}
    if 'other_parameters' not in est or not isinstance(est['other_parameters'], dict):
        est['other_parameters'] = {}

    est['camera_parameters']['fx'] = fx
    est['camera_parameters']['fy'] = fy
    est['transformation_matrix']['T_cw'] = R.tolist()

    # Try to update hz_image from the corresponding video camXX.mp4
    # Derive project_dir from calib_path if possible
    project_dir = None
    try:
        if calib_path.parent.name == 'calibration':
            project_dir = calib_path.parent.parent
    except Exception:
        project_dir = None
    if project_dir is None:
        # Fallback to app_config to resolve
        repo_root = find_repo_root()
        app_cfg = load_app_config(repo_root)
        project_dir = resolve_project_dir(repo_root, app_cfg)

    idx = camera_index if (isinstance(camera_index, int) and camera_index >= 1) else extract_cam_index(camera_key)
    fps = None
    if isinstance(project_dir, Path) and project_dir.exists():
        fps = get_video_fps(project_dir, idx)
    if fps is not None:
        est['other_parameters']['hz_image'] = float(fps)
        print(f"Detected FPS from video cam{idx:02d}.mp4: {fps:.3f}; updated hz_image.")
    else:
        print("[warn] Could not detect video FPS; hz_image left unchanged if present.")

    save_toml(cfg, config_path)
    print(f"Updated {config_path} with fx={fx}, fy={fy}, and T_cw from [{camera_key}].")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update config/config.toml from Calib_scene.toml for a selected camera.")
    p.add_argument('--calib', help='Path to Calib_scene.toml (optional; auto-detect if omitted)')
    p.add_argument('--camera', default=None, help='Camera section name in Calib_scene.toml (e.g., int_cam01_img). If omitted, read from config or default to int_cam01_img.')
    p.add_argument('--config', help='Path to config/config.toml (defaults to <repo>/config/config.toml)')
    p.add_argument('--camera-index', type=int, help='Override camera index (1-based). If omitted, inferred from --camera (e.g., cam01 -> 1).')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    # Resolve config path first (so we can read camera selection from config)
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = (repo_root / 'config' / 'config.toml').resolve()

    # Determine camera selection: CLI overrides config, else defaults
    cfg_camera_key, cfg_camera_index = read_camera_selection(config_path)
    camera_key = args.camera if args.camera else (cfg_camera_key if cfg_camera_key else 'int_cam01_img')
    camera_index = args.camera_index if args.camera_index is not None else cfg_camera_index

    # Resolve calibration file (allow auto-detect when --calib is not provided)
    calib_path = find_calib_file(repo_root, args.calib).resolve()

    update_config(calib_path, camera_key, config_path, camera_index=camera_index)


if __name__ == '__main__':
    main()
