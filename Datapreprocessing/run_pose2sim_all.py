#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the full Pose2Sim pipeline locally using the code under Datapreprocessing/ObtainGround.

Inputs:
  --calibration  Path to the calibration folder (e.g., Data/20251027/calibration)
  --videos       Path to the videos folder (e.g., Data/20251027/videos)
  --config       Path to the Config.toml file (e.g., Data/20251027/Config.toml)

Outputs:
  All outputs are written under the project directory inferred from the Config.toml,
  typically the parent folder of that file (e.g., Data/20251027/).

Notes:
- This script aliases the local package `Datapreprocessing.ObtainGround` to the name `Pose2Sim`
  so that intra-package imports like `from Pose2Sim.common import ...` resolve without
  installing the pip package. It also provides a safe fallback for importlib.metadata.version
  so files that do `__version__ = version('pose2sim')` won't crash if pip package is absent.
- The actual pipeline requires heavy dependencies (OpenSim, rtmlib, PyQt5, etc.). Ensure your
  environment satisfies Pose2Sim requirements before running.

Example (Windows cmd.exe):
  python Datapreprocessing\run_pose2sim_all.py \
    --calibration "d:\\ARCS2\\Project Code\\Data\\20251027\\calibration" \
    --videos      "d:\\ARCS2\\Project Code\\Data\\20251027\\videos" \
    --config      "d:\\ARCS2\\Project Code\\Data\\20251027\\Config.toml"
"""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from typing import cast
import types

# 1) Patch importlib.metadata.version before importing any local modules that use it
try:
    import importlib.metadata as _ilm
    _real_version_func = _ilm.version
    def _safe_version(pkg_name: str) -> str:
        try:
            return _real_version_func(pkg_name)
        except Exception:
            return "local"
    _ilm.version = _safe_version  # type: ignore[assignment]
except Exception:
    pass  # If patching fails, proceed; imports may still work if package is installed

# 2) Ensure project root is on sys.path (so `Datapreprocessing` is importable)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # d:/.../Project Code/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3) Ensure 'toml' is importable; if not, provide a shim via tomllib (Python 3.11+)
try:
    import toml  # type: ignore
    _toml_module = toml  # prevent unused-import linter warning
except Exception:
    try:
        import importlib as _importlib  # type: ignore
        _tomllib = _importlib.import_module('tomllib')  # Python 3.11+
        _toml = types.ModuleType('toml')
        def _load(fp: object) -> object:
            return _tomllib.load(fp)  # type: ignore[call-arg]
        def _loads(s: object) -> object:
            if isinstance(s, str):
                s = s.encode('utf-8')
            return _tomllib.loads(s)  # type: ignore[arg-type]
        setattr(_toml, 'load', _load)
        setattr(_toml, 'loads', _loads)
        sys.modules['toml'] = _toml
    except Exception:
        pass

# 4) Alias `Pose2Sim` to local `Datapreprocessing.ObtainGround` package for intra-module imports
try:
    import Datapreprocessing.ObtainGround as _local_pose2sim_pkg
    _ = sys.modules.setdefault('Pose2Sim', _local_pose2sim_pkg)
except Exception as e:
    print("[ERROR] Unable to load local Pose2Sim package from 'Datapreprocessing.ObtainGround'.", file=sys.stderr)
    print(f"Reason: {e}", file=sys.stderr)
    sys.exit(1)

# Import after patching
try:
    from Datapreprocessing.ObtainGround import Pose2Sim as Pose2SimModule
except Exception as e:
    print("[ERROR] Importing local Pose2Sim module failed.", file=sys.stderr)
    print(f"Reason: {e}", file=sys.stderr)
    sys.exit(1)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Pose2Sim.runAll using local code and a given project setup.")
    _ = p.add_argument('--calibration', help='Path to the calibration folder (..../calibration)')
    _ = p.add_argument('--videos', help='Path to the videos folder (..../videos)')
    _ = p.add_argument('--config', help='Path to the Config.toml file (..../Config.toml)')
    _ = p.add_argument('--runner-config', help='Path to runner_config.toml to select the project folder (overrides auto-detection)')
    _ = p.add_argument('--dry-run', action='store_true', help='Print resolved paths and exit without running the pipeline')
    _ = p.add_argument('--interactive', action='store_true', help='Interactively choose which steps to run')
    _ = p.add_argument('--only', help='Comma-separated steps to run (calibration,pose,sync,assoc,triang,filter,augment,kinematics)')
    _ = p.add_argument('--skip', help='Comma-separated steps to skip')
    return p.parse_args()


def _auto_project_dir() -> Path | None:
    data_dir = PROJECT_ROOT / 'Data'
    if not data_dir.exists() or not data_dir.is_dir():
        return None
    dated = [p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8]
    if dated:
        dated.sort(key=lambda p: p.name, reverse=True)
        return dated[0]
    return None


def _load_runner_config(explicit_path: Path | None = None):
    """Load optional runner_config.toml to allow picking a specific project folder.

    Expected TOML structure:
    [project]
    project_name = "ANY_STRING"   # optional; resolves to Data/<ANY_STRING>
    project_dir = "ABS_OR_REL_PATH"  # optional; if relative, resolved against workspace root
    """
    import toml  # type: ignore

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend([
        PROJECT_ROOT / 'config' / 'config.toml',
        # PROJECT_ROOT / 'config' / 'config.toml',
        # PROJECT_ROOT / 'Datapreprocessing' / 'runner_config.toml',
        # PROJECT_ROOT / 'runner_config.toml',
        # PROJECT_ROOT / 'Data' / 'runner_config.toml',
    ])
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return toml.load(str(p))
        except Exception:
            continue
    return None


def validate_inputs(calibration: Path, videos: Path, config_toml: Path) -> Path:
    if not calibration.exists() or not calibration.is_dir():
        raise FileNotFoundError(f"Calibration folder not found: {calibration}")
    if not videos.exists() or not videos.is_dir():
        raise FileNotFoundError(f"Videos folder not found: {videos}")
    if not config_toml.exists() or not config_toml.is_file():
        raise FileNotFoundError(f"Config.toml not found: {config_toml}")

    # Project dir is typically the parent directory of Config.toml
    project_dir = config_toml.parent

    # Basic consistency checks
    expected_calib = project_dir / 'calibration'
    expected_videos = project_dir / 'videos'
    if calibration.resolve() != expected_calib.resolve():
        print(f"[WARN] Calibration path differs from expected location: {expected_calib}")
    if videos.resolve() != expected_videos.resolve():
        print(f"[WARN] Videos path differs from expected location: {expected_videos}")

    return project_dir


def run_pipeline(config_toml: Path, run_kwargs: dict[str, bool]) -> None:
    """Run Pose2Sim.runAll.

    The library accepts a path to the project directory containing Config.toml or
    the path to Config.toml itself. We'll pass the project directory for clarity.
    """
    # The public API exposes `runAll(config=None, **kwargs)`
    # We pass the directory that contains Config.toml to let the internal loader
    # discover the config and set up paths properly.
    try:
        getattr(Pose2SimModule, 'runAll')(config=str(config_toml.parent), **run_kwargs)
    except TypeError:
        # Some versions accept the direct path to Config.toml
        getattr(Pose2SimModule, 'runAll')(config=str(config_toml), **run_kwargs)


def _normalize_steps_list(s: str | None) -> set[str]:
    if not s:
        return set()
    items = [x.strip().lower() for x in s.split(',') if x.strip()]
    return set(items)


def _compute_run_kwargs(only: str | None, skip: str | None, interactive: bool) -> dict[str, bool]:
    # Map user-friendly step names to runAll kwargs
    step_to_kw = {
        'calibration': 'do_calibration',
        'pose': 'do_poseEstimation', 'poseestimation': 'do_poseEstimation',
        'sync': 'do_synchronization', 'synchronization': 'do_synchronization',
        'assoc': 'do_personAssociation', 'personassociation': 'do_personAssociation',
        'triang': 'do_triangulation', 'triangulation': 'do_triangulation',
        'filter': 'do_filtering', 'filtering': 'do_filtering',
        'augment': 'do_markerAugmentation', 'markeraugmentation': 'do_markerAugmentation',
        'kin': 'do_kinematics', 'kinematics': 'do_kinematics',
    }
    # Canonical order for prompting/printing
    canonical = [
        ('calibration','do_calibration'),
        ('pose','do_poseEstimation'),
        ('sync','do_synchronization'),
        ('assoc','do_personAssociation'),
        ('triang','do_triangulation'),
        ('filter','do_filtering'),
        ('augment','do_markerAugmentation'),
        ('kin','do_kinematics'),
    ]

    only_set = _normalize_steps_list(only)
    skip_set = _normalize_steps_list(skip)

    # Start with defaults: all True
    result: dict[str, bool] = {kw: True for _, kw in canonical}
    # Apply --only: set everything False except those listed
    if only_set:
        result = {kw: False for _, kw in canonical}
        for k, kw in canonical:
            # accept synonyms
            synonyms = [k]
            for s, skw in step_to_kw.items():
                if skw == kw:
                    synonyms.append(s)
            if any(name in only_set for name in synonyms):
                result[kw] = True
    # Apply --skip: set listed to False
    if skip_set:
        for k, kw in canonical:
            synonyms = [k]
            for s, skw in step_to_kw.items():
                if skw == kw:
                    synonyms.append(s)
            if any(name in skip_set for name in synonyms):
                result[kw] = False

    # Interactive overrides
    if interactive:
        print("请选择要执行的步骤（Y/n），直接回车表示默认：")
        for label, kw in canonical:
            default = 'Y' if result[kw] else 'n'
            prompt = f"- {label}: [Y/n] (默认 {default}) > "
            try:
                ans = input(prompt).strip().lower()
            except EOFError:
                ans = ''
            if ans in ('y','yes'):
                result[kw] = True
            elif ans in ('n','no'):
                result[kw] = False
            # else keep default
    return result


def main() -> None:
    args = get_args()
    # Resolve paths from args or auto-detect under Data/YYYYMMDD
    project_dir: Path | None = None
    config_toml: Path | None = None
    calibration: Path | None = None
    videos: Path | None = None

    cfg = getattr(args, 'config', None)
    runner_cfg_path = getattr(args, 'runner_config', None)
    cal = getattr(args, 'calibration', None)
    vid = getattr(args, 'videos', None)

    # 1) Highest priority: explicit --config
    if cfg:
        config_toml = Path(cast(str, cfg)).resolve()
        project_dir = config_toml.parent
    else:
        # 2) Try runner_config.toml
        rc_path = None
        if isinstance(runner_cfg_path, str) and runner_cfg_path.strip():
            rc_path = Path(runner_cfg_path).resolve()
        rc = _load_runner_config(rc_path)  # type: ignore
        if rc:
            candidate: Path | None = None
            try:
                prj = rc['project']  # type: ignore[index]
                raw_dir = prj.get('project_dir')  # type: ignore[call-arg]
                raw_name = prj.get('project_name')  # type: ignore[call-arg]
                # project_dir wins if provided
                if isinstance(raw_dir, str) and raw_dir.strip():
                    p = Path(raw_dir.strip())
                    if not p.is_absolute():
                        # Resolve relative to workspace root
                        p = (PROJECT_ROOT / p).resolve()
                    candidate = p
                elif isinstance(raw_name, str) and raw_name.strip():
                    name = raw_name.strip()
                    candidate = (PROJECT_ROOT / 'Data' / name).resolve()
            except Exception:
                candidate = None

            if candidate is not None:
                candidate_cfg = candidate / 'Config.toml'
                if candidate_cfg.exists():
                    project_dir = candidate
                    config_toml = candidate_cfg
                else:
                    print(f"[WARN] runner_config points to '{candidate}', but Config.toml was not found there.")

        # 3) Fallback to auto-detect under Data/YYYYMMDD
        if project_dir is None:
            project_dir = _auto_project_dir()
            if project_dir is not None:
                config_toml = project_dir / 'Config.toml'

    if cal:
        calibration = Path(cast(str, cal)).resolve()
    elif project_dir is not None:
        calibration = project_dir / 'calibration'

    if vid:
        videos = Path(cast(str, vid)).resolve()
    elif project_dir is not None:
        videos = project_dir / 'videos'

    if config_toml is None or calibration is None or videos is None:
        print('[ERROR] Missing required inputs. Provide --config and optionally --calibration/--videos, or ensure Data/<YYYYMMDD> with Config.toml exists.', file=sys.stderr)
        sys.exit(2)

    try:
        project_dir = validate_inputs(calibration, videos, config_toml)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    print("[INFO] Project directory:", project_dir)
    print("[INFO] Calibration:", calibration)
    print("[INFO] Videos:", videos)
    print("[INFO] Config:", config_toml)

    # Ensure output base exists (Pose2Sim typically creates needed subfolders)
    os.makedirs(project_dir, exist_ok=True)

    dry = bool(getattr(args, 'dry_run', False))
    only_arg = getattr(args, 'only', None)
    skip_arg = getattr(args, 'skip', None)
    # Default to interactive when neither --only nor --skip is provided
    interactive_effective = bool(getattr(args, 'interactive', False)) or (only_arg is None and skip_arg is None)
    run_kwargs = _compute_run_kwargs(only_arg, skip_arg, interactive_effective)
    print(f"[INFO] Interactive: {interactive_effective}")
    print('[INFO] Steps to run:', run_kwargs)
    if dry:
        print('[INFO] Dry run requested; exiting without running Pose2Sim.')
        return

    # Run the full pipeline
    run_pipeline(config_toml, run_kwargs)
    print("[INFO] Pose2Sim pipeline completed.")


if __name__ == '__main__':
    main()
