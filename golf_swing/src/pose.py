"""Local 2D pose with MediaPipe PoseLandmarker (tasks API, offline).

No network, no API key. The model file (pose_landmarker_full.task) is
downloaded once; see README.md. Returns per-frame landmarks in normalised
image coordinates (x, y in [0, 1], origin top-left) plus a presence score,
or None for frames where no person was found.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import cv2
import numpy as np

# MediaPipe's native library links libEGL.so.1 and libGLESv2.so.2, which
# are absent on a bare headless VM. Pre-load real Mesa copies (global
# symbols) so the later `import mediapipe` succeeds without LD_LIBRARY_PATH.
# NOTE: pre-loading only works when the file's SONAME matches the name the
# loader asks for. The Mesa copies in hidden_files/lib/ are named exactly
# libEGL.so.1 / libGLESv2.so.2, so this works; the Chromium copies (SONAME
# libEGL.so / libGLESv2.so) do NOT satisfy it and must not be used.
# main.py also re-execs with LD_LIBRARY_PATH as a second route.
_libdir = (
    Path.home()
    / "workspace/goals/golf-ball-trajectory-overlay/hidden_files/lib"
)
for _soname in ("libEGL.so.1", "libGLESv2.so.2"):
    try:
        _p = _libdir / _soname
        if _p.is_file():
            ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        continue

import mediapipe as mp

import config as cfg

# MediaPipe landmark indices (33-point pose).
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Connections drawn in v1 (a readable subset).
CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_SHOULDER, NOSE),
    (RIGHT_SHOULDER, NOSE),
]


def resolve_model_path() -> Path:
    env = os.environ.get(cfg.MODEL_ENV_VAR)
    if env:
        return Path(env)
    if cfg.MODEL_PATH.is_file():
        return cfg.MODEL_PATH
    for cand in (
        cfg.PROJECT_DIR / "pose_landmarker_full.task",
        Path.cwd() / "pose_landmarker_full.task",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Pose model not found. Download pose_landmarker_full.task from "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task "
        f"and put it at {cfg.MODEL_PATH}, next to the project, or set "
        f"{cfg.MODEL_ENV_VAR} to its path."
    )


def run_pose(video_path: Path, model_path: Path | None = None):
    """Run pose on every frame of a video, streaming (frames are NOT kept).

    Returns (landmarks, fps, width, height, n_frames) where landmarks is a
    list (one per frame) of either a (33, 3) float array [x, y, presence] in
    normalised coordinates or None. Streaming keeps RAM flat even on long
    clips; callers re-open the video for a render pass.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = model_path or resolve_model_path()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=cfg.POSE_MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=cfg.POSE_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=cfg.POSE_MIN_TRACKING_CONFIDENCE,
        num_poses=1,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks: list[np.ndarray | None] = []
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % cfg.POSE_STRIDE == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(round(1000 * idx / fps))
                result = landmarker.detect_for_video(mp_image, ts_ms)
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    arr = np.array(
                        [[p.x, p.y, p.presence] for p in lm], dtype=np.float64
                    )
                    landmarks.append(arr)
                else:
                    landmarks.append(None)
            else:
                landmarks.append(None)  # filled by interpolation in analysis
            idx += 1
    cap.release()
    return landmarks, fps, width, height, len(landmarks)


def landmark_xy(lm: np.ndarray | None, idx: int, w: int, h: int):
    """Pixel (x, y) for landmark idx, or None if missing / low presence."""
    if lm is None:
        return None
    if lm[idx, 2] < cfg.LANDMARK_MIN_PRESENCE:
        return None
    return (float(lm[idx, 0] * w), float(lm[idx, 1] * h))


def midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
