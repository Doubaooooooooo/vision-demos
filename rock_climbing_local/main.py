"""rock_climbing_local v1: single phone, bouldering, fun first.

    python main.py data/input/rock_climbing_peng.mp4

Local, offline, no API key: MediaPipe PoseLandmarker (2D pose only) +
HSV purple-hold segmentation + eye-verified small holds. No VLM Run / SAM.

Outputs per clip, in <out-dir>/<clip-stem>/:
  <clip>_annotated.mp4   single-view annotated video (intermediate)
  <clip>_splitscreen.mp4  LEFT original | RIGHT annotated, same size/fps/
                         duration, frame-aligned. This is the deliverable.
  metrics.json           pose coverage, hold sequence, timing
  run.log                console log

Pose landmarks are cached in data/cache/ keyed on the input file's
size+mtime so re-renders don't redo ~4 minutes of inference.
The final H.264 + audio mux is done with ffmpeg (system binary); the
intermediate files are written with OpenCV's mp4v writer, which has no audio.
"""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _ensure_egl():
    if os.environ.get("ROCK_CLIMB_EGL_OK") == "1":
        return
    if ctypes.util.find_library("EGL"):
        return
    libdir = (
        Path.home()
        / "workspace/goals/golf-ball-trajectory-overlay/hidden_files/lib"
    )
    if (libdir / "libEGL.so.1").is_file():
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(libdir) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        env["ROCK_CLIMB_EGL_OK"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


_ensure_egl()

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from src import analysis as analysis_mod
from src import pose as pose_mod
from src import render as render_mod


def log(msg, log_lines):
    print(msg, flush=True)
    log_lines.append(msg)


def ffmpeg_mux_audio(video_no_audio: Path, source: Path, out: Path):
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_no_audio),
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", str(cfg.OUTPUT_CRF),
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {r.stderr[-500:]}")


def _cache_key(video: Path) -> str:
    st = video.stat()
    return hashlib.md5(f"{video.name}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:12]


def run_one(video: Path, out_root: Path):
    log_lines = []
    stem = video.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Pose (cached).
    cache_dir = cfg.PROJECT_DIR / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{stem}_{_cache_key(video)}_landmarks.npy"
    if cache_file.is_file():
        arr = np.load(str(cache_file), allow_pickle=True)
        landmarks = [None if a is None else a for a in arr]
        # Re-read fps/size from the video.
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        log(f"[{stem}] pose loaded from cache {cache_file.name}", log_lines)
    else:
        landmarks, fps, width, height, _n = pose_mod.run_pose(video)
        arr = np.array([np.full((33, 3), np.nan) if lm is None else lm
                        for lm in landmarks], dtype=object)
        np.save(str(cache_file), arr, allow_pickle=True)
        log(f"[{stem}] pose done, cached to {cache_file.name}", log_lines)
    n_frames = len(landmarks)
    lm_arr = np.array([
        np.full((33, 3), np.nan) if lm is None else lm for lm in landmarks
    ], dtype=float)
    cov = float(np.mean(~np.isnan(lm_arr[:, 0, 0])))
    log(f"[{stem}] frames={n_frames} fps={fps:.2f} size={width}x{height} "
        f"pose coverage: {cov:.2f}", log_lines)

    contacts, timing, hip, height_pct, grips, toes, notes = analysis_mod.analyse(
        lm_arr, fps, width, height)
    for note in notes:
        log(f"[{stem}] correction: {note}", log_lines)
    log(f"[{stem}] timing: start f{timing['start_frame']} "
        f"({timing['start_s']}s) top f{timing['top_frame']} "
        f"({timing['top_s']}s) drop f{timing['drop_frame']} "
        f"({timing['drop_s']}s) climb {timing['climb_s']}s", log_lines)
    log(f"[{stem}] hand contacts: " +
        ", ".join(f"{c['limb']} H{c['hold']} "
                  f"{c['start_s']}-{c['end_s']}s" for c in contacts), log_lines)

    # Render annotated + splitscreen.
    annotated = out_dir / f"{stem}_annotated.mp4"
    splitscreen = out_dir / f"{stem}_splitscreen.mp4"
    cap = cv2.VideoCapture(str(video))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wa = cv2.VideoWriter(str(annotated), fourcc, fps, (width, height))
    ws = cv2.VideoWriter(str(splitscreen), fourcc, fps, (width * 2, height))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = landmarks[idx] if idx < len(landmarks) else None
        ann = render_mod.render_frame(frame, lm, width, height, idx, fps,
                                      contacts, timing, hip, height_pct, grips)
        wa.write(ann)
        ws.write(np.hstack([frame, ann]))
        idx += 1
    cap.release()
    wa.release()
    ws.release()
    log(f"[{stem}] rendered {idx} frames", log_lines)

    # H.264 + audio mux.
    annotated_h264 = out_dir / f"{stem}_annotated_h264.mp4"
    splitscreen_h264 = out_dir / f"{stem}_splitscreen_h264.mp4"
    ffmpeg_mux_audio(annotated, video, annotated_h264)
    ffmpeg_mux_audio(splitscreen, video, splitscreen_h264)

    metrics = {
        "clip": video.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fps": fps,
        "size": [width, height],
        "n_frames": n_frames,
        "pose_coverage": round(cov, 3),
        "frames_with_pose": int(np.sum(~np.isnan(lm_arr[:, 0, 0]))),
        "timing": timing,
        "holds": [
            {"id": hid, "centre": [cx, cy], "radius": r, "found_by": how}
            for hid, cx, cy, r, how in cfg.HOLDS
        ],
        "hand_contacts": contacts,
        "verification_notes": notes + [
            "Start f120 (4.0s): left foot still touches the mat at f113, "
            "both feet clearly off at f130; +-0.3s.",
            "Top f736 (24.53s): later of the two H7 dwell starts (LH f690, "
            "RH f736). Per-frame grip rule fires f687, 1.6s early (RH on H6 "
            "at f680, +45px offset overshoots to H7).",
            "RH H3 13.9-16.0s is transitional (hand shifting on H3 while "
            "the body moves toward H5); kept as H3, not a separate move.",
            "Feet: not reported as hold contacts. Eye-verified long "
            "placements: both feet on the dark-green diamond volume "
            "~5.1-9.6s; RF on H2 ~20.4-26.0s; LF near H1 right edge "
            "~23.1-24.5s (smear, uncertain). Feet dangle after f780.",
            "Hip height during the top hang: MediaPipe hip spikes ~70px "
            "up for ~1s twice (f853-884, f984-1010); max height is "
            "therefore taken over the climb phase only (63.1%).",
        ],
        "hip_height_pct": {
            "start": round(float(height_pct[timing["start_frame"]]), 1)
            if timing["start_frame"] is not None else None,
            "top": round(float(height_pct[timing["top_frame"]]), 1)
            if timing["top_frame"] is not None else None,
            # Max over the climb phase only. During the top hang the hip
            # landmark spikes ~70px up for ~1s twice (f853-884, f984-1010),
            # which would read 76%; the true hanging hip is ~60%.
            "max": round(float(np.max(
                height_pct[timing["start_frame"]:timing["top_frame"] + 1])), 1)
            if timing["start_frame"] is not None and timing["top_frame"] is not None
            else None,
        },
        "caveats": [
            "2D pose only; no 3D.",
            "Grip estimate = wrist + (0,-45px); wrist is not the fingers. "
            "Contacts were checked by eye on the frame.",
            "HSV finds only the big purple volumes + H6; H3/H4/H5/H7 were "
            "defined by eye from the median wall image (their HSV is within "
            "~5 units of the grey wall).",
            "Arms are approximate when raised behind the torso (MediaPipe "
            "collapses them); hip/centre trajectory is the reliable signal.",
            "Feet are not reported as hold contacts (smearing/dangling).",
            "Height is 2D hip height above the mat as % of mat->top-hold.",
            "No fall-force or grip-strength numbers.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    elapsed = time.time() - t0
    log(f"[{stem}] done in {elapsed:.1f}s -> {out_dir}", log_lines)
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    return out_dir, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="clip or folder of clips")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    src = Path(args.input)
    out_root = Path(args.out_dir) if args.out_dir else cfg.OUTPUT_DIR
    videos = []
    if src.is_dir():
        videos = sorted(p for p in src.iterdir()
                        if p.suffix.lower() in cfg.VIDEO_EXTENSIONS)
    else:
        videos = [src]
    for v in videos:
        run_one(v, out_root)


if __name__ == "__main__":
    main()
