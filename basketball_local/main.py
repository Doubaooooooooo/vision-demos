"""basketball_local v1: single phone, gym session, fun first.

    python main.py data/input/basketball_peng.mov
    python main.py data/input/ --out-dir data/output/

Local, offline, no API key: MediaPipe PoseLandmarker (2D pose) + OpenCV
colour ball tracking. No VLM Run / SAM.

Outputs per clip, in <out-dir>/<clip-stem>/:
  <clip>_annotated.mp4   single-view annotated video (intermediate)
  <clip>_splitscreen.mp4  LEFT original | RIGHT annotated, same size/fps/
                         duration, frame-aligned. This is the deliverable.
  metrics.json           pose coverage, ball coverage, per-shot make/miss
  run.log                console log

The final H.264 + audio mux is done with ffmpeg (system binary); the
intermediate files are written with OpenCV's mp4v writer, which has no audio.
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _ensure_egl():
    if os.environ.get("BASKETBALL_EGL_OK") == "1":
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
        env["BASKETBALL_EGL_OK"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


_ensure_egl()

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from src import analysis as analysis_mod
from src import ball as ball_mod
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
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-500:]}")


def process_clip(video_path: Path, out_dir: Path, model_path=None):
    t0 = time.time()
    log_lines: list[str] = []
    stem = video_path.stem
    clip_dir = out_dir / stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{stem}] input: {video_path}", log_lines)

    # Pass 1a: pose (streaming). Drop false-person detections: MediaPipe
    # hallucinates a "person" on the red floor logo when the player is out
    # of frame (bbox ~80x35 px, lying flat). A real player here is upright:
    # bbox height > 150 px and taller than wide. Filtered frames count as
    # no-pose.
    landmarks, fps, width, height, n_frames = pose_mod.run_pose(
        video_path, model_path
    )
    pose_filtered = 0
    for i, lm in enumerate(landmarks):
        if lm is None:
            continue
        pts = [(lm[j, 0] * width, lm[j, 1] * height) for j in range(33)
               if lm[j, 2] >= cfg.LANDMARK_MIN_PRESENCE]
        if not pts:
            landmarks[i] = None
            pose_filtered += 1
            continue
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        if bh < cfg.POSE_MIN_BBOX_HEIGHT or bw > bh:
            landmarks[i] = None
            pose_filtered += 1
    pose_frames = sum(1 for lm in landmarks if lm is not None)
    log(f"[{stem}] frames={n_frames} fps={fps:.2f} size={width}x{height}",
        log_lines)
    log(f"[{stem}] pose coverage: {pose_frames / max(1, n_frames):.2f} "
        f"({pose_frames}/{n_frames} frames; {pose_filtered} false-person "
        f"frames filtered)", log_lines)

    # Pass 1b: ball candidates, rim, tracks, shots.
    per_frame, _bfps, _w, _h, _n = ball_mod.detect_all(video_path)
    cand_frames = sum(1 for c in per_frame if c)
    log(f"[{stem}] ball candidates on {cand_frames}/{n_frames} frames",
        log_lines)
    rim = ball_mod.rim_track(per_frame)
    tracks = ball_mod.link_tracks(per_frame, rim)
    tracks = ball_mod.merge_tracks(tracks)
    flights = ball_mod.flight_tracks(tracks)
    rim_frames = len([f for f, v in rim.items() if v[2]])
    log(f"[{stem}] flight tracks: {len(flights)}; rim measured on "
        f"{rim_frames}/{n_frames} frames", log_lines)
    shots = analysis_mod.analyse_shots(flights, rim)
    for sh in shots:
        log(f"[{stem}] shot {sh['shot']}: {sh['outcome']} "
            f"release f{sh['release_frame']} apex f{sh['apex_frame']} "
            f"crossing {sh['crossing']}", log_lines)
    made = sum(1 for s in shots if s["outcome"] == "make")
    log(f"[{stem}] shots: {made}/{len(shots)} made", log_lines)

    # Pass 2: render, streaming. Two writers: annotated and split-screen.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ann_path = clip_dir / f"{stem}_annotated.mp4"
    split_path = clip_dir / f"{stem}_splitscreen.mp4"
    ann_w = cv2.VideoWriter(str(ann_path), fourcc, fps, (width, height))
    split_w = cv2.VideoWriter(str(split_path), fourcc, fps, (width * 2, height))
    if not ann_w.isOpened() or not split_w.isOpened():
        raise RuntimeError("Could not open VideoWriter")
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    # Counter state over time: made/attempted as decided so far.
    decided = []
    for sh in shots:
        cf = sh["crossing"]["frame"] if sh["crossing"] else sh["end_frame"]
        decided.append((cf, sh["outcome"]))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = landmarks[idx] if idx < len(landmarks) else None
        m = sum(1 for cf, o in decided if cf <= idx and o == "make")
        a = sum(1 for cf, o in decided if cf <= idx)
        ann = render_mod.render_frame(
            frame, lm, width, height, idx, shots, flights, rim, m, a
        )
        ann_w.write(ann)
        split_w.write(np.hstack([frame, ann]))
        idx += 1
    cap.release()
    ann_w.release()
    split_w.release()
    log(f"[{stem}] rendered {idx} frames", log_lines)

    ann_h264 = clip_dir / f"{stem}_annotated_h264.mp4"
    split_h264 = clip_dir / f"{stem}_splitscreen_h264.mp4"
    ffmpeg_mux_audio(ann_path, video_path, ann_h264)
    ffmpeg_mux_audio(split_path, video_path, split_h264)

    # Ball detection coverage during flights.
    flight_frames = set()
    for tr in flights:
        flight_frames.update(tr["points"].keys())
    measured_flight = sum(
        1 for tr in flights for f in tr["points"] if f not in tr["bridged"]
    )
    metrics = {
        "clip": video_path.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fps": fps,
        "size": [width, height],
        "n_frames": n_frames,
        "pose_coverage": pose_frames / max(1, n_frames),
        "frames_with_pose": pose_frames,
        "pose_false_person_frames_filtered": pose_filtered,
        "ball_candidate_frames": cand_frames,
        "flight_frames": len(flight_frames),
        "flight_measured_frames": measured_flight,
        "flight_bridged_frames": sum(len(tr["bridged"]) for tr in flights),
        "rim_measured_frames": rim_frames,
        "shots": shots,
        "made": made,
        "attempted": len(shots),
        "caveats": [
            "2D pose only; no 3D.",
            "Camera moves (handheld pan); the rim is tracked per frame and "
            "interpolated across gaps, it is NOT fixed in the frame.",
            "Ball detection is colour only; occlusions, the red rim and the "
            "floor logo are the main failure modes.",
            "Bridged (dotted) trail segments are interpolated, not measured.",
            "The drawn trail is smoothed (sliding quadratic, window 5) so "
            "the arc reads as a parabola; decisions and metrics use the "
            "raw detected points, and frames near the rim are drawn raw.",
            "Make/miss is a rim-plane crossing rule, checked by eye per shot; "
            "see the crossing frames in shots[].",
            "No shot-form metrics beyond the skeleton.",
        ],
    }
    (clip_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (clip_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    log(f"[{stem}] done in {time.time() - t0:.1f}s -> {clip_dir}", log_lines)
    return clip_dir, split_h264, ann_h264, metrics


def collect_inputs(path: Path):
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in cfg.VIDEO_EXTENSIONS
    )


def main():
    ap = argparse.ArgumentParser(description="basketball_local v1 (2D pose + ball)")
    ap.add_argument("input", nargs="?", default=str(cfg.INPUT))
    ap.add_argument("--out-dir", default=str(cfg.OUTPUT_DIR))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    inputs = collect_inputs(Path(args.input))
    if not inputs:
        raise SystemExit(f"No clips found at {args.input}")
    for video_path in inputs:
        process_clip(video_path, Path(args.out_dir),
                     Path(args.model) if args.model else None)


if __name__ == "__main__":
    main()
