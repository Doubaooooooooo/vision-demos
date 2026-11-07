"""golf_swing v1: single phone, DTL, beginners, fun first.

    python main.py data/input/golfdb-test_video.mp4
    python main.py data/input/ --out-dir data/output/

Local, offline, no API key: MediaPipe PoseLandmarker (2D pose only).

Outputs per clip, in <out-dir>/<clip-stem>/:
  <clip>_annotated.mp4   single-view annotated video (intermediate)
  <clip>_splitscreen.mp4  LEFT original | RIGHT annotated, same size/fps/
                         duration, frame-aligned. This is the deliverable.
  metrics.json           phases, tempo, spine angles, pose coverage
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
    """MediaPipe needs libEGL.so.1, absent on a bare headless VM.

    Pre-loading a libEGL does NOT help: the loader looks the file up by
    SONAME (libEGL.so.1), and Chromium's libEGL carries SONAME libEGL.so.
    The fix is a directory containing a file literally named libEGL.so.1
    on LD_LIBRARY_PATH, which is only read at process start — so re-exec
    once with it set. Guarded by an env flag; no-op when libEGL is present.
    """
    if os.environ.get("GOLF_SWING_EGL_OK") == "1":
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
        env["GOLF_SWING_EGL_OK"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


_ensure_egl()

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from src import analysis as analysis_mod
from src import clubhead as clubhead_mod
from src import pose as pose_mod
from src import render as render_mod


def log(msg, log_lines):
    print(msg, flush=True)
    log_lines.append(msg)


def wrist_pixel(lm, width, height):
    if lm is None:
        return None
    lw = pose_mod.landmark_xy(lm, pose_mod.LEFT_WRIST, width, height)
    rw = pose_mod.landmark_xy(lm, pose_mod.RIGHT_WRIST, width, height)
    return pose_mod.midpoint(lw, rw)


def ffmpeg_mux_audio(video_no_audio: Path, source: Path, out: Path):
    """Re-encode to H.264 and mux the source's audio (if any)."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_no_audio),
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", str(cfg.OUTPUT_CRF),
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(out),
    ]
    # NOTE: no -shortest here. The source audio can be shorter than the
    # video; -shortest would silently drop trailing video frames and break
    # the same-duration guarantee of the split-screen deliverable.
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-500:]}")


def process_clip(video_path: Path, out_dir: Path, model_path=None,
                   takeaway_frame=None, takeaway_time_s=None,
                   with_clubhead=False, with_clubhead_v2=False):
    t0 = time.time()
    log_lines: list[str] = []
    stem = video_path.stem
    clip_dir = out_dir / stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{stem}] input: {video_path}", log_lines)

    # Pass 1: pose (streaming).
    landmarks, fps, width, height, n_frames = pose_mod.run_pose(
        video_path, model_path
    )
    log(f"[{stem}] frames={n_frames} fps={fps:.2f} size={width}x{height}", log_lines)
    analysis = analysis_mod.analyse(
        landmarks, width, height, fps,
        takeaway_frame=takeaway_frame, takeaway_time_s=takeaway_time_s,
    )
    log(f"[{stem}] pose coverage: {analysis['pose_coverage']:.2f} "
        f"({analysis['frames_with_pose']}/{n_frames} frames)", log_lines)
    log(f"[{stem}] events: {analysis['event_times_s']}", log_lines)
    log(f"[{stem}] tempo: back {analysis['backswing_s']}s / down "
        f"{analysis['downswing_s']}s = {analysis['tempo_ratio']}:1", log_lines)
    log(f"[{stem}] spine 2D (deg): {analysis['spine_angle_2d_deg']}", log_lines)
    for note in analysis["notes"]:
        log(f"[{stem}] NOTE: {note}", log_lines)

    # Wrist pixel trail, precomputed from landmarks.
    trail_len = max(2, int(round(cfg.TRAIL_SECONDS * fps)))
    wrist_pts = [wrist_pixel(lm, width, height) for lm in landmarks]

    # Clubhead trail (face-on clip only): eye-verified anchors + bridged
    # interpolation; see src/clubhead.py for the honesty notes.
    clubhead_trail = None
    clubhead_trail_v2 = None
    if with_clubhead_v2:
        clubhead_trail = clubhead_mod.build_trail_v2(n_frames, wrist_pts)
        clubhead_trail_v2 = clubhead_trail
        n_meas = sum(1 for v in clubhead_trail["status"].values()
                     if v == "measured")
        n_sim = sum(1 for v in clubhead_trail["status"].values()
                    if v == "simulated")
        log(f"[{stem}] clubhead v2: {n_meas} measured / {n_sim} simulated / "
            f"{len(clubhead_trail['positions']) - n_meas - n_sim} other "
            f"frames f{clubhead_trail['first_frame']}-"
            f"{clubhead_trail['last_frame']}", log_lines)
    elif with_clubhead:
        clubhead_trail = clubhead_mod.build_trail(n_frames)
        log(f"[{stem}] clubhead: {sum(1 for v in clubhead_trail['measured'].values() if v)}"
            f" measured / {sum(1 for v in clubhead_trail['measured'].values() if not v)}"
            f" bridged frames f{clubhead_trail['first_frame']}-"
            f"{clubhead_trail['last_frame']}", log_lines)

    # Pass 2: render, streaming. Two writers: annotated (single) and
    # split-screen (original | annotated). Both at source fps/size.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ann_path = clip_dir / f"{stem}_annotated.mp4"
    split_path = clip_dir / f"{stem}_splitscreen.mp4"
    ann_w = cv2.VideoWriter(str(ann_path), fourcc, fps, (width, height))
    split_w = cv2.VideoWriter(str(split_path), fourcc, fps, (width * 2, height))
    if not ann_w.isOpened() or not split_w.isOpened():
        raise RuntimeError("Could not open VideoWriter")
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = landmarks[idx] if idx < len(landmarks) else None
        trail = wrist_pts[max(0, idx - trail_len + 1): idx + 1]
        ann = render_mod.render_frame(
            frame, lm, width, height, idx, analysis, trail,
            clubhead_trail=clubhead_trail,
        )
        ann_w.write(ann)
        split_w.write(np.hstack([frame, ann]))
        idx += 1
    cap.release()
    ann_w.release()
    split_w.release()
    log(f"[{stem}] rendered {idx} frames", log_lines)

    # H.264 + audio finals (ffmpeg). Split-screen is the deliverable.
    ann_h264 = clip_dir / f"{stem}_annotated_h264.mp4"
    split_h264 = clip_dir / f"{stem}_splitscreen_h264.mp4"
    ffmpeg_mux_audio(ann_path, video_path, ann_h264)
    ffmpeg_mux_audio(split_path, video_path, split_h264)

    metrics = {
        "clip": video_path.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fps": analysis["fps"],
        "size": [width, height],
        "n_frames": n_frames,
        "events": analysis["events"],
        "event_times_s": analysis["event_times_s"],
        "takeaway_frame": analysis.get("takeaway_frame"),
        "takeaway_time_s": analysis.get("takeaway_time_s"),
        "backswing_s": analysis["backswing_s"],
        "downswing_s": analysis["downswing_s"],
        "tempo_ratio": analysis["tempo_ratio"],
        "tempo_display": analysis.get("tempo_display"),
        "spine_angle_2d_deg": analysis["spine_angle_2d_deg"],
        "pose_coverage": analysis["pose_coverage"],
        "frames_with_pose": analysis["frames_with_pose"],
        "notes": analysis["notes"],
        "caveats": [
            "2D pose only; no 3D.",
            "Spine angles are 2D projected angles, DTL only.",
            "Phases are a hand-height heuristic, not a learned detector.",
            "Club is not tracked; wrist trail is a hands proxy.",
            "Tempo seconds are video time (stretched on slow-motion source); "
            "only the ratio is comparable across clips.",
        ],
    }
    (clip_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (clip_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    if clubhead_trail_v2 is not None:
        tr = clubhead_trail_v2
        model_cmp = None
        if all(wrist_pts[f] is not None
               for f in range(tr["first_frame"], tr["last_frame"] + 1)):
            model_cmp = clubhead_mod.model_comparison_v2(wrist_pts)
        ch2 = {
            "clip": video_path.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "version": 2,
            "method": (
                "Stage 1: frame-by-frame re-search of the downswing gap "
                "f182-f192 with the wrist + club-length constraint "
                "(visual grounding cross-checked against the shaft line, "
                "the annulus around the wrist, and angle continuity; "
                "grounding validated to 10-24 px on the known f181/f184 "
                "anchors). Found f182/f183 as new anchors; f184/f185 "
                "updated; f186/f188/f191/f192 not found; f187 kept only "
                "as a LOW streak-end anchor (v1 f187 anchor removed — it "
                "was the hand blur). Stage 2: monotone-cubic (PCHIP) arc "
                "through the anchors for the remaining gap frames; the "
                "rigid-rod and body-centre-circle models were checked and "
                "rejected (see model_comparison). Simulated frames are "
                "drawn dotted with a hollow head marker."
            ),
            "first_frame": tr["first_frame"],
            "last_frame": tr["last_frame"],
            "simulated_frames": tr["simulated_frames"],
            "n_frames": len(tr["positions"]),
            "n_measured": sum(1 for v in tr["status"].values()
                              if v == "measured"),
            "n_low": sum(1 for v in tr["status"].values() if v == "low"),
            "n_simulated": sum(1 for v in tr["status"].values()
                               if v == "simulated"),
            "n_bridged": sum(1 for v in tr["status"].values()
                             if v == "bridged"),
            "n_solid_drawn": sum(1 for v in tr["solid"].values() if v),
            "n_dotted_drawn": sum(1 for v in tr["solid"].values() if not v),
            "per_frame": [
                {
                    "frame": f,
                    "x": tr["positions"][f][0],
                    "y": tr["positions"][f][1],
                    "status": tr["status"][f],
                    "confidence": tr["confidence"][f],
                    "wrist": ([round(float(wrist_pts[f][0])),
                               round(float(wrist_pts[f][1]))]
                              if wrist_pts[f] is not None else None),
                    "wrist_dist_px": tr["wrist_dist"].get(f),
                    "shaft_angle_deg": tr["shaft_angle"].get(f),
                    "drawn_solid": tr["solid"][f],
                }
                for f in sorted(tr["positions"])
            ],
            "anchors": tr["anchors"],
            "per_phase": clubhead_mod.phase_counts(
                tr,
                {**analysis["events"],
                 "takeaway_frame": analysis.get("takeaway_frame")},
            ),
            "model_comparison": model_cmp,
            "caveats": [
                "Clubhead positions are 2D image-plane pixels, not 3D.",
                "Only high/medium anchors are measured sightings. Low "
                "anchors are streak-end reads; f190 is inferred from the "
                "ball launch point; f186/f188/f191/f192 are simulated "
                "(PCHIP arc), not measured — do not use them for "
                "clubhead speed or downswing-plane numbers.",
                "The simulated downswing arc relies on the club-length "
                "foreshortening dip (wrist-head distance ~104-149 px "
                "mid-gap, shaft near camera-perpendicular). The f186 dip "
                "to 104 px is the least-constrained number in the trail.",
                "No ball trajectory is drawn in this video: the face-on "
                "ball had 0 confirmed flight points.",
            ],
        }
        (clip_dir / "clubhead_v2.json").write_text(json.dumps(ch2, indent=2))
    if clubhead_trail is not None and clubhead_trail_v2 is None:
        ch = {
            "clip": video_path.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "method": (
                "Eye-verified anchors (see src/clubhead.py). Automated "
                "detection was tried and failed: plain Hough locks onto "
                "the static red pillar / golf-bag lines; "
                "background-differencing Hough finds shaft fragments "
                "whose far endpoint is mid-shaft, not the head; dark-blob "
                "search is swamped by static dark clutter (trees, pillar "
                "bolts, '13' sign, golf bag). Frames between anchors are "
                "linearly interpolated + sliding-quadratic smoothed "
                "(window 5, as for the basketball ball trail). Drawing: "
                "SOLID for intervals between two high/medium anchors at "
                "most 5 frames apart (well constrained); DOTTED for "
                "everything else. The fast downswing f182-f192 is a "
                "motion-blur gap: no smoothing is applied there, the "
                "connector stays linear and dotted, and no head marker "
                "is drawn inside the gap."
            ),
            "first_frame": clubhead_trail["first_frame"],
            "last_frame": clubhead_trail["last_frame"],
            "blur_gap_frames": clubhead_trail["blur_gap"],
            "n_frames": len(clubhead_trail["positions"]),
            "n_measured": sum(1 for v in clubhead_trail["measured"].values() if v),
            "n_bridged": sum(1 for v in clubhead_trail["measured"].values() if not v),
            "n_solid_drawn": sum(1 for v in clubhead_trail["solid"].values() if v),
            "n_dotted_drawn": sum(1 for v in clubhead_trail["solid"].values() if not v),
            "per_frame": [
                {
                    "frame": f,
                    "x": clubhead_trail["positions"][f][0],
                    "y": clubhead_trail["positions"][f][1],
                    "measured": clubhead_trail["measured"][f],
                    "confidence": clubhead_trail["confidence"][f],
                    "drawn_solid": clubhead_trail["solid"][f],
                }
                for f in sorted(clubhead_trail["positions"])
            ],
            "anchors": clubhead_trail["anchors"],
            "per_phase": clubhead_mod.phase_counts(
                clubhead_trail,
                {**analysis["events"],
                 "takeaway_frame": analysis.get("takeaway_frame")},
            ),
            "caveats": [
                "Clubhead positions are 2D image-plane pixels, not 3D.",
                "Only high/medium anchor frames are measured sightings; "
                "low anchors are blur-streak reads and f190 is inferred "
                "from the ball launch point, not a sighting.",
                "The downswing segment f182-f192 is bridged through a "
                "motion-blur gap: the drawn arc there is a constrained "
                "interpolation, not a measured path. Do not use it for "
                "clubhead speed or downswing plane numbers.",
                "No ball trajectory is drawn in this video: the face-on "
                "ball had 0 confirmed flight points.",
            ],
        }
        (clip_dir / "clubhead.json").write_text(json.dumps(ch, indent=2))
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
    ap = argparse.ArgumentParser(description="golf_swing v1 (2D pose, DTL)")
    ap.add_argument("input", nargs="?", default=str(cfg.INPUT),
                    help="clip or folder of clips")
    ap.add_argument("--out-dir", default=str(cfg.OUTPUT_DIR))
    ap.add_argument("--model", default=None,
                    help="path to pose_landmarker_full.task")
    ap.add_argument("--takeaway-frame", type=int, default=None,
                    help="eye-verified takeaway frame: backswing is measured "
                         "from here, not from the address low point")
    ap.add_argument("--takeaway-time", type=float, default=None,
                    help="eye-verified takeaway time in seconds "
                         "(overrides --takeaway-frame for the tempo math)")
    ap.add_argument("--clubhead", action="store_true",
                    help="draw the clubhead trail (src/clubhead.py anchors; "
                         "face-on clip only)")
    ap.add_argument("--clubhead-v2", action="store_true",
                    help="draw the v2 clubhead trail: PCHIP downswing arc, "
                         "simulated gap frames dotted with hollow markers "
                         "(face-on clip only)")
    args = ap.parse_args()
    inputs = collect_inputs(Path(args.input))
    if not inputs:
        raise SystemExit(f"No clips found at {args.input}")
    for video_path in inputs:
        process_clip(video_path, Path(args.out_dir),
                     Path(args.model) if args.model else None,
                     takeaway_frame=args.takeaway_frame,
                     takeaway_time_s=args.takeaway_time,
                     with_clubhead=args.clubhead,
                     with_clubhead_v2=args.clubhead_v2)


if __name__ == "__main__":
    main()
