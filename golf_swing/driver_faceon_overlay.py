"""Overlay the driver FACE-ON clubhead trail + ball dots onto the pipeline
splitscreen render (right half), producing the final side-by-side video.

Input : /tmp/dfo-out2/driver_faceon/driver_faceon_splitscreen_h264.mp4
        (1440x1280, 30fps, 179 frames; left = original, right = skeleton)
Output: files/golf_driver_faceon.mp4 (same geometry; audio from trimmed input).

Clubhead drawing (magenta):
- solid line + solid dot = measured anchors (high/medium) and short
  bridges (<=5 frames) between them
- NO line and NO marker where the head is out of frame / occluded
  (f79-103 top, f109-115 wrap) -- no simulated top arc in this version
Ball drawing (cyan):
- LAUNCH cross at (395,985) from f106 onward
- flight dots at f107 (447,767) and f108 (433,645), appearing once
  reached and persisting; NO arc (only 2 confirmed flight points)
- "FACE-ON" tag on the right half: the pipeline's spine numbers are
  face-on lateral lean, NOT DTL spine angles.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clubhead_driver_faceon import build_trail  # noqa: E402

MAGENTA = (255, 0, 255)      # BGR
CYAN = (255, 255, 0)         # BGR
WHITE = (255, 255, 255)


def draw_trail(frame, trail, cur_frame):
    pos = trail["positions"]
    off = frame.shape[1] // 2
    shown = []
    for f in range(trail["first_frame"], cur_frame + 1):
        p = pos.get(f)
        if p is None:
            continue
        x, y = p
        if not (0 <= x < 720 and 0 <= y < 1280):
            continue
        shown.append((f, x + off, y, trail["solid"][f]))
    for i in range(len(shown) - 1):
        f0, x0, y0, s0 = shown[i]
        f1, x1, y1, s1 = shown[i + 1]
        if f1 - f0 > 1:
            continue  # break at gaps
        if s0 and s1:
            cv2.line(frame, (x0, y0), (x1, y1), MAGENTA, 3, cv2.LINE_AA)
    if cur_frame in pos and pos[cur_frame] is not None:
        x, y = pos[cur_frame]
        if 0 <= x < 720 and 0 <= y < 1280:
            cx, cy = x + off, y
            cv2.circle(frame, (cx, cy), 8, MAGENTA, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 10, WHITE, 2, cv2.LINE_AA)


def draw_ball(frame, trail, cur_frame):
    off = frame.shape[1] // 2
    if cur_frame < trail["ball_launch_frame"]:
        return
    x, y = trail["ball_launch"]
    cx, cy = x + off, y
    s = 12
    cv2.line(frame, (cx - s, cy), (cx + s, cy), CYAN, 3, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - s), (cx, cy + s), CYAN, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 16, CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, "LAUNCH", (cx + 22, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)
    for f, info in sorted(trail["ball_flight"].items()):
        if cur_frame < f:
            continue
        bx, by = info["xy"]
        cv2.circle(frame, (bx + off, by), 7, CYAN, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx + off, by), 9, WHITE, 2, cv2.LINE_AA)


def draw_legend(frame, cur_frame):
    off = frame.shape[1] // 2
    # top-right of the right half: the pipeline's Phase/Tempo/Spine text
    # occupies the top-left, so FACE-ON must not sit there (first render
    # overlapped "Phase: impact").
    cv2.putText(frame, "FACE-ON", (off + 560, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    if cur_frame < 68:
        return
    txt = "CLUBHEAD magenta: solid=measured  gap=head out of frame"
    cv2.putText(frame, txt, (off + 18, 1206),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    txt2 = "BALL cyan: launch + 2 flight dots, no arc"
    cv2.putText(frame, txt2, (off + 18, 1236),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splitscreen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-src", default=None)
    args = ap.parse_args()

    trail = build_trail()
    cap = cv2.VideoCapture(args.splitscreen)
    assert cap.isOpened(), args.splitscreen
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp = args.out + ".novideo.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw_trail(frame, trail, f)
        draw_ball(frame, trail, f)
        draw_legend(frame, f)
        vw.write(frame)
        f += 1
    vw.release()
    cap.release()
    assert f == n, f"wrote {f} frames, expected {n}"

    if args.audio_src:
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-i", tmp, "-i", args.audio_src,
               "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", args.out]
        subprocess.run(cmd, check=True)
        Path(tmp).unlink()
    else:
        Path(tmp).rename(args.out)
    print(f"wrote {args.out} frames={f} fps={fps} size={w}x{h}")
    print(f"clubhead: measured={trail['n_measured']} bridged={trail['n_bridged']} gap={trail['n_gap']}")


if __name__ == "__main__":
    main()
