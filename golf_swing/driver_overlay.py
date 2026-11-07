"""Overlay the driver clubhead trail + ball launch marker onto the pipeline
splitscreen render (right half), producing the final side-by-side video.

Input : hidden_files/driver-work/out2/driver_dtl/driver_dtl_splitscreen_h264.mp4
        (1440x1280, 29.97fps, 210 frames; left = original, right = skeleton)
Output: files/golf_driver_dtl.mp4 (same geometry; audio muxed from the
        trimmed input).

Clubhead drawing (magenta):
- solid line + solid dot = measured anchors (high/medium), interpolated
  only between anchors <=5 frames apart
- dotted line + hollow dot = simulated (Bezier arcs, low anchor)
- no line and no marker where the head is out of frame (backswing f87-106,
  downswing f112-116, follow-through f124-132)
- simulated arcs appear retrospectively (downswing arc from f117 onward,
  follow-through arc from f133 onward) so no phantom head marker is drawn
  while the head is out of frame.

Ball (cyan): LAUNCH cross at (590,960) from f117 onward. No flight arc --
0 confirmed flight points.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clubhead_driver import build_trail  # noqa: E402

MAGENTA = (255, 0, 255)      # BGR
CYAN = (255, 255, 0)         # BGR
WHITE = (255, 255, 255)


def draw_trail(frame, trail, cur_frame):
    """Draw clubhead trail history up to cur_frame on the right half."""
    pos = trail["positions"]
    W = frame.shape[1]
    off = W // 2  # right half starts here
    shown = []  # (frame, x, y, solid)
    for f in range(trail["first_frame"], cur_frame + 1):
        p = pos.get(f)
        if p is None:
            continue
        x, y = p
        # simulated arc points only appear once their arc's end is reached
        if trail["confidence"][f] == "simulated":
            if 112 <= f <= 116 and cur_frame < 117:
                continue
            if 124 <= f <= 132 and cur_frame < 133:
                continue
        if not (0 <= x < 720 and 0 <= y < 1280):
            continue
        shown.append((f, x + off, y, trail["solid"][f]))
    # draw segments between consecutive shown frames in the same run
    for i in range(len(shown) - 1):
        f0, x0, y0, s0 = shown[i]
        f1, x1, y1, s1 = shown[i + 1]
        if f1 - f0 > 1:
            continue  # break at gaps
        solid = s0 and s1
        col = MAGENTA
        if solid:
            cv2.line(frame, (x0, y0), (x1, y1), col, 3, cv2.LINE_AA)
        else:
            # dotted: long dashes with clear gaps so simulated reads as
            # dotted even at full-frame scale
            n = max(1, int(np.hypot(x1 - x0, y1 - y0) // 24))
            for k in range(n):
                t0 = k / n
                t1 = (k + 0.5) / n
                ax = int(x0 + (x1 - x0) * t0)
                ay = int(y0 + (y1 - y0) * t0)
                bx = int(x0 + (x1 - x0) * t1)
                by = int(y0 + (y1 - y0) * t1)
                cv2.line(frame, (ax, ay), (bx, by), col, 3, cv2.LINE_AA)
    # current head marker
    if cur_frame in pos and pos[cur_frame] is not None:
        x, y = pos[cur_frame]
        if 0 <= x < 720 and 0 <= y < 1280:
            cx, cy = x + off, y
            if trail["solid"][cur_frame]:
                cv2.circle(frame, (cx, cy), 8, MAGENTA, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 10, WHITE, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), 8, MAGENTA, 2, cv2.LINE_AA)


def draw_ball(frame, trail, cur_frame):
    if cur_frame < trail["ball_launch_frame"]:
        return
    x, y = trail["ball_launch"]
    off = frame.shape[1] // 2
    cx, cy = x + off, y
    s = 12
    cv2.line(frame, (cx - s, cy), (cx + s, cy), CYAN, 3, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - s), (cx, cy + s), CYAN, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 16, CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, "LAUNCH", (cx + 22, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)


def draw_legend(frame, cur_frame):
    if cur_frame < 80:
        return
    off = frame.shape[1] // 2
    txt = "CLUBHEAD magenta: solid=measured  dotted=sim (arc)  |  BALL: launch only"
    cv2.putText(frame, txt, (off + 18, 1206),
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


if __name__ == "__main__":
    main()
