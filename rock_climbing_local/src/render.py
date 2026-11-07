"""Draw the annotated (right-hand) panel: skeleton, holds, hip trail, timer.

The left panel of the delivered split-screen is the untouched original
frame; this module only draws on the annotated copy. Skeleton is drawn
from the RAW landmarks (as in golf v1) — it is approximate when the arms
are raised behind the torso. Green dots mark the grip estimates
(wrist + 45px up), which are what the hold contacts are measured from.
"""

from __future__ import annotations

import cv2
import numpy as np

import config as cfg
from src import pose as pose_mod

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_skeleton(frame, lm, width, height):
    if lm is None:
        return
    pts = {}
    for idx in range(33):
        p = pose_mod.landmark_xy(lm, idx, width, height)
        if p is not None:
            pts[idx] = (int(round(p[0])), int(round(p[1])))
    for a, b in pose_mod.CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], cfg.SKELETON_COLOR,
                     cfg.SKELETON_THICKNESS, cv2.LINE_AA)
    for idx, (x, y) in pts.items():
        cv2.circle(frame, (x, y), cfg.JOINT_RADIUS, cfg.JOINT_COLOR, -1,
                   cv2.LINE_AA)


def _text(frame, s, org, scale=None, color=(255, 255, 255), thickness=None):
    cv2.putText(frame, s, org, _FONT, scale or cfg.FONT_SCALE, (0, 0, 0),
                (thickness or cfg.FONT_THICKNESS) + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, _FONT, scale or cfg.FONT_SCALE, color,
                thickness or cfg.FONT_THICKNESS, cv2.LINE_AA)


def _smooth_run(xs, ys):
    """Sliding quadratic smoothing (window 5, degree 2), as in basketball."""
    n = len(xs)
    if n < 5:
        return list(zip(xs, ys))
    out = list(zip(xs, ys))
    for i in range(2, n - 2):
        t = np.arange(i - 2, i + 3, dtype=float)
        cx = np.polyfit(t, xs[i - 2:i + 3], 2)
        cy = np.polyfit(t, ys[i - 2:i + 3], 2)
        out[i] = (int(round(float(np.polyval(cx, i)))),
                  int(round(float(np.polyval(cy, i)))))
    return out


def render_frame(frame, lm, width, height, frame_idx, fps, contacts,
                 timing, hip, height_pct, grips):
    """Return a new annotated copy of `frame`."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (width, cfg.PANEL_HEIGHT), (20, 20, 20), -1)

    # Holds: dim until used, bright once used (by this frame).
    used = {c["hold"] for c in contacts if c["start_frame"] <= frame_idx}
    for hid, cx, cy, r, _how in cfg.HOLDS:
        col = cfg.HOLD_USED_COLOR if hid in used else cfg.HOLD_UNUSED_COLOR
        thick = 3 if hid in used else 1
        cv2.circle(out, (cx, cy), r, col, thick, cv2.LINE_AA)
        _text(out, str(hid), (cx - 6, cy + 6), scale=0.55, color=col)

    # Hip trail (smoothed), fading.
    lo = max(0, frame_idx - cfg.HIP_TRAIL_FRAMES)
    xs = [int(hip[i, 0]) for i in range(lo, frame_idx + 1)]
    ys = [int(hip[i, 1]) for i in range(lo, frame_idx + 1)]
    sm = _smooth_run(xs, ys)
    for a, b in zip(sm, sm[1:]):
        cv2.line(out, a, b, cfg.HIP_TRAIL_COLOR, 3, cv2.LINE_AA)
    if frame_idx < len(hip):
        cv2.circle(out, (int(hip[frame_idx, 0]), int(hip[frame_idx, 1])),
                   6, cfg.HIP_TRAIL_COLOR, -1, cv2.LINE_AA)

    _draw_skeleton(out, lm, width, height)

    # Grip estimates (green dots) — what contacts are measured from.
    for limb in ("LH", "RH"):
        x, y = grips[limb][frame_idx]
        cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), -1, cv2.LINE_AA)

    # Current holds per hand.
    cur = {}
    for c in contacts:
        if c["start_frame"] <= frame_idx <= c["end_frame"]:
            cur[c["limb"]] = c["hold"]
    seq = " ".join(
        f"H{c['hold']}" for c in sorted(contacts, key=lambda c: c["start_frame"])
        if c["start_frame"] <= frame_idx)
    # Timer.
    t = timing
    if t["start_frame"] is not None and frame_idx >= t["start_frame"]:
        if t["top_frame"] is not None and frame_idx >= t["top_frame"]:
            clock = f"TOP {t['climb_s']:.1f}s"
        else:
            clock = f"{(frame_idx - t['start_frame']) / fps:.1f}s"
    else:
        clock = "0.0s"
    status = ""
    if t["drop_frame"] is not None and frame_idx >= t["drop_frame"]:
        status = "DROP!"
    elif t["top_frame"] is not None and frame_idx >= t["top_frame"]:
        status = "TOP - both hands on H7"
    _text(out, f"TIME {clock}  {status}", (12, 38))
    _text(out, f"HEIGHT {height_pct[frame_idx]:.0f}%  "
               f"LH:H{cur.get('LH', '-')} RH:H{cur.get('RH', '-')}",
          (12, 74), scale=0.6)
    if seq:
        _text(out, f"holds: {seq}", (12, 92), scale=0.45, color=(0, 255, 255))
    return out
