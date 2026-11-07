"""Draw the annotated (right-hand) video: skeleton, ball trail, rim, counter.

The left-hand side of the delivered split-screen is the untouched original
frame; this module only ever draws on the annotated copy.

The ball trail is SMOOTHED for drawing only: raw per-frame detections
jitter by a few pixels, which draws as a jagged polyline, so each measured
run is smoothed with a sliding quadratic (window 5, degree 2). The first
and last 2 points of a run stay raw (no look-ahead at the ends). This only
changes how the line is drawn: decisions (release, crossing, make/miss)
and the metrics always use the raw points, and frames within +-3 of the
rim crossing are drawn raw so smoothing can never move the ball to the
other side of the rim. Bridged (interpolated) segments are dotted yellow
and are NOT smoothed. The ball marker itself is always drawn at the RAW
detected position.
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
    """Sliding quadratic smoothing (window 5, degree 2) of one measured run.

    For each interior point, fit a quadratic to the 5 points around it and
    take the fitted centre value; the first/last 2 points stay raw. A
    window-5 quadratic moves a point by a few pixels at most, far less than
    the rim radius. Runs shorter than 5 points are drawn raw: smoothing
    them would invent shape the data does not support.
    """
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


def _draw_trail(frame, points, bridged, upto, crossing_frame=None):
    """points: {frame: (x, y)} for one shot; bridged: set of frames.

    Draws the trail up to frame `upto`. Solid orange between two measured
    points, dotted yellow across a bridged gap.
    """
    fs = sorted(f for f in points if f <= upto)
    if not fs:
        return
    # Build the drawn position per frame: smoothed on measured runs,
    # raw on bridged frames and near the crossing.
    draw = {}

    def flush(run):
        if not run:
            return
        xs = [points[f][0] for f in run]
        ys = [points[f][1] for f in run]
        sm = _smooth_run(xs, ys)
        for f, p in zip(run, sm):
            draw[f] = p

    run = []
    for f in fs:
        if f in bridged or (
            crossing_frame is not None
            and abs(f - crossing_frame) <= cfg.SMOOTH_PROTECT_FRAMES
        ):
            flush(run)
            run = []
            draw[f] = points[f]
        else:
            run.append(f)
    flush(run)
    for a, b in zip(fs, fs[1:]):
        pa, pb = draw[a], draw[b]
        if a in bridged or b in bridged:
            # dotted: short dashes
            n = max(2, int(((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2) ** 0.5 // 6))
            for k in range(n):
                if k % 2:
                    continue
                t0, t1 = k / n, (k + 1) / n
                p0 = (int(pa[0] + (pb[0] - pa[0]) * t0),
                      int(pa[1] + (pb[1] - pa[1]) * t0))
                p1 = (int(pa[0] + (pb[0] - pa[0]) * t1),
                      int(pa[1] + (pb[1] - pa[1]) * t1))
                cv2.line(frame, p0, p1, cfg.SIM_COLOR, 3, cv2.LINE_AA)
        else:
            cv2.line(frame, pa, pb, cfg.BALL_COLOR, 3, cv2.LINE_AA)
    # Ball marker at the RAW detected position of the latest frame.
    x, y = points[fs[-1]]
    cv2.circle(frame, (x, y), cfg.BALL_RADIUS, cfg.BALL_COLOR, -1,
               cv2.LINE_AA)
    cv2.circle(frame, (x, y), cfg.BALL_RADIUS + 3, (255, 255, 255), 2,
               cv2.LINE_AA)


def render_frame(frame, lm, width, height, frame_idx, shots, flights,
                 rim, made, attempted):
    """Return a new annotated copy of `frame`."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (width, cfg.PANEL_HEIGHT), (20, 20, 20), -1)
    _draw_skeleton(out, lm, width, height)
    # Rim marker, when known (measured or interpolated).
    if frame_idx in rim:
        rx, ry, measured = rim[frame_idx]
        cv2.circle(out, (rx, ry), cfg.RIM_RADIUS_PX, cfg.RIM_COLOR, 2,
                   cv2.LINE_AA)
        _text(out, "篮筐" if measured else "篮筐(估计)",
              (rx + 8, ry - cfg.RIM_RADIUS_PX - 6), scale=0.45,
              color=(0, 0, 255))
    # Ball trail for the shot that is live at this frame (or the last one).
    live = None
    for tr, sh in zip(flights, shots):
        if sh["release_frame"] <= frame_idx <= sh.get("trail_end_frame", sh["end_frame"]) + 15:
            live = (tr, sh)
    if live is not None:
        tr, sh = live
        cf = sh["crossing"]["frame"] if sh["crossing"] else None
        _draw_trail(out, tr["points"], set(tr["bridged"]), frame_idx, cf)
    # Counter + current shot status.
    _text(out, f"投篮 {made}/{attempted}", (12, 38))
    status = ""
    for sh in shots:
        if sh["release_frame"] <= frame_idx:
            if sh["crossing"] is not None and frame_idx >= sh["crossing"]["frame"]:
                status = f"第{sh['shot']}球: {'进' if sh['outcome'] == 'make' else '没进'}"
            elif frame_idx <= sh["end_frame"]:
                status = f"第{sh['shot']}球飞行中"
            else:
                status = f"第{sh['shot']}球: {'进' if sh['outcome'] == 'make' else '没进'}"
    if status:
        _text(out, status, (12, 74), scale=0.6)
    _text(out, "虚线 = 补帧(模拟), 非实测", (12, 92), scale=0.4,
          color=(0, 255, 255))
    return out
