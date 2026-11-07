"""Draw the annotated (right-hand) video: skeleton, wrist trail, phase text.

The left-hand side of the delivered split-screen is the untouched original
frame; this module only ever draws on the annotated copy.
"""

from __future__ import annotations

import cv2

import config as cfg
from src import pose as pose_mod
from src.analysis import EVENTS, phase_at_frame

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


def _draw_trail(frame, hand_trail, color, thickness=3):
    pts = [(int(round(x)), int(round(y))) for p in hand_trail
           if p is not None for x, y in [p]]
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts))
        cv2.line(frame, pts[i - 1], pts[i], color,
                 max(1, int(round(thickness * alpha))), cv2.LINE_AA)


# Clubhead trail: magenta, distinct from the amber skeleton and the orange
# wrist trail. Solid = both endpoints measured (eye-verified sightings);
# dotted = bridged / interpolated, including the whole fast-downswing blur
# gap. The trail persists from the takeaway so the full arc is visible.
CLUBHEAD_COLOR = (255, 0, 255)  # BGR, magenta


def _draw_dotted_segment(frame, p0, p1, color, thickness=3, dash=7, gap=6):
    import math
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d = 0.0
    while d < length:
        a = (int(round(x0 + ux * d)), int(round(y0 + uy * d)))
        b = (int(round(x0 + ux * min(length, d + dash))),
             int(round(y0 + uy * min(length, d + dash))))
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
        d += dash + gap


def _draw_clubhead_trail(frame, trail, frame_idx):
    """Draw the clubhead trail up to the current frame, on the annotated
    copy only. `trail` is the dict from src.clubhead.build_trail()."""
    if trail is None:
        return
    pos = trail["positions"]
    measured = trail["measured"]
    first = trail["first_frame"]
    if frame_idx < first:
        return
    solid = trail.get("solid", {})
    for f in range(first + 1, min(frame_idx, trail["last_frame"]) + 1):
        if f not in pos or (f - 1) not in pos:
            continue
        p0, p1 = pos[f - 1], pos[f]
        if solid.get(f) and solid.get(f - 1):
            cv2.line(frame, p0, p1, CLUBHEAD_COLOR, 3, cv2.LINE_AA)
        else:
            _draw_dotted_segment(frame, p0, p1, CLUBHEAD_COLOR, 3)
    # Current head marker: white-ringed magenta dot + label. Suppressed
    # inside the fast-downswing blur gap: a marker there would claim a
    # position the video does not show. The dotted bridge stays.
    # v2: simulated heads (and low/inferred anchors) get a HOLLOW marker —
    # white ring, no magenta fill — visually distinct from a measured head.
    # The gap suppression only applies to v1 (blur_gap set); v2 sets
    # blur_gap to [0, -1] so its simulated heads draw hollow instead.
    gap = trail.get("blur_gap") or [0, -1]
    if frame_idx in pos and not (gap[0] <= frame_idx <= gap[1]):
        p = pos[frame_idx]
        status = (trail.get("status") or {}).get(frame_idx, "")
        is_sim = (trail.get("simulated") or {}).get(frame_idx, False) \
            or status in ("simulated", "low", "inferred")
        label = "CLUBHEAD (sim)" if status == "simulated" else "CLUBHEAD"
        cv2.circle(frame, p, 9, (255, 255, 255), 2, cv2.LINE_AA)
        if not is_sim:
            cv2.circle(frame, p, 5, CLUBHEAD_COLOR, -1, cv2.LINE_AA)
        # Label flips to the left of the marker near the right edge.
        w = frame.shape[1]
        lx = p[0] + 12 if p[0] < w - 140 else p[0] - 128
        cv2.putText(frame, label, (lx, p[1] - 10),
                    _FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, p[1] - 10),
                    _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _text(frame, s, org, scale=None, color=(255, 255, 255), thickness=None):
    cv2.putText(frame, s, org, _FONT, scale or cfg.FONT_SCALE, (0, 0, 0),
                (thickness or cfg.FONT_THICKNESS) + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, _FONT, scale or cfg.FONT_SCALE, color,
                thickness or cfg.FONT_THICKNESS, cv2.LINE_AA)


def render_frame(frame, lm, width, height, frame_idx, analysis, hand_trail,
                 clubhead_trail=None):
    """Return a new annotated copy of `frame`."""
    out = frame.copy()
    # Top panel.
    cv2.rectangle(out, (0, 0), (width, cfg.PANEL_HEIGHT), (20, 20, 20), -1)
    phase = phase_at_frame(analysis["events"], frame_idx)
    _draw_skeleton(out, lm, width, height)
    _draw_trail(out, hand_trail, cfg.TRAIL_COLOR)
    _draw_clubhead_trail(out, clubhead_trail, frame_idx)
    _text(out, f"Phase: {phase}", (12, 34))
    tempo = analysis.get("tempo_display") or analysis["tempo_ratio"]
    tempo_s = f"Tempo {tempo}:1" if tempo else "Tempo --"
    _text(out, f"{tempo_s}  (back {analysis['backswing_s']}s / down "
               f"{analysis['downswing_s']}s)", (12, 66), scale=0.55)
    # Short on purpose: the full "address/top/impact" labels overflow a
    # 354 px wide frame. Order is address / top / impact, degrees, 2D.
    spine = analysis["spine_angle_2d_deg"]
    spine_s = " / ".join(
        f"{spine[ev]}" if spine[ev] is not None else "--"
        for ev in cfg.SPINE_AT
    )
    _text(out, f"Spine 2D: {spine_s} deg", (12, 90), scale=0.45)
    # Clubhead legend (only when a clubhead trail is being drawn).
    if clubhead_trail is not None:
        if clubhead_trail.get("version") == 2:
            _text(out, "Clubhead magenta: solid=measured, dotted=simulated,"
                       " hollow=simulated head",
                  (12, height - cfg.TIMELINE_HEIGHT - 8), scale=0.42)
        else:
            _text(out, "Clubhead magenta: solid=measured, dotted=bridged",
                  (12, height - cfg.TIMELINE_HEIGHT - 8), scale=0.42)
    # Bottom timeline: ticks for the 8 events.
    y0 = height - cfg.TIMELINE_HEIGHT
    cv2.rectangle(out, (0, y0), (width, height), (20, 20, 20), -1)
    n = analysis["n_frames"]
    for i, ev in enumerate(EVENTS):
        x = int(analysis["events"][ev] / max(1, n - 1) * (width - 4)) + 2
        cv2.line(out, (x, y0 + 2), (x, height - 2), (0, 200, 255), 2)
        if frame_idx >= analysis["events"][ev]:
            cv2.circle(out, (x, y0 + cfg.TIMELINE_HEIGHT // 2), 3,
                       (0, 200, 255), -1)
    # Takeaway tick (green): the verified backswing start, distinct from
    # the address tick — the address pose is stance, not the swing start.
    if analysis.get("takeaway_frame") is not None:
        x = int(analysis["takeaway_frame"] / max(1, n - 1) * (width - 4)) + 2
        cv2.line(out, (x, y0 + 2), (x, height - 2), (0, 255, 0), 2)
        if frame_idx >= analysis["takeaway_frame"]:
            cv2.circle(out, (x, y0 + cfg.TIMELINE_HEIGHT // 2), 3,
                       (0, 255, 0), -1)
    # Progress line.
    px = int(frame_idx / max(1, n - 1) * width)
    cv2.line(out, (px, y0), (px, height), (255, 255, 255), 1)
    return out
