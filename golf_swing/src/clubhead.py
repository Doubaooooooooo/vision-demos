"""Clubhead tracking for the face-on clip (peng_faceon, bay 13).

Honest status: the clubhead is a small, thin, dark object that moves fast.
Automated detection was tried and failed on this clip:
- Plain Hough on Canny edges locks onto the static red pillar / golf-bag
  lines, not the shaft (the shaft is 2-3 px wide and low contrast).
- Background-differencing Hough finds shaft fragments only; the far
  endpoint of a fragment is mid-shaft, not the head.
- Dark-blob search in a sector around the wrist is swamped by static
  dark clutter (trees, pillar bolts, the "13" sign, the golf bag).

What worked: eye-verified anchors. The clubhead was located by inspecting
frames at full resolution (plus crops / contact sheets) and recording the
head position where it is actually visible. Frames between anchors are
interpolated (PCHIP-style: here linear + sliding-quadratic smoothing, the
same drawing smoothing as basketball_local) and are labelled BRIDGED, never
measured. The fast downswing (f182-f192) is a motion-blur gap: the head is
not visible in most of those frames, so that whole segment is drawn DOTTED
and flagged as unreliable, even though it is constrained at both ends and
at f184 (shaft-top blob), f189 (pre-impact blur streak) and f190 (ball
position at contact, INFERRED from the ball, not a sighting).

Anchor confidence:
  high   - head clearly visible as a distinct blob at the end of the shaft
  medium - head visible but small / at the frame edge / on a busy background
  low    - head read off a motion-blur streak, not a distinct blob
  inferred - f190: the ball is gone at f190, so contact happened just
             before; the head is placed at the ball's launch point
             (319, 1159). This is a deduction, not a sighting.

Coordinates are pixels in the 720x1280 frame.
"""

from __future__ import annotations

import numpy as np

# frame -> ((x, y), confidence)
ANCHORS: dict[int, tuple[tuple[int, int], str]] = {
    164: ((15, 790), "low"),      # shaft exits the left edge; head at/out of frame
    166: ((50, 715), "low"),      # blur, low and left of the hands
    168: ((100, 570), "low"),     # shaft tip up-left of the hands
    170: ((95, 445), "medium"),   # crop: shaft tip up-left
    173: ((155, 300), "medium"),  # crop: top of the vertical shaft; Hough (150, 324) agrees
    176: ((380, 200), "high"),    # crop: head at the "13" sign
    177: ((388, 192), "medium"),  # top, stable
    179: ((390, 190), "high"),    # top
    180: ((395, 185), "high"),    # just past the top
    181: ((380, 200), "high"),    # top, stable
    184: ((205, 215), "medium"),  # small blob at the top of the vertical shaft
    185: ((75, 400), "low"),      # vertical blur streak, head read off the streak
    187: ((110, 650), "low"),     # blur left of the hands
    189: ((200, 1100), "medium"), # pre-impact blur streak sweeping to the ball
    190: ((319, 1159), "inferred"),  # ball launch point; ball gone at f190
    193: ((690, 640), "medium"),  # head at the right edge, arms extended
    194: ((700, 630), "medium"),
    195: ((705, 550), "medium"),
    196: ((710, 460), "medium"),
    199: ((700, 360), "medium"),
    201: ((665, 340), "medium"),
    205: ((690, 330), "medium"),
    210: ((695, 350), "high"),    # finish, head clear at top right
    220: ((700, 340), "high"),    # finish, held
}

FIRST_FRAME = min(ANCHORS)
LAST_FRAME = max(ANCHORS)

# Frames whose drawn segment must stay linear + dotted: the fast-downswing
# blur gap. Smoothing a spline through here would invent a pretty arc and
# present it as measured.
BLUR_GAP = (182, 192)


def _sliding_quadratic_smooth(pts: np.ndarray, window: int = 5) -> np.ndarray:
    """Sliding quadratic (degree 2) smoothing, as used for the basketball
    ball trail. `pts` is (n, 2); returns the same shape. Endpoints are
    left unsmoothed (fewer than `window` neighbours)."""
    n = len(pts)
    out = pts.copy()
    half = window // 2
    for i in range(half, n - half):
        xs = np.arange(i - half, i + half + 1, dtype=float)
        for j in range(2):
            coef = np.polyfit(xs - i, pts[i - half:i + half + 1, j], 2)
            out[i, j] = np.polyval(coef, 0.0)
    return out


def build_trail(n_frames: int) -> dict:
    """Per-frame clubhead positions for frames FIRST_FRAME..LAST_FRAME.

    Returns a dict with:
      positions: {frame: (x, y)} drawn positions (smoothed except in the
                 blur gap, where they are the raw linear interpolation)
      measured:  {frame: bool} True only for anchor frames with confidence
                 high/medium (low/inferred anchors count as bridged for
                 drawing purposes: they are evidence, not sightings)
      confidence:{frame: str} anchor confidence, or "bridged"
      anchors:   the raw anchor table
    Frames outside [FIRST_FRAME, LAST_FRAME] get no position.
    """
    frames = sorted(ANCHORS)
    xs = np.array([ANCHORS[f][0][0] for f in frames], dtype=float)
    ys = np.array([ANCHORS[f][0][1] for f in frames], dtype=float)
    all_frames = np.arange(FIRST_FRAME, LAST_FRAME + 1)
    ix = np.interp(all_frames, frames, xs)
    iy = np.interp(all_frames, frames, ys)
    drawn = np.column_stack([ix, iy])
    # Smooth everywhere except the blur gap, where the line must stay an
    # honest straight connector between the constraining anchors.
    smooth = _sliding_quadratic_smooth(drawn, window=5)
    gap_mask = (all_frames >= BLUR_GAP[0]) & (all_frames <= BLUR_GAP[1])
    drawn[~gap_mask] = smooth[~gap_mask]

    positions, measured, confidence = {}, {}, {}
    anchor_set = set(frames)
    for k, f in enumerate(all_frames):
        f = int(f)
        positions[f] = (round(float(drawn[k, 0])), round(float(drawn[k, 1])))
        if f in anchor_set:
            conf = ANCHORS[f][1]
            confidence[f] = conf
            # Only high/medium anchors draw as measured. Low anchors and
            # the inferred f190 are evidence for the bridge, not sightings.
            measured[f] = conf in ("high", "medium")
        else:
            confidence[f] = "bridged"
            measured[f] = False

    # Solid runs: intervals between two measured (high/medium) anchors at
    # most 5 frames apart are well constrained, so the interpolated frames
    # between them draw SOLID too. Anything else (longer bridges, low /
    # inferred anchors, and the whole blur gap) draws DOTTED. The blur gap
    # is excluded even where its bounding anchors are measured.
    solid = {f: False for f in all_frames}
    measured_frames = [f for f in frames if ANCHORS[f][1] in ("high", "medium")]
    for a, b in zip(measured_frames, measured_frames[1:]):
        if b - a <= 5 and not (a < BLUR_GAP[1] and b > BLUR_GAP[0]):
            for f in range(a, b + 1):
                solid[int(f)] = True
    return {
        "positions": positions,
        "measured": measured,
        "confidence": confidence,
        "solid": {int(f): bool(solid[int(f)]) for f in all_frames},
        "anchors": {f: {"xy": list(ANCHORS[f][0]), "confidence": ANCHORS[f][1]}
                    for f in frames},
        "first_frame": FIRST_FRAME,
        "last_frame": LAST_FRAME,
        "blur_gap": list(BLUR_GAP),
    }


def phase_counts(trail: dict, events: dict) -> dict:
    """Measured vs bridged frame counts per swing phase."""
    order = ["takeaway", "toe-up", "mid-backswing", "top", "mid-downswing",
             "impact", "mid-follow-through", "finish"]
    bounds = {
        "takeaway": (events.get("takeaway_frame", 164), events["toe-up"]),
        "toe-up": (events["toe-up"], events["mid-backswing"]),
        "mid-backswing": (events["mid-backswing"], events["top"]),
        "top": (events["top"], events["mid-downswing"]),
        "mid-downswing": (events["mid-downswing"], events["impact"]),
        "impact": (events["impact"], events["mid-follow-through"]),
        "mid-follow-through": (events["mid-follow-through"], events["finish"]),
        "finish": (events["finish"], trail["last_frame"] + 1),
    }
    out = {}
    for name in order:
        a, b = bounds[name]
        fr = [f for f in range(a, b) if f in trail["positions"]]
        out[name] = {
            "frames": [a, b - 1],
            "n_frames": len(fr),
            "n_measured": sum(1 for f in fr if trail["measured"][f]),
            "n_bridged": sum(1 for f in fr if not trail["measured"][f]),
        }
    return out
