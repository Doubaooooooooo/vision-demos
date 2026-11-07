"""Clubhead trail for Peng's DRIVER DTL clip (golf_driver_dtl, trimmed 14-21s).

Frames are TRIMMED-clip frames (720x1280, 29.97fps). Trim start = original
14.0s; impact trimmed f118 = original f538 (ball on tee f117, clubhead
contact f118, ball gone f119 -- verified frame by frame on tee crops).

Method (same honesty rules as the face-on clubhead v2):
- Anchors are eye-verified sightings (full-res frames + crops) or
  differencing blobs forming a smooth decelerating chain. Automated
  detection was NOT used for the head: Hough locks onto static lines,
  dark-blob search is swamped by trees/net/poles.
- In DTL the driver head LEAVES THE FRAME on the backswing (f87-106) and
  again in the follow-through (f124-132), and is out of frame through most
  of the downswing (f112-116, shaft visible entering the top-left corner,
  head beyond it). Where the head is out of frame, NO trail is drawn for
  those frames -- a line there would invent a position.
- The downswing arc (f111 -> f117) and follow-through arc (f123 -> f133)
  are quadratic Beziers whose control points sit OFF-FRAME (top-left), so
  the arcs loop out of the frame the way the shaft directions say they
  must. They appear retrospectively (only once their end anchor is
  reached), are drawn DOTTED with hollow markers, and are labelled
  SIMULATED. The downswing is a CURVE, never a straight chord (Peng's
  correction on the face-on clip). Uncertainty on the simulated arcs is
  ~+/-200px in the middle -- do not use them for clubhead speed or
  downswing-plane numbers.
- Ball: launch point only. 0 confirmed flight points (see ball note
  below), so NO flight arc is drawn.

Ball note: reference-frame differencing (median f170-200) after impact
finds an up-left decelerating chain f119 (504,768) -> f120 (471,691) ->
f121 (454,651) -> f122 (443,626) -> f123 (437,609). That chain is the
CLUBHEAD after impact (verified in crops: grey blur + shaft), NOT the
ball. A bright white spot moves with it (likely the clubface highlight);
it is NOT claimed as the ball. The ball leaves the tee toward the range
(away from camera, DTL foreshortening) and is sub-pixel / motion-blurred
invisible at 30fps -- same as the iron DTL clip, which had 3 faint
points; the driver (faster) has 0.
"""

from __future__ import annotations

import numpy as np

# frame -> ((x, y), confidence)
# high: head clearly visible as a distinct blob at the end of the shaft
# medium: head visible but small / blurred / at the frame edge
# low: head read off a motion-blur streak, not a distinct blob
ANCHORS: dict[int, tuple[tuple[int, int], str]] = {
    80: ((645, 1125), "high"),    # takeaway, big black driver head at the ball
    82: ((640, 1130), "high"),    # head still at the ball, moving left
    84: ((565, 1160), "medium"),  # blur streak along the ground
    86: ((10, 1010), "low"),      # blur at the left edge, exiting frame
    107: ((15, 640), "medium"),   # top, head hanging down the left edge
    110: ((15, 470), "medium"),   # head up the left edge, shaft from hands
    111: ((8, 400), "medium"),    # head at the left edge, lagging
    117: ((545, 985), "low"),     # blur just above the ball, 1 frame pre-impact
    118: ((585, 965), "medium"),  # impact blur at the ball (ball at 590,960)
    119: ((504, 768), "medium"),  # differencing blob, grey blur in crop
    120: ((471, 691), "medium"),  # differencing blob
    121: ((454, 651), "medium"),  # differencing blob
    122: ((443, 626), "medium"),  # differencing blob
    123: ((437, 609), "medium"),  # differencing blob
    133: ((490, 450), "high"),    # finish, head clear at top right (re-verified
    # 2026-11-07 on a fresh f133 crop: black oval centred (490,450); the
    # earlier (505,560) read was ~110px too low -- corrected, not reused)
}

# Quadratic Bezier arcs for the simulated segments.
# (start_frame, end_frame, control_point, appear_frame)
# The arc is drawn only once appear_frame is reached (retrospective).
SIM_ARCS = [
    # downswing: f111 (8,400) -> f117 (545,985), control off-frame top-left
    (111, 117, (-400, -200), 117),
    # follow-through: f123 (437,609) -> f133 (505,560), control off-frame
    (123, 133, (-700, -400), 133),
]

# Frames where the head is out of frame: no trail, no marker.
GAPS = [(87, 106), (112, 116), (124, 132)]

BALL_LAUNCH = (590, 960)  # ball centre on the tee, last seen f117
BALL_LAUNCH_FRAME = 117

FIRST_FRAME = 80
LAST_FRAME = 133


def _bezier(p0, c, p1, t):
    t = float(t)
    return (
        (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1],
    )


def _sliding_quadratic_smooth(pts: np.ndarray, window: int = 5) -> np.ndarray:
    n = len(pts)
    out = pts.copy()
    half = window // 2
    for i in range(half, n - half):
        xs = np.arange(i - half, i + half + 1, dtype=float)
        for j in range(2):
            coef = np.polyfit(xs - i, pts[i - half:i + half + 1, j], 2)
            out[i, j] = np.polyval(coef, 0.0)
    return out


def build_trail() -> dict:
    """Per-frame clubhead positions for f80..f133.

    Returns dict with:
      positions: {frame: (x, y)} -- drawn positions (None where the head is
                 out of frame / no trail)
      measured:  {frame: bool} -- True only for high/medium anchor frames
      solid:     {frame: bool} -- True where the drawn segment is solid
      confidence:{frame: str}
      anchors, sim_arcs, gaps: as defined above
    """
    positions: dict[int, tuple[int, int] | None] = {}
    measured: dict[int, bool] = {}
    solid: dict[int, bool] = {}
    confidence: dict[int, str] = {}

    # Segment A: f80-86, linear interp + smoothing, solid.
    seg_a = sorted(f for f in ANCHORS if 80 <= f <= 86)
    xs = np.array([ANCHORS[f][0][0] for f in seg_a], dtype=float)
    ys = np.array([ANCHORS[f][0][1] for f in seg_a], dtype=float)
    fr = np.arange(80, 87)
    ix = np.interp(fr, seg_a, xs)
    iy = np.interp(fr, seg_a, ys)
    sm = _sliding_quadratic_smooth(np.column_stack([ix, iy]), window=5)
    for k, f in enumerate(fr):
        positions[int(f)] = (round(float(sm[k, 0])), round(float(sm[k, 1])))
        confidence[int(f)] = ANCHORS[int(f)][1] if int(f) in ANCHORS else "bridged"
        measured[int(f)] = int(f) in ANCHORS and ANCHORS[int(f)][1] in ("high", "medium")
        solid[int(f)] = True

    # Gap f87-106: no trail.
    for f in range(87, 107):
        positions[f] = None
        measured[f] = False
        solid[f] = False
        confidence[f] = "out-of-frame"

    # Segment B: f107-111, anchors, solid (linear interp between anchors).
    seg_b = sorted(f for f in ANCHORS if 107 <= f <= 111)
    xs = np.array([ANCHORS[f][0][0] for f in seg_b], dtype=float)
    ys = np.array([ANCHORS[f][0][1] for f in seg_b], dtype=float)
    for f in range(107, 112):
        positions[f] = (round(float(np.interp(f, seg_b, xs))),
                        round(float(np.interp(f, seg_b, ys))))
        measured[f] = f in ANCHORS
        solid[f] = True
        confidence[f] = ANCHORS[f][1] if f in ANCHORS else "bridged"

    # Simulated downswing arc f112-116 (Bezier), plus f117 anchor.
    p0 = ANCHORS[111][0]
    p1 = ANCHORS[117][0]
    c = SIM_ARCS[0][2]
    for f in range(112, 117):
        t = (f - 111) / 6.0
        x, y = _bezier(p0, c, p1, t)
        positions[f] = (round(x), round(y))
        measured[f] = False
        solid[f] = False
        confidence[f] = "simulated"
    positions[117] = ANCHORS[117][0]
    measured[117] = False  # low anchor: evidence, not a sighting
    solid[117] = False
    confidence[117] = "low"

    # Segment C: f118-123, anchors every frame, solid.
    for f in range(118, 124):
        positions[f] = ANCHORS[f][0]
        measured[f] = True
        solid[f] = True
        confidence[f] = ANCHORS[f][1]

    # Simulated follow-through arc f124-132 (Bezier), appears at f133.
    p0 = ANCHORS[123][0]
    p1 = ANCHORS[133][0]
    c = SIM_ARCS[1][2]
    for f in range(124, 133):
        t = (f - 123) / 10.0
        x, y = _bezier(p0, c, p1, t)
        positions[f] = (round(x), round(y))
        measured[f] = False
        solid[f] = False
        confidence[f] = "simulated"
    positions[133] = ANCHORS[133][0]
    measured[133] = True
    solid[133] = True
    confidence[133] = "high"

    return {
        "positions": positions,
        "measured": measured,
        "solid": solid,
        "confidence": confidence,
        "anchors": {f: {"xy": list(ANCHORS[f][0]), "confidence": ANCHORS[f][1]}
                    for f in sorted(ANCHORS)},
        "sim_arcs": [{"from": a, "to": b, "control": list(c), "appear": ap}
                     for (a, b, c, ap) in SIM_ARCS],
        "gaps": [list(g) for g in GAPS],
        "first_frame": FIRST_FRAME,
        "last_frame": LAST_FRAME,
        "ball_launch": list(BALL_LAUNCH),
        "ball_launch_frame": BALL_LAUNCH_FRAME,
        "ball_flight_points": [],  # 0 confirmed flight points
    }


def phase_counts(trail: dict, events: dict) -> dict:
    order = ["takeaway", "toe-up", "mid-backswing", "top", "mid-downswing",
             "impact", "mid-follow-through", "finish"]
    bounds = {
        "takeaway": (events.get("takeaway_frame", 80), events["toe-up"]),
        "toe-up": (events["toe-up"], events["mid-backswing"]),
        "mid-backswing": (events["mid-backswing"], events["top"]),
        "top": (events["top"], events["mid-downswing"]),
        "mid-downswing": (events["mid-downswing"], events["impact"]),
        "impact": (events["impact"], events["mid-follow-through"]),
        "mid-follow-through": (events["mid-follow-through"], events["finish"]),
        "finish": (events["finish"], events["finish"]),
    }
    out = {}
    for ph in order:
        a, b = bounds[ph]
        frames = [f for f in range(a, b + 1)
                  if f in trail["positions"]]
        n_meas = sum(1 for f in frames if trail["measured"].get(f))
        n_sim = sum(1 for f in frames if trail["confidence"].get(f) == "simulated")
        n_oof = sum(1 for f in frames if trail["confidence"].get(f) == "out-of-frame")
        out[ph] = {"frames": [a, b], "n_frames": len(frames),
                   "n_measured": n_meas, "n_simulated": n_sim,
                   "n_out_of_frame": n_oof,
                   "n_bridged": len(frames) - n_meas - n_sim - n_oof}
    return out
