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


# ---------------------------------------------------------------------------
# v2: fix the downswing arc (Peng: the backswing arc is right, the downswing
# was drawn as a straight dotted connector — the real path is a curve).
#
# STAGE 1 — frame-by-frame re-search of the gap f182-f192, wrist + club
# length constrained, cross-checked with visual grounding:
#   * f182 (328,195) and f183 (284,200): FOUND. Visual grounding located
#     the head at the end of the visible shaft; both lie on the shaft line
#     from the wrist, L 294/280 px, shaft angle -60/-68 deg, continuing the
#     top sequence (-50 deg at f181) smoothly.
#   * f184: grounding (215,214) agrees with the v1 anchor (205,215) to
#     10 px. Anchor updated to the grounding value.
#   * f185: grounding (91,366), a streak-end read on the vertical blur
#     streak (v1 anchor (75,400) is within 35 px). L has dropped to ~197 px.
#   * f186: NOT FOUND. No distinct head in the full frame or in a 2x
#     contrast-enhanced crop of the left region; blob/diff search returns
#     only clutter. Grounding on f186/f187 points at the hands (L ~92 px),
#     which is a failure mode, not a sighting — rejected.
#   * f187: NOT FOUND as a distinct blob. A faint diagonal streak runs
#     down-left from the hands and ends around (90-140, 700-740); recorded
#     as a LOW anchor (120,730), a streak-end read, not a sighting. The v1
#     anchor (110,650) is REMOVED: it sits on the hand blur, ~20 px from
#     the wrist, and is not the head.
#   * f188, f191, f192: NOT FOUND. No distinct head; f192 shows no club
#     at all. (f189 streak-end and f190 ball-inferred anchors stand.)
#   * Validation of the grounding method: f181 grounding (356,197) vs the
#     high-confidence v1 anchor (380,200) — 24 px agreement; f184 10 px.
#
# CLUB-LENGTH CALIBRATION (wrist midpoint from MediaPipe, presence 0.99-1.0
# through the gap — the wrist itself is reliable):
#   top f176-181: L 311-341 px, shaft -49 deg (nearly in the image plane)
#   f184: L 286 px, -81 deg; f185: L 197 px, -113 deg
#   impact f189/f190: L 401/448 px (shaft back in the image plane)
#   follow-through f193-196: L 157-174 px (shaft pointing at the camera)
#   finish f210/220: L 285-290 px
#   => pixel length is NOT constant: it is foreshortened whenever the shaft
#   points toward/away from the camera, dipping to ~160 px in follow-
#   through and (by the same physics) mid-downswing. A single fixed pixel
#   length must not be assumed.
#
# STAGE 2 — model comparison for the gap (f186-188, f191-192):
#   Model A (rigid rod about the wrist, shaft angle + L interpolated from
#     anchors): without the f187 streak anchor it predicts the head off
#     the left edge at f186-187; with the anchor it lands within ~55 px of
#     the PCHIP arc in the downswing, but diverges ~110-125 px post-impact
#     (the angle unwrap is unreliable through the wrist release). Not used
#     as the generator; the downswing convergence is a cross-check.
#   Model B (circle about the body centre ~(468,677), r~503 px): fits the
#     downswing anchors to +/-25 px, but predicts the head at the left
#     edge (x ~ -35..30) in f186, and a constant radius cannot explain L
#     varying 160-448 px. REJECTED.
#   Model C (monotone-cubic / PCHIP interpolation of the head position
#     (x(t), y(t)) through the anchors, wrist-distance checked): USED.
#     It passes through every anchor, is curved everywhere (no straight
#     chord), passes through the f187 streak-end anchor, and implies a
#     wrist-head distance that dips to ~104-149 px mid-gap (shaft near
#     camera-perpendicular, the same foreshortening seen in follow-through)
#     and recovers to 279-448 px at impact. The dip to 104 px at f186 is
#     the least-constrained number in the whole trail and is flagged as
#     such in the JSON.
# ---------------------------------------------------------------------------

def _pchip_slopes(x, y):
    """Fritsch-Carlson monotone-cubic slopes (no scipy in the venv)."""
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0] = ((2 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    if m[0] * d[0] < 0:
        m[0] = 0.0
    elif d[0] * d[1] < 0 and abs(m[0]) > abs(3 * d[0]):
        m[0] = 3 * d[0]
    m[-1] = ((2 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    if m[-1] * d[-1] < 0:
        m[-1] = 0.0
    elif d[-1] * d[-2] < 0 and abs(m[-1]) > abs(3 * d[-1]):
        m[-1] = 3 * d[-1]
    return m


class _Pchip:
    """Monotone cubic interpolant (shape-preserving, no overshoot)."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.m = _pchip_slopes(self.x, self.y)

    def __call__(self, xq):
        xq = np.asarray(xq, dtype=float)
        out = np.empty_like(xq)
        idx = np.searchsorted(self.x, xq) - 1
        idx = np.clip(idx, 0, len(self.x) - 2)
        h = self.x[idx + 1] - self.x[idx]
        t = (xq - self.x[idx]) / h
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        out = (h00 * self.y[idx] + h10 * h * self.m[idx]
               + h01 * self.y[idx + 1] + h11 * h * self.m[idx + 1])
        return out


# frame -> ((x, y), confidence, source)
# Changes vs v1 ANCHORS: +f182/f183 (Stage 1 found), f184/f185 updated to
# visual-grounding values, f187 v1 anchor REMOVED (it was the hand blur)
# and replaced by a LOW streak-end anchor (120,730).
ANCHORS_V2: dict[int, tuple[tuple[int, int], str, str]] = {
    164: ((15, 790), "low", "v1"),
    166: ((50, 715), "low", "v1"),
    168: ((100, 570), "low", "v1"),
    170: ((95, 445), "medium", "v1"),
    173: ((155, 300), "medium", "v1"),
    176: ((380, 200), "high", "v1"),
    177: ((388, 192), "medium", "v1"),
    179: ((390, 190), "high", "v1"),
    180: ((395, 185), "high", "v1"),
    181: ((380, 200), "high", "v1"),
    182: ((328, 195), "medium", "stage1-grounding"),
    183: ((284, 200), "medium", "stage1-grounding"),
    184: ((215, 214), "medium", "stage1-grounding (v1 205,215 within 10px)"),
    185: ((91, 366), "low", "stage1-grounding streak-end (v1 75,400 within 35px)"),
    187: ((120, 730), "low", "stage1 faint-streak end, not a distinct blob"),
    189: ((200, 1100), "medium", "v1 streak-end"),
    190: ((319, 1159), "inferred", "v1 ball launch point"),
    193: ((690, 640), "medium", "v1"),
    194: ((700, 630), "medium", "v1"),
    195: ((705, 550), "medium", "v1"),
    196: ((710, 460), "medium", "v1"),
    199: ((700, 360), "medium", "v1"),
    201: ((665, 340), "medium", "v1"),
    205: ((690, 330), "medium", "v1"),
    210: ((695, 350), "high", "v1"),
    220: ((700, 340), "high", "v1"),
}

# Gap frames that Stage 1 could not find and Stage 2 simulates.
SIMULATED_FRAMES_V2 = (186, 188, 191, 192)


def build_trail_v2(n_frames: int, wrist_pts=None) -> dict:
    """v2 clubhead trail: PCHIP arc through ANCHORS_V2 for every frame.

    Status per frame: "measured" (high/medium anchor), "low" (low anchor /
    streak-end read), "simulated" (PCHIP, frames 186/188/191/192),
    "inferred" (f190 ball launch point), "bridged" (PCHIP between anchors
    outside the downswing gap, e.g. the long follow-through holds).
    `wrist_pts` (optional, list of wrist (x, y) or None per frame) is used
    only to report the wrist-head distance and shaft angle per frame — the
    trail itself is the PCHIP arc, and the wrist check is reported, not
    used to move the head.
    """
    frames = sorted(ANCHORS_V2)
    xs = np.array([ANCHORS_V2[f][0][0] for f in frames], dtype=float)
    ys = np.array([ANCHORS_V2[f][0][1] for f in frames], dtype=float)
    all_frames = np.arange(min(frames), max(frames) + 1)
    px = _Pchip(np.array(frames, dtype=float), xs)
    py = _Pchip(np.array(frames, dtype=float), ys)
    drawn_x = px(all_frames)
    drawn_y = py(all_frames)

    positions, status, confidence = {}, {}, {}
    wrist_dist, shaft_angle = {}, {}
    for k, f in enumerate(all_frames):
        f = int(f)
        positions[f] = (round(float(drawn_x[k])), round(float(drawn_y[k])))
        if f in ANCHORS_V2:
            conf = ANCHORS_V2[f][1]
            confidence[f] = conf
            status[f] = ("measured" if conf in ("high", "medium")
                         else "inferred" if conf == "inferred" else "low")
        elif f in SIMULATED_FRAMES_V2:
            confidence[f] = "simulated"
            status[f] = "simulated"
        else:
            confidence[f] = "bridged"
            status[f] = "bridged"
        if wrist_pts is not None and f < len(wrist_pts) and wrist_pts[f] is not None:
            wx, wy = wrist_pts[f]
            dx, dy = positions[f][0] - wx, positions[f][1] - wy
            wrist_dist[f] = round(float(np.hypot(dx, dy)), 1)
            shaft_angle[f] = round(float(np.degrees(np.arctan2(dy, dx))), 1)

    # Solid runs: intervals between two MEASURED anchors at most 5 frames
    # apart, with no simulated/low frame between them. Simulated and low
    # frames always draw dotted.
    solid = {int(f): False for f in all_frames}
    measured_frames = [f for f in frames
                       if ANCHORS_V2[f][1] in ("high", "medium")]
    for a, b in zip(measured_frames, measured_frames[1:]):
        if b - a <= 5:
            seg = list(range(a, b + 1))
            if all(status[f] == "measured" for f in seg):
                for f in seg:
                    solid[f] = True

    return {
        "version": 2,
        "positions": positions,
        "measured": {f: status[f] == "measured" for f in all_frames},
        "status": {int(f): status[int(f)] for f in all_frames},
        "confidence": confidence,
        "solid": {int(f): bool(solid[int(f)]) for f in all_frames},
        "simulated": {int(f): status[int(f)] == "simulated"
                      for f in all_frames},
        "wrist_dist": wrist_dist,
        "shaft_angle": shaft_angle,
        "anchors": {f: {"xy": list(ANCHORS_V2[f][0]),
                        "confidence": ANCHORS_V2[f][1],
                        "source": ANCHORS_V2[f][2]}
                    for f in frames},
        "first_frame": min(frames),
        "last_frame": max(frames),
        "blur_gap": [0, -1],
        "simulated_frames": list(SIMULATED_FRAMES_V2),
    }


def model_comparison_v2(wrist_pts) -> dict:
    """Check the two rejected models against the v2 anchors (Stage 2).

    Model A: rigid rod about the wrist, shaft angle (unwrapped) and L
    interpolated (PCHIP) from the anchors. Model B: circle about the body
    centre, fitted to the downswing anchors. Both are reported with their
    gap-frame predictions so the rejection is checkable.
    """
    import math
    frames = sorted(ANCHORS_V2)
    # Unwrapped shaft angles and L per anchor.
    th, L = {}, {}
    for f in frames:
        hx, hy = ANCHORS_V2[f][0]
        wx, wy = wrist_pts[f]
        dx, dy = hx - wx, hy - wy
        L[f] = float(np.hypot(dx, dy))
        th[f] = float(np.degrees(np.arctan2(dy, dx)))
    unwrapped, prev = [], None
    for f in frames:
        t = th[f]
        if prev is not None:
            while t - prev > 180:
                t -= 360
            while t - prev < -180:
                t += 360
        unwrapped.append(t)
        prev = t
    fs = np.array(frames, dtype=float)
    ptu = _Pchip(fs, np.array(unwrapped))
    pL = _Pchip(fs, np.array([L[f] for f in frames]))
    model_a = {}
    for f in SIMULATED_FRAMES_V2:
        t = float(ptu(f))
        l = float(pL(f))
        wx, wy = wrist_pts[f]
        model_a[f] = {
            "theta_deg": round(t, 1),
            "L_px": round(l, 1),
            "xy": [round(wx + l * math.cos(math.radians(t))),
                   round(wy + l * math.sin(math.radians(t)))],
        }
    # Model B: circle fit to downswing anchors f181..f190.
    pts = np.array([ANCHORS_V2[f][0] for f in
                    (181, 182, 183, 184, 185, 189, 190)], dtype=float)
    cx, cy, r = 250.0, 650.0, 450.0
    for _ in range(60):
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        res = d - r
        J = np.column_stack([-(pts[:, 0] - cx) / d,
                             -(pts[:, 1] - cy) / d,
                             -np.ones(len(pts))])
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.max(np.abs(step)) < 1e-9:
            break
    resid = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
    # Polar angle of each anchor about the centre, unwrapped, PCHIP in time.
    ph = [float(np.degrees(np.arctan2(y - cy, x - cx))) for x, y in pts]
    unw, prev = [], None
    for t in ph:
        if prev is not None:
            while t - prev > 180:
                t -= 360
            while t - prev < -180:
                t += 360
        unw.append(t)
        prev = t
    pp = _Pchip(np.array((181, 182, 183, 184, 185, 189, 190), dtype=float),
                np.array(unw))
    model_b = {}
    for f in SIMULATED_FRAMES_V2:
        if f in (191, 192):
            continue  # circle model is a downswing model only
        a = math.radians(float(pp(f)))
        model_b[f] = {"xy": [round(cx + r * math.cos(a)),
                             round(cy + r * math.sin(a))]}
    return {
        "club_length_calibration_px": {
            "top_f176_181": [round(L[f], 1) for f in (176, 177, 179, 180, 181)],
            "f184": round(L[184], 1),
            "f185": round(L[185], 1),
            "impact_f189_190": [round(L[189], 1), round(L[190], 1)],
            "follow_through_f193_196": [round(L[f], 1) for f in (193, 194, 195, 196)],
            "finish_f210_220": [round(L[210], 1), round(L[220], 1)],
            "note": ("Pixel length is foreshortened when the shaft points "
                     "toward/away from the camera; it is not a fixed "
                     "constant. Max ~448 px at impact (shaft in plane), "
                     "~160 px in follow-through."),
        },
        "model_a_rigid_rod_wrist": {
            "predictions": model_a,
            "verdict": ("NOT USED as the generator, but no longer "
                        "discrepant once the f187 streak-end anchor is in: "
                        "it lands within ~55 px of the PCHIP arc in the "
                        "downswing (f186 (46,559) vs (100,543); f188 "
                        "(125,912) vs (149,940)). Without that anchor it "
                        "predicted the head off the left edge, because it "
                        "assumes L rises monotonically 197->401 and misses "
                        "the foreshortening dip. Post-impact it diverges "
                        "more (f191 (591,1049) vs (465,1028); f192 (719,767) "
                        "vs (613,782), ~110-125 px) — the shaft-angle "
                        "unwrap is unreliable through the wrist release, so "
                        "the PCHIP arc is used there too."),
        },
        "model_b_circle_body_centre": {
            "centre": [round(float(cx), 1), round(float(cy), 1)],
            "radius_px": round(float(r), 1),
            "anchor_residuals_px": [round(float(v), 1) for v in resid],
            "predictions": model_b,
            "verdict": ("REJECTED: fits the anchors to +/-25 px but "
                        "predicts the head at the left edge (x~-35..30) in "
                        "f186, and a constant radius cannot explain L "
                        "varying 160-448 px. With the f187 streak anchor "
                        "included, Model B f188 (65,977) is ~85 px from the "
                        "PCHIP point (149,940)."),
        },
        "model_c_pchip_arc_used": {
            "note": ("PCHIP (x(t), y(t)) through the anchors. Curved "
                     "everywhere, passes through the f187 streak-end "
                     "anchor, and implies a wrist-head distance dipping "
                     "to ~104-149 px mid-gap (shaft near "
                     "camera-perpendicular, the same foreshortening seen "
                     "in follow-through L~160) and recovering to 279-448 "
                     "px at impact. The f186 dip to 104 px is the "
                     "least-constrained number in the trail."),
        },
    }
