"""Swing analysis from per-frame 2D landmarks. All heuristic, all 2D.

Signal: the hands (midpoint of the two wrists), height in the frame,
normalised by torso length. In a DTL swing the hands rise through the
backswing, peak at the top, fall to impact, and rise again in the
follow-through. The 8 GolfDB events are read off that curve:

  address -> toe-up -> mid-backswing -> top -> mid-downswing -> impact
  -> mid-follow-through -> finish

Tempo = backswing time (address -> top) vs downswing time (top -> impact).
Spine angle = angle of shoulder-mid -> hip-mid from vertical, in the image
plane, at address / top / impact. DTL only. NOT a 3D angle.
"""

from __future__ import annotations

import numpy as np

import config as cfg
from src import pose as pose_mod

EVENTS = [
    "address",
    "toe-up",
    "mid-backswing",
    "top",
    "mid-downswing",
    "impact",
    "mid-follow-through",
    "finish",
]


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _interp_missing(series: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs; leading/trailing NaNs take nearest value."""
    n = len(series)
    if n == 0:
        return series
    idx = np.arange(n)
    good = ~np.isnan(series)
    if not good.any():
        return np.zeros(n)
    if good.sum() == 1:
        return np.full(n, series[good][0])
    return np.interp(idx, idx[good], series[good])


def extract_signals(landmarks, width, height):
    """Return dict of per-frame signals, with coverage stats."""
    n = len(landmarks)
    hand_y = np.full(n, np.nan)
    hand_x = np.full(n, np.nan)
    torso = np.full(n, np.nan)
    shoulder_mid = np.full((n, 2), np.nan)
    hip_mid = np.full((n, 2), np.nan)
    present = np.zeros(n, dtype=bool)

    for i, lm in enumerate(landmarks):
        if lm is None:
            continue
        present[i] = True
        ls = pose_mod.landmark_xy(lm, pose_mod.LEFT_SHOULDER, width, height)
        rs = pose_mod.landmark_xy(lm, pose_mod.RIGHT_SHOULDER, width, height)
        lh = pose_mod.landmark_xy(lm, pose_mod.LEFT_HIP, width, height)
        rh = pose_mod.landmark_xy(lm, pose_mod.RIGHT_HIP, width, height)
        lw = pose_mod.landmark_xy(lm, pose_mod.LEFT_WRIST, width, height)
        rw = pose_mod.landmark_xy(lm, pose_mod.RIGHT_WRIST, width, height)
        sm = pose_mod.midpoint(ls, rs)
        hm = pose_mod.midpoint(lh, rh)
        wm = pose_mod.midpoint(lw, rw)
        if sm is not None:
            shoulder_mid[i] = sm
        if hm is not None:
            hip_mid[i] = hm
        if wm is not None:
            hand_x[i], hand_y[i] = wm
        if sm is not None and hm is not None:
            torso[i] = float(np.hypot(sm[0] - hm[0], sm[1] - hm[1]))

    torso_med = float(np.nanmedian(torso)) if np.isfinite(torso).any() else 1.0
    if torso_med <= 0:
        torso_med = 1.0
    # Hand height above the hips, in torso lengths. Larger = hands higher.
    hip_y = hip_mid[:, 1]
    hand_height = (_interp_missing(hip_y) - _interp_missing(hand_y)) / torso_med
    hand_height = _smooth(hand_height, cfg.SMOOTH_WINDOW)
    hand_x_s = _smooth(_interp_missing(hand_x), cfg.SMOOTH_WINDOW)
    return {
        "hand_height": hand_height,
        "hand_x": hand_x_s,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "torso_med_px": torso_med,
        "coverage": float(present.mean()) if n else 0.0,
        "n_frames": n,
        "n_present": int(present.sum()),
    }


def _first_crossing(sig, start, end, level, rising=True):
    """First frame in [start, end) where sig crosses level."""
    for i in range(start, max(start + 1, end)):
        a, b = sig[i - 1], sig[i]
        if rising and a < level <= b:
            return i
        if not rising and a > level >= b:
            return i
    return None


def _argmax(sig, start, end):
    if end <= start:
        return start
    return int(start + int(np.argmax(sig[start:end])))


def _argmin(sig, start, end):
    if end <= start:
        return start
    return int(start + int(np.argmin(sig[start:end])))


def _first_prominent_peak(sig, start, end, min_height, drop, lookahead_s, fps):
    """First local maximum in [start, end) that is a real peak, not a wiggle.

    A peak must be a local max over +/-0.75 s AND be followed by a fall of
    at least `drop` within `lookahead_s`. The fall requirement is what
    separates a true top from a noise bump on a slow rise: on slow-motion
    source the backswing takes seconds, and a +/-8-frame window calls every
    small bump on the way up a "peak". Taking the FIRST such peak matters
    too: the hands peak twice (top, then follow-through), and the
    follow-through peak is often higher, so a global argmax lands on the
    finish, not the top.
    """
    window = max(8, int(round(0.75 * fps)))
    lookahead = max(1, int(round(lookahead_s * fps)))
    for i in range(max(start, window), min(end, len(sig) - window)):
        if sig[i] < min_height:
            continue
        if sig[i] != max(sig[i - window:i + window + 1]):
            continue
        future_min = float(np.min(sig[i:min(len(sig), i + lookahead)]))
        if sig[i] - future_min >= drop:
            return i
    return None


def segment_phases(sig, fps):
    """Find the 8 events. Returns {event: frame_index} and a notes list."""
    n = len(sig)
    notes: list[str] = []
    # Address level: median hand height over the first ADDRESS_STABLE_SECONDS.
    n_addr = max(1, int(round(cfg.ADDRESS_STABLE_SECONDS * fps)))
    addr_level = float(np.median(sig[:n_addr]))
    # Global max after the first second, only to set the scale.
    search_from = min(n - 1, int(round(1.0 * fps)))
    global_max = float(np.max(sig[search_from:]))
    rng_global = global_max - addr_level
    if rng_global < cfg.SIGNAL_MIN_PROMINENCE:
        notes.append(
            f"hand-height range too small ({rng_global:.2f} torso lengths); "
            "phase detection is unreliable on this clip"
        )
    # Top: FIRST prominent peak, not the global max (see _first_prominent_peak).
    top = _first_prominent_peak(
        sig, search_from, n, addr_level + 0.5 * rng_global,
        drop=0.25 * rng_global, lookahead_s=4.0, fps=fps,
    )
    if top is None:
        top = _argmax(sig, search_from, n)
        notes.append("no prominent top peak found; fell back to global max")
    top_level = float(sig[top])
    rng = top_level - addr_level
    # Impact: the deep dip after the top. Find where the signal has really
    # started falling (below top - 0.5*range), then take the minimum before
    # it recovers to within 0.3*range of the top. A fixed 2 s window is
    # wrong on slow-motion source, where top -> impact takes seconds.
    fall_level = top_level - 0.5 * rng
    recover_level = top_level - 0.3 * rng
    fell_at = None
    for i in range(top, n):
        if sig[i] < fall_level:
            fell_at = i
            break
    if fell_at is not None:
        recover_at = n
        for i in range(fell_at, n):
            if sig[i] > recover_level:
                recover_at = i
                break
        impact = _argmin(sig, fell_at, recover_at)
    else:
        impact_end = min(n, top + int(round(cfg.IMPACT_SEARCH_SECONDS * fps)))
        impact = _argmin(sig, top, impact_end)
        notes.append("no deep dip after top; impact is the min in a 2 s window")
    # Finish: FIRST prominent peak after impact (the held finish pose), not
    # the last frame and not the global max after impact.
    max_after_impact = float(np.max(sig[impact:])) if impact < n else 0.0
    finish = _first_prominent_peak(
        sig, impact, n,
        float(sig[impact]) + 0.5 * (max_after_impact - float(sig[impact])),
        drop=0.2, lookahead_s=3.0, fps=fps,
    )
    if finish is None:
        finish = _argmax(sig, impact, n)
        notes.append("no prominent finish peak found; fell back to max after impact")
    # Crossings on the way up (address -> top) and down (top -> impact).
    toe_up = _first_crossing(
        sig, 0, top, addr_level + cfg.TOE_UP_FRACTION * rng, rising=True
    )
    mid_bs = _first_crossing(
        sig, 0, top, addr_level + cfg.MID_BACKSWING_FRACTION * rng, rising=True
    )
    mid_ds = _first_crossing(
        sig, top, impact, addr_level + cfg.MID_DOWNSWING_FRACTION * rng,
        rising=False,
    )
    # Mid-follow-through: halfway up from impact to finish.
    mid_ft = _first_crossing(
        sig, impact, finish,
        float(sig[impact]) + 0.5 * (float(sig[finish]) - float(sig[impact])),
        rising=True,
    )
    # Address: the low point of the hands just before the takeaway, not
    # frame 0 — a golfer may waggle / stand at address for seconds before
    # moving, and that idle time is not backswing.
    if toe_up is not None and toe_up > 0:
        address = _argmin(sig, 0, toe_up + 1)
    else:
        address = 0
    events = {
        "address": address,
        "toe-up": toe_up if toe_up is not None else max(0, top // 3),
        "mid-backswing": mid_bs if mid_bs is not None else max(0, 2 * top // 3),
        "top": top,
        "mid-downswing": mid_ds if mid_ds is not None else (top + impact) // 2,
        "impact": impact,
        "mid-follow-through": mid_ft if mid_ft is not None else (impact + finish) // 2,
        "finish": finish,
    }
    for ev in ("toe-up", "mid-backswing", "mid-downswing", "mid-follow-through"):
        if events[ev] is None:
            notes.append(f"{ev} crossing not found; fell back to interpolation")
    # Monotonicity guard: events must be in order.
    ordered = sorted(events.items(), key=lambda kv: kv[1])
    if [k for k, _ in ordered] != EVENTS:
        notes.append("events out of order after detection; kept as detected")
    return events, notes


def spine_angle(shoulder_mid, hip_mid, frame):
    """2D spine angle at one frame: shoulder-mid -> hip-mid vs vertical, deg.

    Returns None if either point is missing. Positive = top of the spine
    leaning toward the target side of the frame.
    """
    if frame < 0 or frame >= len(shoulder_mid):
        return None
    sm, hm = shoulder_mid[frame], hip_mid[frame]
    if not (np.isfinite(sm).all() and np.isfinite(hm).all()):
        return None
    dx = sm[0] - hm[0]
    dy = sm[1] - hm[1]  # image y grows downward; spine points up, so dy < 0
    if dy == 0:
        return None
    angle = float(np.degrees(np.arctan2(dx, -dy)))
    return angle


def phase_at_frame(events, frame):
    """Event name in force at this frame (the last event at or before it)."""
    current = "address"
    for ev in EVENTS:
        if events[ev] <= frame:
            current = ev
    return current


def analyse(landmarks, width, height, fps):
    sig = extract_signals(landmarks, width, height)
    events, notes = segment_phases(sig["hand_height"], fps)
    addr, top, impact = events["address"], events["top"], events["impact"]
    backswing_s = (top - addr) / fps
    downswing_s = (impact - top) / fps
    tempo_ratio = backswing_s / downswing_s if downswing_s > 0 else None
    spine = {
        ev: spine_angle(sig["shoulder_mid"], sig["hip_mid"], events[ev])
        for ev in cfg.SPINE_AT
    }
    return {
        "events": events,
        "event_times_s": {ev: round(events[ev] / fps, 3) for ev in EVENTS},
        "backswing_s": round(backswing_s, 3),
        "downswing_s": round(downswing_s, 3),
        "tempo_ratio": round(tempo_ratio, 2) if tempo_ratio else None,
        "spine_angle_2d_deg": {
            ev: (round(v, 1) if v is not None else None) for ev, v in spine.items()
        },
        "pose_coverage": round(sig["coverage"], 3),
        "frames_with_pose": sig["n_present"],
        "n_frames": sig["n_frames"],
        "fps": round(float(fps), 3),
        "notes": notes,
        "signals": sig,
    }
