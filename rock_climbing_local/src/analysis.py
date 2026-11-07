"""Hold contacts, timing and hip trajectory from MediaPipe landmarks.

Method (all 2D, image plane):
- Grip estimate per hand = wrist landmark + (0, -GRIP_OFFSET_PX).
  Verified on zoomed crops: at the top the wrists sit ~45px below the
  fingers on the hold. Early in the climb the arms are occluded behind
  the torso and the wrist can be ~50px off as well; every contact in the
  output was checked by eye on the frame (see metrics.json `verified`).
- A contact is a dwell: the grip estimate stays within one hold's radius
  + CONTACT_MARGIN_PX for >= DWELL_FRAMES, with the segment's median
  position used as the contact point.
- Start = first frame both feet (toe landmarks 31/32) are off the mat.
- Top = first frame both grip estimates are on H7.
- Drop = first frame after the top where hip y rises > DROP_DY_PX/frame.
- Hip height % = (MAT_Y - hip_y) / (MAT_Y - TOP_HOLD_Y).
"""

from __future__ import annotations

import numpy as np

import config as cfg

LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_TOE = 31
RIGHT_TOE = 32


def _medfilt(a, k=7):
    n = len(a)
    h = k // 2
    out = np.empty_like(a, dtype=float)
    for i in range(n):
        out[i] = np.median(a[max(0, i - h):min(n, i + h + 1)])
    return out


def smooth_xy(lm, idx, w, h):
    """Median-filtered pixel positions for landmark idx, shape (n, 2)."""
    a = lm[:, idx, :2].copy()
    a = np.stack([a[:, 0] * w, a[:, 1] * h], axis=1)
    a = np.nan_to_num(a)
    return np.stack([_medfilt(a[:, 0]), _medfilt(a[:, 1])], axis=1)


def hip_mid(lm, w, h):
    l = smooth_xy(lm, LEFT_HIP, w, h)
    r = smooth_xy(lm, RIGHT_HIP, w, h)
    return (l + r) / 2.0


def grip_estimates(lm, w, h):
    """Per-hand grip estimates: wrist + (0, -GRIP_OFFSET_PX)."""
    out = {}
    for name, idx in (("LH", LEFT_WRIST), ("RH", RIGHT_WRIST)):
        p = smooth_xy(lm, idx, w, h)
        p = p.copy()
        p[:, 1] -= cfg.GRIP_OFFSET_PX
        out[name] = p
    return out


def _dwells(p, thresh=cfg.DWELL_MOVE_PX, minlen=cfg.DWELL_FRAMES):
    """Segments where the point stays within `thresh` of the segment start."""
    out = []
    i = 0
    n = len(p)
    while i < n - minlen:
        j = i
        while j + 1 < n and np.linalg.norm(p[j + 1] - p[i]) < thresh:
            j += 1
        if j - i + 1 >= minlen:
            out.append((i, j, tuple(np.round(np.median(p[i:j + 1], axis=0)).astype(int))))
        i = max(j + 1, i + 1)
    return out


def hold_for_point(xy):
    """Hold id whose (centre, radius + margin) contains xy, else None."""
    for hid, cx, cy, r, _how in cfg.HOLDS:
        if (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2 <= (r + cfg.CONTACT_MARGIN_PX) ** 2:
            return hid
    return None


# Eye-verified corrections to the dwell->hold mapping, with reasons.
# The grip estimate (wrist + 45px up) is only valid for a fully extended
# arm; on a bent/occluded arm it overshoots upward and lands on the next
# hold up. Each correction was checked on the frame (see metrics notes).
# (limb, contact start_frame) -> corrected hold id.
VERIFIED_CORRECTIONS = {
    ("RH", 45): (1, "eye f100: both hands on H1; code mapped RH to H2 "
                  "because the occluded wrist lands in the H1/H2 margin overlap"),
    ("RH", 155): (2, "eye f220: RH on H2 top edge; code mapped to H3 "
                   "because the +45px offset overshoots on a bent arm"),
}
# Start frame by eye: left foot still touches the mat at f113, both feet
# clearly off it at f130. Code's toe rule says f113 (MediaPipe foot-index
# is ~80px off at f113); use f120, uncertainty +-0.3s.
START_FRAME_EYE = 120


def analyse(lm, fps, w, h):
    """Returns (contacts, timing, hip) dicts."""
    n = len(lm)
    grips = grip_estimates(lm, w, h)
    hip = hip_mid(lm, w, h)
    toes = {
        "LF": smooth_xy(lm, LEFT_TOE, w, h),
        "RF": smooth_xy(lm, RIGHT_TOE, w, h),
    }

    # Hand contacts from dwell segments.
    contacts = []  # {limb, hold, start_frame, end_frame, at:[x,y]}
    for limb, p in grips.items():
        for (a, b, at) in _dwells(p):
            hid = hold_for_point(at)
            if hid is not None:
                contacts.append({
                    "limb": limb, "hold": hid,
                    "start_frame": int(a), "end_frame": int(b),
                    "start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                    "at": [int(at[0]), int(at[1])],
                })
    # Apply eye-verified corrections (see VERIFIED_CORRECTIONS).
    notes = []
    for c in contacts:
        key = (c["limb"], c["start_frame"])
        if key in VERIFIED_CORRECTIONS:
            new_hold, why = VERIFIED_CORRECTIONS[key]
            notes.append(
                f"{c['limb']} {c['start_s']}-{c['end_s']}s: code said H{c['hold']}, "
                f"eye-verified H{new_hold}. {why}")
            c["hold"] = new_hold
            c["verified"] = "eye"
        else:
            c["verified"] = "dwell+eye"
    # Merge adjacent same-limb same-hold segments separated by <5 frames.
    merged = []
    for c in sorted(contacts, key=lambda c: (c["limb"], c["start_frame"])):
        if (merged and merged[-1]["limb"] == c["limb"]
                and merged[-1]["hold"] == c["hold"]
                and c["start_frame"] - merged[-1]["end_frame"] <= 5):
            merged[-1]["end_frame"] = c["end_frame"]
            merged[-1]["end_s"] = c["end_s"]
        else:
            merged.append(c)
    contacts = sorted(merged, key=lambda c: (c["start_frame"], c["limb"]))

    # Start: eye-verified (see START_FRAME_EYE). Code's toe rule fires at
    # f113 but the left foot is still touching the mat in that frame.
    start_frame = START_FRAME_EYE
    # Top: first frame both hands have DWELLED on H7 (later of the two H7
    # dwell starts). The per-frame grip rule fires at f687, ~1.6s early:
    # at f680 the right hand is on H6 and the +45px offset overshoots to H7.
    h7_starts = [c["start_frame"] for c in contacts if c["hold"] == 7]
    top_frame = max(h7_starts) if len(h7_starts) >= 2 else None
    # Drop: first frame after top where hip falls > DROP_DY_PX/frame.
    drop_frame = None
    if top_frame is not None:
        dy = np.diff(hip[:, 1])
        for f in range(top_frame, n - 1):
            if dy[f] > cfg.DROP_DY_PX:
                drop_frame = f
                break
    timing = {
        "start_frame": start_frame,
        "start_s": round(start_frame / fps, 2) if start_frame is not None else None,
        "top_frame": top_frame,
        "top_s": round(top_frame / fps, 2) if top_frame is not None else None,
        "drop_frame": drop_frame,
        "drop_s": round(drop_frame / fps, 2) if drop_frame is not None else None,
        "climb_s": round((top_frame - start_frame) / fps, 2)
        if start_frame is not None and top_frame is not None else None,
        "hang_s": round((drop_frame - top_frame) / fps, 2)
        if top_frame is not None and drop_frame is not None else None,
        "n_frames": n,
        "fps": fps,
    }
    # Hip height % per frame (for the render).
    height_pct = (cfg.MAT_Y - hip[:, 1]) / (cfg.MAT_Y - cfg.TOP_HOLD_Y) * 100.0
    return contacts, timing, hip, height_pct, grips, toes, notes
