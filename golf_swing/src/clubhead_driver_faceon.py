"""Clubhead trail + ball dots for Peng's DRIVER FACE-ON clip
(golf_driver_faceon, trimmed 15-21s of the original, 720x1280, 30fps).

Frames are TRIMMED-clip frames. Trim start = original 15.0s (f450);
impact trimmed f106 = original f556 (ball on tee f105, clubhead contact
f106, tee empty f107 -- verified frame by frame on tee crops).
Takeaway trimmed f71 = original f521 (wrist midpoint dead steady f60-70
at y~645, first sustained move f71; top trimmed f95 = original f545,
wrist-y minimum 364px; pipeline top f94, 1 frame earlier -- tempo uses
the pipeline's f94: back 0.77s / down 0.40s = 1.93:1; manual wrist-min
top f95 gives 2.18:1, i.e. tempo is 1.9-2.2:1 within the +-1 frame top
ambiguity).

Method (same honesty rules as the DTL driver and face-on iron clubhead):
- Anchors are eye-verified sightings on FULL-RES frames (or crops).
  Automated detection was NOT used: Hough locks onto static lines,
  dark-blob search is swamped by trees/net/poles.
- Where the head is out of frame or occluded, NO trail is drawn for
  those frames -- a line there would invent a position. There is NO
  simulated top arc in this version: at the top (f90 full-res) NO shaft
  is visible in frame at all, so any arc would be pure guesswork.
- Gaps: f79-103 (head out of frame through the whole top; the shaft
  points up-left out of frame) and f109-115 (head wraps behind the body
  / out of the top of frame; f110 verified absent in a crop, f112/f114
  show shaft only).
- Two tile-read anchors from the first pass were REMOVED after crop
  checks: f100 (no head in the (80-300, 560-740) crop -- the "blob" was
  a misread) and f110 (no head in the (0-200, 620-820) crop -- the blob
  was the player's leg/shadow). f104 was corrected from (15,300) to
  (15,405) on the crop (blur at the bottom-left corner).
- Ball: launch point only + 2 confirmed flight points. f107 (447,767)
  clear white ball in a crop, medium; f108 (433,645) clear white ball
  in a crop, medium-high. f109/f110 differencing candidates do NOT form
  a consistent chain with the visual points (dy -122 then ~0), so they
  are NOT claimed. With only 2 flight points NO physics arc is fitted
  and NO ball speed / distance / launch angle is claimed. Dots only.
- NO smoothing on the trail: the first render used a sliding quadratic
  over f104-108 and pulled the impact point off the ball. Trail is raw
  anchors + linear interp between anchors <=5 frames apart.
- Downswing foreshortening: f105 head (330,720) is a big blurred oval
  only ~100px from the hands -- the shaft points nearly at the camera
  that frame, so the head's IMAGE position jumps ~490px from f104 and
  ~270px to f106 while its true 3D speed stays plausible. Do not read
  image-plane speed off these three frames.
"""

from __future__ import annotations

import numpy as np

BALL_LAUNCH = (395, 985)  # ball centre on the tee, last seen f106
BALL_LAUNCH_FRAME = 106
BALL_FLIGHT = {
    107: ((447, 767), "medium"),       # clear white ball in crop, v107
    108: ((433, 645), "medium-high"),  # clear white ball in crop, fl-02
}

FIRST_FRAME = 68
LAST_FRAME = 133

# frame -> ((x, y), confidence)
# high: head clearly visible as a distinct blob at the end of the shaft
# medium: head visible but small / blurred / at the frame edge
ANCHORS: dict[int, tuple[tuple[int, int], str]] = {
    68: ((395, 985), "high"),    # address, head at the ball
    70: ((393, 983), "high"),    # address, head at the ball
    72: ((390, 980), "high"),    # address, head at the ball
    74: ((358, 998), "medium"),  # takeaway, head just behind the ball
    76: ((290, 855), "medium"),  # head lifted, shaft diagonal
    78: ((10, 690), "medium"),   # head half out of frame, left edge
    104: ((15, 405), "medium"),  # re-entry, head at left edge (full-res)
    105: ((330, 720), "medium"), # head blurred, shaft foreshortened
    106: ((390, 980), "high"),   # IMPACT, head at the ball (tee crop)
    107: ((390, 845), "medium"), # head just past impact (crop grey oval)
    108: ((210, 725), "medium"), # follow-through, head left (full-res)
    116: ((235, 235), "high"),   # follow-through, head top-left
    118: ((390, 540), "high"),   # follow-through, head mid-right
    120: ((460, 620), "high"),   # follow-through, head right
    124: ((460, 620), "high"),   # held finish, head settles
    128: ((460, 655), "high"),   # held finish, head settles
    133: ((490, 560), "high"),   # held finish, head right
}

# Frames where the head is out of frame / occluded: no trail, no marker.
GAPS = [(79, 103), (109, 115)]


def _interp_segment(frames, anchors):
    """Linear interp between anchors. NO smoothing: smoothing pulled the
    impact point off the ball in the first render (sharp V at impact)."""
    fs = sorted(frames)
    xs = np.array([anchors[f][0][0] for f in fs], dtype=float)
    ys = np.array([anchors[f][0][1] for f in fs], dtype=float)
    fr = np.arange(fs[0], fs[-1] + 1)
    ix = np.interp(fr, fs, xs)
    iy = np.interp(fr, fs, ys)
    out = {}
    for k, f in enumerate(fr):
        out[int(f)] = (round(float(ix[k])), round(float(iy[k])))
    return out


def build_trail() -> dict:
    positions: dict[int, tuple[int, int] | None] = {}
    measured: dict[int, bool] = {}
    solid: dict[int, bool] = {}
    confidence: dict[int, str] = {}

    for f in range(FIRST_FRAME, LAST_FRAME + 1):
        positions[f] = None
        measured[f] = False
        solid[f] = False
        confidence[f] = "out-of-frame"

    # Segment A: f68-78, anchors, solid.
    seg = _interp_segment([68, 70, 72, 74, 76, 78], ANCHORS)
    for f, p in seg.items():
        positions[f] = p
        measured[f] = f in ANCHORS
        solid[f] = True
        confidence[f] = ANCHORS[f][1] if f in ANCHORS else "bridged"
    # Segment B: f104-108, anchors every frame, solid.
    seg = _interp_segment([104, 105, 106, 107, 108], ANCHORS)
    for f, p in seg.items():
        positions[f] = p
        measured[f] = f in ANCHORS
        solid[f] = True
        confidence[f] = ANCHORS[f][1] if f in ANCHORS else "bridged"
    # Segment C: f116-133, anchors, solid.
    seg = _interp_segment([116, 118, 120, 124, 128, 133], ANCHORS)
    for f, p in seg.items():
        positions[f] = p
        measured[f] = f in ANCHORS
        solid[f] = True
        confidence[f] = ANCHORS[f][1] if f in ANCHORS else "bridged"

    n_measured = sum(1 for f in positions if measured.get(f))
    n_bridged = sum(
        1 for f in positions
        if positions[f] is not None and not measured.get(f)
    )
    n_gap = sum(1 for f in positions if positions[f] is None)
    return {
        "positions": positions,
        "measured": measured,
        "solid": solid,
        "confidence": confidence,
        "anchors": {f: {"xy": list(ANCHORS[f][0]), "confidence": ANCHORS[f][1]}
                    for f in ANCHORS},
        "gaps": GAPS,
        "ball_launch": BALL_LAUNCH,
        "ball_launch_frame": BALL_LAUNCH_FRAME,
        "ball_flight": {f: {"xy": list(v[0]), "confidence": v[1]}
                        for f, v in BALL_FLIGHT.items()},
        "first_frame": FIRST_FRAME,
        "last_frame": LAST_FRAME,
        "n_measured": n_measured,
        "n_bridged": n_bridged,
        "n_gap": n_gap,
    }
