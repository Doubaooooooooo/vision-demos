"""Colour ball + rim detection and tracking, local only (OpenCV, no API key).

The ball is orange in bright light and dark red against the gym ceiling, so
two masks are combined: an HSV orange mask and a "R is clearly above G and
B" rule. The red floor logo (y ~ 680) and the player's pink shoes are the
main false positives; flight tracking only accepts candidates with y below
FLIGHT_MAX_Y, and tracks must rise and fall and travel toward the hoop.

The rim is the red rim + backboard-padding blob in the right-hand part of
the frame. It is the MEDIAN of rim-region candidates over a +-30 frame
window: the real rim is there for most of the window, the ball only for a
few frames, so the median locks onto the rim. A per-frame "nearest blob"
tracker was tried first and failed: it followed the ball down through the
rim and the crossing never fired.
"""

from __future__ import annotations

import cv2
import numpy as np

import config as cfg


def candidates(frame):
    """All ball-like blobs in one frame, largest first.

    Returns a list of dicts {x, y, area, circ} in pixel coordinates.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m_or = cv2.inRange(
        hsv,
        np.array(cfg.HSV_LO, dtype=np.uint8),
        np.array(cfg.HSV_HI, dtype=np.uint8),
    )
    b, g, r = cv2.split(frame.astype(np.int16))
    m_red = (
        (r - g > cfg.RED_R_MINUS_G)
        & (r - b > cfg.RED_R_MINUS_B)
        & (r > cfg.RED_R_MIN)
        & (r < cfg.RED_R_MAX)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(m_or, m_red)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < cfg.BALL_MIN_AREA or a > cfg.BALL_MAX_AREA:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (per * per) if per else 0.0
        if circ < cfg.BALL_MIN_CIRCULARITY:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append(
            {
                "x": int(round(m["m10"] / m["m00"])),
                "y": int(round(m["m01"] / m["m00"])),
                "area": int(round(a)),
                "circ": round(float(circ), 2),
            }
        )
    out.sort(key=lambda d: -d["area"])
    return out[: cfg.BALL_MAX_CANDIDATES]


def detect_all(video_path):
    """Per-frame ball candidates for the whole clip.

    Returns (candidates_per_frame, fps, width, height, n_frames).
    Streaming: frames are not kept.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    per_frame = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        per_frame.append(candidates(frame))
    cap.release()
    return per_frame, fps, width, height, len(per_frame)


def _static_mask(per_frame):
    """Mark candidates that are (nearly) static objects, not the ball.

    The rim blob and the red floor logo sit at almost the same pixel for
    many frames (they move only with the slow camera pan). The ball never
    does that: even at the apex it still travels ~8 px/frame horizontally.
    A candidate with a near-duplicate within 6 px in at least 4 of the
    previous 6 frames is static and is excluded from ball linking. (It is
    still kept in per_frame for rim detection.)
    Returns a list of sets of candidate indices to skip, per frame.
    """
    skip = []
    for f, cands in enumerate(per_frame):
        s = set()
        for j, c in enumerate(cands):
            hits = 0
            for g in range(max(0, f - 6), f):
                for d in per_frame[g]:
                    if ((d["x"] - c["x"]) ** 2 + (d["y"] - c["y"]) ** 2) ** 0.5 <= 6:
                        hits += 1
                        break
            if hits >= 4:
                s.add(j)
        skip.append(s)
    return skip


def link_tracks(per_frame, rim=None):
    """Link per-frame candidates into tracks.

    A track is a dict {points: {frame: (x, y)}, bridged: [frames]}. Only
    candidates with y < FLIGHT_MAX_Y are linked. Static blobs (rim, floor
    logo) are excluded by _static_mask, and any candidate within
    RIM_EXCLUDE_PX of the rim centre is excluded too: for 2-3 frames around
    the rim the ball and the red rim merge into one blob, and the linker
    would otherwise lock onto the rim and follow it instead of the ball.
    Those frames become a gap that merge_tracks bridges (drawn dotted).
    A track dies after TRACK_MAX_GAP frames with no matching candidate;
    gaps of <= BRIDGE_MAX_GAP inside a track are filled by linear
    interpolation and recorded in `bridged`.
    """
    skip = _static_mask(per_frame)
    tracks = []
    active = []  # each: {track, last_xy, vel, last_frame}
    for f, cands in enumerate(per_frame):
        rim_xy = rim.get(f, (None, None, False))[:2] if rim else (None, None)
        kept = []
        for j, c in enumerate(cands):
            if c["y"] >= cfg.FLIGHT_MAX_Y or j in skip[f]:
                continue
            if rim_xy[0] is not None:
                if ((c["x"] - rim_xy[0]) ** 2 + (c["y"] - rim_xy[1]) ** 2) ** 0.5 < cfg.RIM_EXCLUDE_PX:
                    continue
            kept.append(c)
        cands = kept
        used = set()
        still_active = []
        for a in active:
            if f - a["last_frame"] > cfg.TRACK_MAX_GAP:
                continue  # dead
            px = a["last_xy"][0] + a["vel"][0]
            py = a["last_xy"][1] + a["vel"][1]
            best, bd = None, 1e9
            for j, c in enumerate(cands):
                if j in used:
                    continue
                d = ((c["x"] - px) ** 2 + (c["y"] - py) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, (j, c)
            if best is not None and bd < cfg.TRACK_MAX_JUMP_PX:
                j, c = best
                used.add(j)
                gap = f - a["last_frame"]
                if gap > 1:
                    x0, y0 = a["last_xy"]
                    for g in range(1, gap):
                        t = g / gap
                        a["track"]["points"][a["last_frame"] + g] = (
                            int(round(x0 + (c["x"] - x0) * t)),
                            int(round(y0 + (c["y"] - y0) * t)),
                        )
                        a["track"]["bridged"].append(a["last_frame"] + g)
                a["vel"] = (c["x"] - a["last_xy"][0], c["y"] - a["last_xy"][1])
                a["last_xy"] = (c["x"], c["y"])
                a["last_frame"] = f
                a["track"]["points"][f] = (c["x"], c["y"])
                still_active.append(a)
            else:
                still_active.append(a)  # keep waiting out its gap
        active = still_active
        for j, c in enumerate(cands):
            if j in used:
                continue
            tr = {"points": {f: (c["x"], c["y"])}, "bridged": []}
            tracks.append(tr)
            active.append(
                {"track": tr, "last_xy": (c["x"], c["y"]),
                 "vel": (0, 0), "last_frame": f}
            )
    return tracks


def merge_tracks(tracks):
    """Merge tracks split by a detection gap into one shot.

    The ball vanishes for a few frames where it merges with the backboard /
    rim, so a flight often arrives as two tracks with a 6-10 frame gap. Two
    tracks merge when the second starts within TRACK_MERGE_MAX_GAP frames
    of the first's end and its first point is within TRACK_MERGE_MAX_DIST
    of the first track's end extrapolated with its last velocity plus
    gravity (vy grows ~2 px/frame^2 in image units). Gap frames are filled
    by linear interpolation and recorded in `bridged` (drawn dotted, never
    as measured). Overlapping tracks are never merged; tracks with fewer
    than 3 points are junk and are dropped first.
    """
    ordered = sorted(
        (tr for tr in tracks if len(tr["points"]) >= 3),
        key=lambda tr: min(tr["points"]),
    )
    merged = []
    for tr in ordered:
        best = None
        for prev in merged:
            pf = sorted(prev["points"])
            tf = sorted(tr["points"])
            if tf[0] <= pf[-1]:
                continue  # overlapping tracks never merge
            gap = tf[0] - pf[-1]
            if gap > cfg.TRACK_MERGE_MAX_GAP:
                continue
            x0, y0 = prev["points"][pf[-1]]
            x1, y1 = prev["points"][pf[-2]] if len(pf) > 1 else (x0, y0)
            vx, vy = x0 - x1, y0 - y1
            px, py = x0, y0
            for _ in range(gap):
                vy += 2
                px, py = px + vx, py + vy
            sx, sy = tr["points"][tf[0]]
            dist = ((sx - px) ** 2 + (sy - py) ** 2) ** 0.5
            if dist <= cfg.TRACK_MERGE_MAX_DIST and (
                best is None or gap < best[0]
            ):
                best = (gap, prev, pf, tf, x0, y0, sx, sy)
        if best is not None:
            gap, prev, pf, tf, x0, y0, sx, sy = best
            for g in range(1, gap):
                t = g / gap
                prev["points"][pf[-1] + g] = (
                    int(round(x0 + (sx - x0) * t)),
                    int(round(y0 + (sy - y0) * t)),
                )
                prev["bridged"].append(pf[-1] + g)
            prev["points"].update(tr["points"])
            prev["bridged"].extend(tr["bridged"])
            continue
        merged.append(tr)
    return merged


def flight_tracks(tracks):
    """Keep only tracks that look like a shot in flight."""
    out = []
    for tr in tracks:
        pts = tr["points"]
        real = {f: p for f, p in pts.items() if f not in tr["bridged"]}
        if len(real) < cfg.TRACK_MIN_POINTS:
            continue
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        if max(xs) - min(xs) < cfg.TRACK_MIN_X_RANGE:
            continue
        if max(ys) - min(ys) < cfg.TRACK_MIN_Y_RANGE:
            continue
        # Must rise then fall: the minimum y (apex) is not at either end.
        frames = sorted(pts)
        apex_f = min(pts, key=lambda f: pts[f][1])
        if apex_f in (frames[0], frames[-1]):
            continue
        # A real shot rises into the upper half of the frame. A dribble arc
        # (apex y ~ 520) travels and rises too, but stays low: reject it.
        if pts[apex_f][1] > cfg.FLIGHT_MAX_APEX_Y:
            continue
        out.append(tr)
    out.sort(key=lambda tr: min(tr["points"]))
    return out


def rim_track(per_frame):
    """Per-frame rim centre.

    The rim at frame f is the MEDIAN of all rim-region candidates in a
    +-30 frame window: the real rim is there for most of the window, the
    ball only for a few frames, so the median locks onto the rim and the
    ball is an outlier. `measured` is True when a candidate sits within
    25 px of that median in frame f itself.
    Returns {frame: (x, y, measured: bool)}.
    """
    cands = {}
    for f, clist in enumerate(per_frame):
        got = []
        for c in clist:
            if not (cfg.RIM_REGION_X[0] <= c["x"] <= cfg.RIM_REGION_X[1]):
                continue
            if not (cfg.RIM_REGION_Y[0] <= c["y"] <= cfg.RIM_REGION_Y[1]):
                continue
            if not (cfg.RIM_MIN_AREA <= c["area"] <= cfg.RIM_MAX_AREA):
                continue
            got.append((c["x"], c["y"]))
        if got:
            cands[f] = got
    n = len(per_frame)
    out = {}
    for f in range(n):
        xs, ys = [], []
        for g in range(max(0, f - 30), min(n, f + 31)):
            for x, y in cands.get(g, []):
                xs.append(x)
                ys.append(y)
        if len(xs) < 5:
            continue
        mx, my = int(round(float(np.median(xs)))), int(round(float(np.median(ys))))
        measured = any(
            ((x - mx) ** 2 + (y - my) ** 2) ** 0.5 <= 25
            for x, y in cands.get(f, [])
        )
        out[f] = (mx, my, measured)
    return out
