"""Shot segmentation and make/miss, from ball flight tracks + rim track.

A shot is one flight track (see ball.flight_tracks). Release is its first
frame, apex is its highest point. Make/miss is decided at the rim: the first
frame where the ball, falling, crosses the rim's y while its centre is within
RIM_RADIUS_PX of the rim centre. The verdict is then VERIFIED by the
fall-through: the ball merges with the red rim for 2-3 frames around the
rim, so those points are rim-blob junk, not the ball. Skipping them, a MAKE
keeps falling below the rim, inside the net area, for the next frames. A
bounce off the rim comes back up or flies away instead. Every verdict in
the v1 run was also checked by eye against the frames around the rim; the
frame numbers are recorded in the metrics so anyone can re-check them.
"""

from __future__ import annotations

import config as cfg


def analyse_shots(flights, rim):
    """Return a list of per-shot dicts, in time order."""
    shots = []
    for i, tr in enumerate(flights):
        pts = tr["points"]
        frames = sorted(pts)
        release = frames[0]
        apex_f = min(pts, key=lambda f: pts[f][1])
        end = frames[-1]
        # Rim crossing: first falling crossing of the rim plane.
        crossing = None
        for f in frames:
            if f not in rim:
                continue
            rx, ry, _measured = rim[f]
            x, y = pts[f]
            if f == frames[0]:
                continue
            prev = pts.get(f - 1)
            if prev is None:
                continue
            if prev[1] < ry <= y and abs(x - rx) <= cfg.RIM_RADIUS_PX:
                crossing = f
                break
        outcome = "unknown"
        crossing_rim = None
        if crossing is not None:
            rx, ry, rim_measured = rim[crossing]
            crossing_rim = {"frame": crossing, "rim": [rx, ry],
                            "rim_measured": rim_measured,
                            "ball": list(pts[crossing]),
                            "crossing_frame_bridged": crossing in tr["bridged"]}
            through = 0
            for g in range(crossing + 3, crossing + 13):
                if g in pts:
                    x, y = pts[g]
                    if y > ry + 30 and abs(x - rx) <= 70:
                        through += 1
            outcome = "make" if through >= 3 else "miss"
        else:
            # No crossing inside the rim radius: it was short / wide.
            outcome = "miss"
        real_points = len([f for f in pts if f not in tr["bridged"]])
        # Display: stop the trail shortly after the rim, not at the end of
        # the ball's bounce on the floor.
        trail_end = end
        if crossing is not None:
            trail_end = min(end, crossing + 15)
        shots.append(
            {
                "shot": i + 1,
                "release_frame": release,
                "apex_frame": int(apex_f),
                "apex_xy": list(pts[apex_f]),
                "end_frame": end,
                "trail_end_frame": trail_end,
                "n_ball_frames": len(pts),
                "n_measured_ball_frames": real_points,
                "n_bridged_frames": len(tr["bridged"]),
                "bridged_frames": sorted(tr["bridged"]),
                "crossing": crossing_rim,
                "outcome": outcome,
            }
        )
    return shots
