# basketball_local

Tracks a shooting session from a phone, local only: MediaPipe 2D pose for
the skeleton, OpenCV colour tracking for the basketball, per-frame rim
tracking for make/miss. No VLM Run key, no SAM.

This is the local counterpart to upstream `basketball/`, which needs the VLM
Run Gateway (ViTPose + SAM) and a phone that stays still. This demo runs on
a handheld, panning phone: the rim is tracked per frame while the backboard
is visible (median of rim-region candidates over a +-30 frame window, so the
ball passing through cannot drag it), and every make/miss verdict is checked
by eye against the frames around the rim.

The drawn ball trail is smoothed (sliding quadratic, window 5) so the arc
reads as a parabola; decisions and metrics always use the raw detected
points, and frames near the rim are drawn raw. Bridged (interpolated) trail
segments are dotted yellow and labelled simulated.

## Run it

1. Put the pose model at
   `~/workspace/goals/golf-ball-trajectory-overlay/hidden_files/models/pose_landmarker_full.task`
   (or set `BASKETBALL_POSE_MODEL`), same model as `golf_swing/`.
2. Put a clip in `data/input/`.
3. Run it:

   ```bash
   python main.py data/input/basketball_peng.mov
   ```

## Output

Each run writes to `data/output/<clip-stem>/`:

| file | what |
|---|---|
| `<clip>_splitscreen.mp4` | LEFT original \| RIGHT annotated, same size/fps/duration |
| `<clip>_annotated.mp4` | annotated side only |
| `metrics.json` | pose coverage, ball coverage, per-shot make/miss with crossing frames |
| `run.log` | console log |

The annotated side shows the skeleton (amber), the ball trail (solid orange
= measured, dotted yellow = bridged/simulated), the rim circle (red), and a
running counter "投篮 made/attempted".

## Honest limits

- 2D only, no 3D, no shot-form metrics beyond the skeleton.
- The camera moves, so the rim moves in the frame; make/miss is a rim-plane
  crossing rule plus a fall-through check, not a 3D through-the-hoop
  measurement. Crossing frames are often bridged (the ball merges with the
  red rim for 2-3 frames), which the metrics flag.
- Colour ball detection fails on occlusion, on the red rim/backboard, and
  on the red floor logo. MediaPipe also hallucinates a flat "person" on the
  floor logo when the player is out of frame; those frames are filtered by
  a bbox rule (upright, height > 150 px).
- Tested on one clip: Peng's gym session, 25.6 s, 720x1280, 30 fps, 3 shots,
  3/3 made.

## License

Apache-2.0 (same as the repo).
