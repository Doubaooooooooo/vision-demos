# Rock Climbing (local) + Computer Vision

Segments a boulder problem off the wall, follows the climber up it, and
reads the route back: which holds were used, in what order, and how long
the send took. Local version of the upstream `rock_climbing/` demo: no
VLM Run key, no SAM, no ViTPose. MediaPipe PoseLandmarker (2D pose) +
HSV purple-hold segmentation + eye-verified small holds.

## Run

```bash
python main.py data/input/rock_climbing_peng.mp4
```

Outputs in `data/output/<clip>/`: `<clip>_splitscreen.mp4` (left original,
right annotated), `metrics.json`, `run.log`. Pose landmarks are cached in
`data/cache/` keyed on the input file, so re-renders are fast.

## What it draws

- Orange skeleton (MediaPipe, 2D), green dots at the grip estimates
  (wrist + 45px up — the wrist is not the fingers).
- Purple hold circles, numbered bottom-to-top, lit once used.
- Red hip-centre trail (smoothed), running timer, hip height as % of
  mat -> top hold, and the hold sequence so far.

## Honest limits

- 2D only. Facing away from the camera, MediaPipe collapses the arms:
  wrist landmarks sit ~45px below the actual grip when the arms are
  raised, and up to ~50px off when the arms are occluded behind the
  torso (early climb). Every hold contact in `metrics.json` was checked
  by eye on the frame; the code's dwell rule alone is not trusted.
- HSV segmentation only finds the two big purple volumes + one mid hold:
  the small dark-purple holds are within ~5 HSV units of the grey wall,
  so they were defined by eye from the median wall image.
- Feet are not reported as hold contacts (smearing and dangling).
- Height is 2D hip height above the mat. No fall force, no grip strength.
