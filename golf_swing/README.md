# golf_swing

Single phone, single camera, down-the-line, for beginners, fun first.
Point a phone at a golfer from behind (down the line), record one swing,
and this draws the swing back: skeleton overlay, swing phases, and tempo.

## What v1 does

- 2D pose on every frame, locally, offline — MediaPipe PoseLandmarker.
  No VLM Run key, no network, no API key.
- Segments one swing into GolfDB's 8 events (address, toe-up,
  mid-backswing, top, mid-downswing, impact, mid-follow-through, finish)
  from the hands' height curve.
- Tempo: backswing time vs downswing time, as a ratio.
- 2D spine angle at address / top / impact, plus a fading wrist trail.

## What v1 does NOT do (read this before trusting a number)

- **No 3D.** Everything is 2D. The spine angle is a projected angle in the
  image plane, valid only on a down-the-line view. It is not a true 3D angle.
- **Phases are a heuristic**, not a learned event detector. They are checked
  visually per clip; on a new clip they can be off.
- **The club is not tracked.** The orange trail follows the hands, not the
  clubhead.
- **Tempo seconds are video time.** On slow-motion source (e.g. 250 fps
  played back at 30 fps) the seconds are stretched; only the
  backswing:downswing ratio is comparable across clips.

## Output

Per clip, in `data/output/<clip>/`:

| file | what |
|---|---|
| `<clip>_splitscreen_h264.mp4` | **The deliverable.** LEFT original \| RIGHT annotated, same size, fps and duration, frame-aligned. The left side has no annotations. |
| `<clip>_annotated_h264.mp4` | Single-view annotated video (intermediate). |
| `metrics.json` | Phases, tempo, spine angles, pose coverage, caveats. |
| `run.log` | Console log for the run. |

## Run it

1. Get the pose model (once, ~9 MB):

   ```bash
   mkdir -p ~/.cache/golf_swing
   curl -L -o ~/.cache/golf_swing/pose_landmarker_full.task \
     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
   export GOLF_SWING_POSE_MODEL=~/.cache/golf_swing/pose_landmarker_full.task
   ```

   Or put the file next to the project; `main.py` also looks there.

2. Create the environment:

   ```bash
   python3 -m venv .venv-golf-swing
   .venv-golf-swing/bin/pip install -r environment.yml
   ```

   ffmpeg is a system dependency and is not installable here.

3. Put a clip in `data/input/`: down-the-line, the phone set down and
   still, whole body and club in shot, 120 fps slow motion if you can.

4. Run it:

   ```bash
   .venv-golf-swing/bin/python main.py data/input/your_swing.mp4
   ```

Note for headless Linux: MediaPipe's native library needs `libEGL.so.1`.
If it is missing, either install `libegl1` or point the loader at a copy,
e.g. `LD_LIBRARY_PATH=/path/with/libEGL.so.1`. `src/pose.py` also tries to
pre-load a libEGL from `/opt/meta-chromium/libEGL.so` if present.

## How it works

`src/pose.py` runs MediaPipe PoseLandmarker in VIDEO mode and returns
per-frame 2D landmarks. `src/analysis.py` builds the hand-height signal
(hands above the hips, in torso lengths), smooths it, and reads the 8
events off it: the top is the hands' highest point, impact is the first
local minimum after the top, the rest are crossings of fractions of the
address→top range. `src/render.py` draws the skeleton, the wrist trail,
the phase / tempo / spine text, and a timeline bar with the 8 event ticks.

## License

[Apache-2.0](../LICENSE).
