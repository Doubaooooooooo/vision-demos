"""Settings for the golf_swing demo. Edit here, or override per run with flags.

v1 is 2D pose only, down-the-line (DTL) only, for beginners, fun first.
No VLM Run key is needed: pose runs locally with MediaPipe PoseLandmarker.

Honest limits (see README.md):
- Angles are 2D projected angles in the image plane, NOT true 3D angles.
- Swing phases are a heuristic read off the hands' height curve, not a
  learned event detector. They are checked visually per clip, not guaranteed.
- The club is NOT tracked in v1. The wrist trail is a proxy for the hands,
  not the clubhead.
- Tempo in seconds is video time. On slow-motion source (e.g. 250 fps played
  back at 30 fps) the seconds are stretched; only the backswing:downswing
  RATIO is comparable across clips.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT = DATA_DIR / "input"   # a clip, or a folder of clips
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Pose: MediaPipe PoseLandmarker (local, offline) ──────────────────────────
# Model file is downloaded once by the user; see README.md. It is NOT in git.
MODEL_PATH = (
    Path.home()
    / "workspace/goals/golf-ball-trajectory-overlay/hidden_files/models/pose_landmarker_full.task"
)
# Fallbacks: a model file next to the project, or one named in the env var.
MODEL_ENV_VAR = "GOLF_SWING_POSE_MODEL"
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
# Run pose on every Nth frame and interpolate in between (1 = every frame).
POSE_STRIDE = 1
# Landmarks with presence below this are treated as missing for that frame.
LANDMARK_MIN_PRESENCE = 0.5

# ── Signal ───────────────────────────────────────────────────────────────────
# Hands: midpoint of the two wrists, height in the frame, normalised by torso
# length (shoulder-mid to hip-mid, median over the clip) so the signal does
# not depend on how far the camera was.
SMOOTH_WINDOW = 5          # frames, centred moving average on the hand signal
SIGNAL_MIN_PROMINENCE = 0.15  # of the address->top hand-height range

# ── Phases (GolfDB's 8 events) ───────────────────────────────────────────────
# Fractions of the address->top hand-height range used for the intermediate
# events. Toe-up / mid-backswing are crossings on the way up, mid-downswing
# is a crossing on the way down.
TOE_UP_FRACTION = 0.25
MID_BACKSWING_FRACTION = 0.55
MID_DOWNSWING_FRACTION = 0.55
# Impact: first local minimum of hand height after the top, within this window.
IMPACT_SEARCH_SECONDS = 2.0
# Finish: the follow-through peak (max hand height after impact).
# Address: the first frame whose hand height is within this of the clip's
# early median, before the backswing starts.
ADDRESS_STABLE_SECONDS = 0.5

# ── Spine angle (2D, image plane) ────────────────────────────────────────────
# Angle of the shoulder-mid -> hip-mid line from vertical, in degrees.
# Positive = leaning toward the target (to the right in a DTL view of a
# right-handed golfer). DTL only; on any other view this number is meaningless.
SPINE_AT = ("address", "top", "impact")

# ── Render ───────────────────────────────────────────────────────────────────
EXPORT_HEIGHT = None       # None = source height; otherwise scale output to this
OUTPUT_CRF = 20
# Skeleton: draw these connections only (a readable subset of the 33 points).
SKELETON_COLOR = (0, 200, 255)      # BGR, amber-ish
SKELETON_THICKNESS = 2
JOINT_RADIUS = 3
JOINT_COLOR = (255, 255, 255)       # BGR
# Wrist (hands) trail: fading line of the last N frames.
TRAIL_SECONDS = 1.5
TRAIL_COLOR = (0, 140, 255)         # BGR, orange
PANEL_HEIGHT = 96                   # px, text strip at the top of the frame
TIMELINE_HEIGHT = 18                # px, phase timeline bar at the bottom
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
