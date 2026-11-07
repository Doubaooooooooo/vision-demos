"""Settings for the basketball_local demo. Edit here, or override per run.

v1 is 2D pose + colour ball tracking only, for a phone-shot gym session.
No VLM Run key is needed: pose runs locally with MediaPipe PoseLandmarker,
the ball is found by colour (orange HSV + dark-red channel rule).

Honest limits (see README.md):
- The camera in the v1 test clip MOVES (handheld, pans to follow the ball),
  so the hoop is NOT fixed in the frame. The rim is tracked per frame when
  the backboard is visible and interpolated across short gaps; make/miss is
  decided from the ball crossing the rim plane in the frames around the rim,
  and every verdict was checked by eye against those frames.
- Ball detection is colour only. It fails when the ball is occluded by the
  player, merges with the red rim/backboard padding, or is a small dark blob
  against the ceiling lights. Gaps inside a flight are bridged by
  interpolation and drawn DOTTED (simulated), never as measured.
- 2D only, no 3D, no shot-form metrics beyond the skeleton.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT = DATA_DIR / "input"   # a clip, or a folder of clips
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Pose: MediaPipe PoseLandmarker (local, offline) ──────────────────────────
MODEL_PATH = (
    Path.home()
    / "workspace/goals/golf-ball-trajectory-overlay/hidden_files/models/pose_landmarker_full.task"
)
MODEL_ENV_VAR = "BASKETBALL_POSE_MODEL"
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_STRIDE = 1
LANDMARK_MIN_PRESENCE = 0.5
# A real player here is upright: bbox height > 150 px and taller than wide.
# MediaPipe hallucinates a flat "person" on the red floor logo when the
# player is out of frame; those frames are dropped (see main.py).
POSE_MIN_BBOX_HEIGHT = 150

# ── Ball detection (colour) ──────────────────────────────────────────────────
# Orange ball in bright light: HSV hue 3-26. Dark-red ball against the
# ceiling: R clearly above G and B, R in 55..210.
HSV_LO = (3, 110, 70)
HSV_HI = (26, 255, 255)
RED_R_MIN = 55
RED_R_MAX = 210
RED_R_MINUS_G = 12
RED_R_MINUS_B = 8
BALL_MIN_AREA = 60
BALL_MAX_AREA = 3000
BALL_MIN_CIRCULARITY = 0.45
BALL_MAX_CANDIDATES = 4
# Candidates with y above this are flight/gather candidates; the red floor
# logo lives at y ~ 680 and is excluded from flight tracking.
FLIGHT_MAX_Y = 640
# A candidate within this of the rim centre is the rim blob, not the ball:
# for 2-3 frames around the rim the ball and the red rim merge into one
# blob. Those frames become a bridged gap (drawn dotted).
RIM_EXCLUDE_PX = 30

# ── Ball tracking ────────────────────────────────────────────────────────────
TRACK_MAX_JUMP_PX = 55       # per frame, from the velocity-predicted position
TRACK_MAX_GAP = 4            # frames; longer gaps split the track
TRACK_MERGE_MAX_GAP = 15     # frames; two tracks this close are one shot
TRACK_MERGE_MAX_DIST = 150   # px; predicted end to next track start
TRACK_MIN_POINTS = 12        # a flight track must have at least this many
TRACK_MIN_X_RANGE = 80       # px, ball must travel toward the hoop
TRACK_MIN_Y_RANGE = 120      # px, ball must rise and fall
FLIGHT_MAX_APEX_Y = 400      # px; a real shot rises into the upper half.
                             # A dribble arc (apex y ~ 520) is not a shot.
BRIDGE_MAX_GAP = 4           # frames bridged by interpolation (drawn dotted)

# ── Hoop / rim ───────────────────────────────────────────────────────────────
# Rim blob: red, in the right-hand part of the frame, y 430..570.
RIM_REGION_X = (560, 720)
RIM_REGION_Y = (430, 570)
RIM_MIN_AREA = 200
RIM_MAX_AREA = 1000
RIM_RADIUS_PX = 35           # ball centre within this of rim centre = through
RIM_FALL_CHECK_FRAMES = 6    # (kept for reference; see analysis.py)

# ── Render ───────────────────────────────────────────────────────────────────
OUTPUT_CRF = 20
SKELETON_COLOR = (0, 200, 255)      # BGR, amber-ish (same as golf v1)
SKELETON_THICKNESS = 2
JOINT_RADIUS = 3
JOINT_COLOR = (255, 255, 255)       # BGR
BALL_COLOR = (0, 140, 255)          # BGR, orange: measured trail + ball
BALL_RADIUS = 8
SIM_COLOR = (0, 255, 255)           # BGR, yellow: bridged (simulated) trail
RIM_COLOR = (0, 0, 255)             # BGR, red: rim circle
PANEL_HEIGHT = 96
FONT_SCALE = 0.7
FONT_THICKNESS = 2
# Trail smoothing: sliding quadratic, window 5, on measured runs only.
SMOOTH_WINDOW = 5
# Frames within this of the rim crossing are drawn raw, never smoothed.
SMOOTH_PROTECT_FRAMES = 3
