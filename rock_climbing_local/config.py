"""Settings for rock_climbing_local: single phone, bouldering, fun first.

Local, offline, no VLM Run key: MediaPipe PoseLandmarker (2D pose only)
+ HSV colour segmentation for the big purple holds + eye-verified small
holds. Follows the upstream rock_climbing/ pattern (holds light up in
order, sequence + time) without SAM / ViTPose.

Honest limits (see README.md):
- Pose is 2D only. Facing away from the camera, MediaPipe collapses the
  arms: wrist landmarks sit ~45px below the actual grip when the arms are
  raised (wrist != fingers), and up to ~50px off when the arms are occluded
  behind the torso (early climb). Grip points are therefore estimated as
  wrist + (0, -45) and every hold contact was checked by eye on the frame.
- The small dark-purple holds are almost the same HSV as the grey wall
  (within ~5 hue units), so HSV segmentation only finds the two big volumes
  + one mid hold. The four small holds below were defined by eye from the
  median wall image and are marked as such in the metrics.
- Hold contacts are dwell-based (>=0.5s within 25px) on the grip estimate.
  A foot smeared on the wall or dangling in the air is not a hold.
- Height is 2D: hip height above the mat as % of (mat -> top hold).
  No 3D, no fall force, no grip strength.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

INPUT = DATA_DIR / "input"
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Pose: MediaPipe PoseLandmarker (local, offline) ──────────────────────────
MODEL_PATH = (
    Path.home()
    / "workspace/goals/golf-ball-trajectory-overlay/hidden_files/models/pose_landmarker_full.task"
)
MODEL_ENV_VAR = "ROCK_CLIMB_POSE_MODEL"
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_STRIDE = 1
LANDMARK_MIN_PRESENCE = 0.5

# ── Holds ───────────────────────────────────────────────────────────────────
# (id, centre x, centre y, radius px, how it was found)
# Frame is 720x720. H1/H2/H6 found by HSV on the median wall image;
# H3/H4/H5/H7 defined by eye from the median wall image + grip frames.
HOLDS = [
    (1, 215, 418, 40, "hsv"),
    (2, 330, 425, 40, "hsv"),
    (3, 285, 368, 20, "eye"),
    (4, 256, 338, 18, "eye"),
    (5, 334, 328, 18, "eye"),
    (6, 292, 265, 20, "hsv"),
    (7, 310, 226, 22, "eye"),
]
# HSV ranges used for the purple segmentation test (OpenCV HSV).
HSV_PURPLE_LO = (125, 60, 40)
HSV_PURPLE_HI = (165, 255, 255)

# ── Contact detection ────────────────────────────────────────────────────────
# Grip estimate: wrist + (0, -GRIP_OFFSET_PX). Verified on zoomed crops:
# at the top the wrists sit ~45px below the fingers on the hold.
GRIP_OFFSET_PX = 45
# A hand is "on" a hold when its grip estimate stays within the hold radius
# + this margin for DWELL_FRAMES.
CONTACT_MARGIN_PX = 15
DWELL_FRAMES = 15          # 0.5s at 30fps
DWELL_MOVE_PX = 25         # max drift inside one dwell
# Feet: toe = foot-index landmark (31/32). Same dwell rule, but a foot is
# only reported on a hold when eye-verified (feet smear and dangle a lot).
FOOT_DWELL_FRAMES = 15

# ── Timing ───────────────────────────────────────────────────────────────────
MAT_Y = 640                # mat top edge, px (wall-floor junction ~y640)
TOP_HOLD_Y = 226           # H7 centre y, px
# Start: first frame both feet are off the mat (toe y < MAT_Y - 60).
START_TOE_CLEAR_PX = 60
# Top: first frame both grip estimates are on H7.
# Drop: first frame hip y rises >15px/frame after the top (free fall).
DROP_DY_PX = 15

# ── Render ───────────────────────────────────────────────────────────────────
EXPORT_HEIGHT = None
OUTPUT_CRF = 20
SKELETON_COLOR = (0, 200, 255)      # BGR, amber (same as golf v1)
SKELETON_THICKNESS = 2
JOINT_RADIUS = 3
JOINT_COLOR = (255, 255, 255)
HOLD_COLOR = (220, 80, 170)         # BGR, purple
HOLD_USED_COLOR = (220, 80, 170)
HOLD_UNUSED_COLOR = (90, 90, 90)
HIP_TRAIL_COLOR = (0, 0, 255)       # BGR, red centre trail
HIP_TRAIL_FRAMES = 90
PANEL_HEIGHT = 96
FONT_SCALE = 0.7
FONT_THICKNESS = 2

OUTPUT_DIR = DATA_DIR / "output"
