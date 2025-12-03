# Global configuration for the project

TOWN = "Town10HD_Opt"
# Stable ID for the TL/lane we want to control for single-lane tests.
# Change this to point at a different ROI/ticks directory (e.g., tl_roadX_laneY_*).
CONTROL_TL_IDS = ["road5_lane-1_s10058"]
# Fallback group index if CONTROL_TL_IDS cannot be found (kept for compatibility).
GROUP_INDEX = 0
NUM_AUTOPILOT = 40              # number of NPC vehicles (0 to skip)
RES = (1280, 720)               # (W, H)
CAM_FOV = 70
SAVE_ROOT = "outputs"
SAVE_EVERY_N = 10               # save every Nth frame
DT = 0.05                       # world.fixed_delta_seconds (sync mode)

# Detection / Tracking
YOLO_MODEL_PATH = "models/yolo11n.pt"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle"}

IOU_MATCH_THR = 0.3
TRACK_MAX_AGE = 20              # ticks to keep a lost track
STOP_SPEED_PX_THR = 1.3         # px/tick; below = "stopped"
STRIPE_CONFIRM_FRAMES = 2       # frames in stripe before counting crossing

# Smoothing
EMA_ALPHA_QUEUE = 0.3
EMA_ALPHA_ARRIVAL = 0.2
EMA_ALPHA_DISCH = 0.2
