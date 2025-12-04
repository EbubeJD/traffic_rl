# Global configuration for the project

TOWN = "Town10HD_Opt"
# Stable ID for the TL/lane we want to control for single-lane tests.
# Change this to point at a different ROI/ticks directory (e.g., tl_roadX_laneY_*).
CONTROL_TL_IDS = ["road5_lane-1_s10058", "road6_lane1_s569", "road18_lane5_s532"]
# Fallback group index if CONTROL_TL_IDS cannot be found (kept for compatibility).
GROUP_INDEX = 0
NUM_AUTOPILOT = 8              # number of NPC vehicles (0 to skip)
RES = (640, 360)               # (W, H)
CAM_FOV = 70
SAVE_ROOT = "outputs"
# Image saving is disabled by default to avoid huge disk usage at scale.
# Set to a positive integer (e.g., 10) if you need periodic snapshots.
SAVE_EVERY_N = 0                # 0 = do not save frames/visualizations

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

# Simulation timestep (sync mode)
DT = 0.1
