
"""Global configuration for the CS2 aimbot project."""

CONFIG = {
    # Runtime
    "MON_IDX": 0,
    "LIST_MONITORS": False,
    "SHOW_DEBUG": True,

    # Model
    "MODEL_PATH": "sunxds_0.5.6.pt",
    "USE_TENSORRT": True,
    "MODEL_W": 640,
    "CONF_THRES": 0.28,
    "HEAD_CLASS_ID": 7,

    # Game ↔ Screen mapping
    "SENSITIVITY": 1.25,
    "FOV_HORIZONTAL": 90.0,
    "MOVE_SMOOTH_STEPS": 8,
    "MIN_BOX_SIZE": 24,
    "FIRE_ERROR_RADIUS_DEG": 0.3,

    # Stability knobs
    "FOV_PIX_RADIUS": 250,
    "MAX_COUNTS_STEP": 60,
    "TARGET_STICK_MS": 300,
}
