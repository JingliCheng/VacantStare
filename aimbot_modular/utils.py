
import math
from typing import Tuple
import dxcam
import re
from .config import CONFIG

def vertical_fov(horizontal_fov_deg: float, aspect_w: int, aspect_h: int) -> float:
    """Compute vertical FOV from horizontal FOV and aspect ratio."""
    rad = math.radians(horizontal_fov_deg)
    v = 2 * math.atan(math.tan(rad / 2) * (aspect_h / aspect_w))
    return math.degrees(v)

def clamp(val: float, mn: float, mx: float) -> float:
    return max(mn, min(mx, val))

def counts_from_pixels(dx_px: float, dy_px: float, scaler_w: float, scaler_h: float,
                       mon_w: int, mon_h: int, fov_h: float) -> Tuple[int, int]:
    """Convert pixel delta in cropped frame to raw mouse counts."""
    dx_screen = dx_px * scaler_w
    dy_screen = dy_px * scaler_h
    fov_v = vertical_fov(fov_h, mon_w, mon_h)
    theta_x = dx_screen * fov_h / mon_w
    theta_y = dy_screen * fov_v / mon_h
    counts_x = theta_x / (0.022 * CONFIG["SENSITIVITY"])
    counts_y = theta_y / (0.022 * CONFIG["SENSITIVITY"])
    return int(round(counts_x)), int(round(counts_y))

def dx_outputs_to_dict() -> dict[int, dict]:
    """
    Parse the two-line string from dxcam.output_info() into:
      {0: {'left': 0, 'top': 0, 'width': 2560, 'height': 1440},
       1: {'left': 2560, 'top': 0, 'width': 1920, 'height': 1080}}
    The 'left' of each monitor is placed right-to-left, like Windows default.
    """
    pattern = re.compile(r"Output\[(\d+)]\:.*?Res:\((\d+),\s*(\d+)\)")
    monitors, x_offset = {}, 0

    for line in dxcam.output_info().splitlines():
        m = pattern.search(line)
        if not m:
            continue
        idx, w, h = map(int, m.groups())
        monitors[idx] = {
            "left":   x_offset,
            "top":    0,
            "width":  w,
            "height": h
        }
        x_offset += w                         # simple horizontal layout
    return monitors