
"""Controller thread: consumes targets, decides mouse movement."""
import queue, threading, math, random, time
from typing import List, Optional

import win32api

from .config import CONFIG
from .utils import counts_from_pixels, clamp
from .mouse import send_move, send_click
from .detector import Target

def start_controller(
    in_q: "queue.Queue[List[Target]]",
    stop_evt: threading.Event,
    mon_w: int,
    mon_h: int,
):
    enabled = True
    toggle_key = 0x77  # F8
    esc_key = 0x1B
    last_toggle = False
    crop_center = CONFIG["MODEL_W"] // 2

    last_target: Optional[Target] = None
    last_seen_ms = 0.0

    while not stop_evt.is_set():
        cur = bool(win32api.GetAsyncKeyState(toggle_key))
        if cur and not last_toggle:
            enabled = not enabled
            print(f"[Controller] {'ON' if enabled else 'OFF'}")
        last_toggle = cur
        if win32api.GetAsyncKeyState(esc_key):
            stop_evt.set()
            break

        try:
            t_wait0 = time.perf_counter()
            targets = in_q.get(timeout=0.01)
            t_wait1 = time.perf_counter()
            wait_ms = (t_wait1 - t_wait0) * 1000
        except queue.Empty:
            continue
        if not enabled or not targets:
            continue

        now = time.time() * 1000
        if last_target and now - last_seen_ms < CONFIG["TARGET_STICK_MS"]:
            chosen = min(
                targets,
                key=lambda t: (t.cx - last_target.cx) ** 2 + (t.cy - last_target.cy) ** 2,
            )
        else:
            chosen = min(
                targets,
                key=lambda t: (t.cx - crop_center) ** 2 + (t.cy - crop_center) ** 2,
            )

        dx_crop = chosen.cx - crop_center
        dy_crop = chosen.cy - crop_center
        dist_px = math.hypot(dx_crop, dy_crop)
        if dist_px > CONFIG["FOV_PIX_RADIUS"]:
            continue

        mx, my = counts_from_pixels(
            dx_crop, dy_crop, 1, 1, mon_w, mon_h, CONFIG["FOV_HORIZONTAL"]
        )
        t_aim0 = time.perf_counter()

        mx = clamp(
            mx,
            -CONFIG["MAX_COUNTS_STEP"] * CONFIG["MOVE_SMOOTH_STEPS"],
            CONFIG["MAX_COUNTS_STEP"] * CONFIG["MOVE_SMOOTH_STEPS"],
        )
        my = clamp(
            my,
            -CONFIG["MAX_COUNTS_STEP"] * CONFIG["MOVE_SMOOTH_STEPS"],
            CONFIG["MAX_COUNTS_STEP"] * CONFIG["MOVE_SMOOTH_STEPS"],
        )

        steps = CONFIG["MOVE_SMOOTH_STEPS"]
        for _ in range(steps):
            dxs = int(clamp(mx / steps, -CONFIG["MAX_COUNTS_STEP"], CONFIG["MAX_COUNTS_STEP"]))
            dys = int(clamp(my / steps, -CONFIG["MAX_COUNTS_STEP"], CONFIG["MAX_COUNTS_STEP"]))
            send_move(dxs + random.randint(-1, 1), dys + random.randint(-1, 1))
            time.sleep(0.0005)

        t_aim1 = time.perf_counter()
        aim_ms = (t_aim1 - t_aim0) * 1000

        if math.hypot(mx, my) * 0.022 * CONFIG["SENSITIVITY"] < CONFIG["FIRE_ERROR_RADIUS_DEG"]:
            send_click()
        print('[CTL] wait={:.2f}ms aim={:.2f}ms total={:.2f}ms'.format(wait_ms, aim_ms, wait_ms+aim_ms))

        last_target, last_seen_ms = chosen, now
