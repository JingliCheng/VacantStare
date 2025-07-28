import queue, threading, time
from aimbot_modular.config import CONFIG
from aimbot_modular.controller import start_controller, Target
from aimbot_modular import controller   # <<--- 用来打桩

# ---- 打桩 ----
controller.send_move  = lambda dx,dy: print(f"[move] {dx:+} {dy:+}")
controller.send_click = lambda: print("[click]")

# ---- 屏幕 & 缩放 ----
MON_W, MON_H = 1920, 1080
scaler_w = MON_W / CONFIG["MODEL_W"]
scaler_h = MON_H / CONFIG["MODEL_W"]

# ---- 启动 ----
q = queue.Queue(maxsize=2)
stop_evt = threading.Event()
threading.Thread(
    target=start_controller,
    args=(q, stop_evt, scaler_w, scaler_h, MON_W, MON_H),
    daemon=True,
).start()

# ---- 连续注入假目标 ----
for _ in range(30):
    q.put([Target(480, 480, 60, 60, 1.0)])
    time.sleep(0.05)

stop_evt.set()
