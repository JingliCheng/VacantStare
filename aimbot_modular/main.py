
"""Entry point for modular CS2 aimbot."""
import queue, threading, time
import dxcam
from .config import CONFIG
from .detector import start_detector, _grab_rect
from .controller import start_controller
from .utils import dx_outputs_to_dict

def run():
    outputs = dx_outputs_to_dict()
    if CONFIG["LIST_MONITORS"]:
        for idx, m in outputs.items():
            print(f"{idx}: {m['width']}x{m['height']}  "
                  f"@ ({m['left']},{m['top']})")
        return
    idx = CONFIG["MON_IDX"]
    mon = outputs[idx]

    print(f"[*] Capturing monitor {idx}: {mon['width']}×{mon['height']}")
    rect = _grab_rect(mon, CONFIG['MODEL_W'])

    box_q: "queue.Queue[list]" = queue.Queue(maxsize=2)
    stop_evt = threading.Event()

    threads = [
        threading.Thread(
            target=start_detector, args=(rect, box_q, stop_evt), daemon=True
        ),
        threading.Thread(
            target=start_controller,
            args=(box_q, stop_evt, mon['width'], mon['height']),
            daemon=True,
        ),
    ]
    for t in threads: t.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_evt.set()
    finally:
        print("Bye.")

if __name__ == "__main__": run()
