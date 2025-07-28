
"""Detector thread: grabs screen, runs YOLO, outputs head targets."""
import queue, threading
from pathlib import Path
from typing import List
import time, collections

import numpy as np
import cv2
import dxcam
from ultralytics import YOLO

from .config import CONFIG
from .utils import clamp

class Target:
    __slots__ = ("cx", "cy", "w", "h", "conf")
    def __init__(self, cx: int, cy: int, w: int, h: int, conf: float):
        self.cx, self.cy, self.w, self.h, self.conf = cx, cy, w, h, conf

def _grab_rect(mon: dict, size: int):
    cx = mon["left"] + mon["width"] // 2
    cy = mon["top"] + mon["height"] // 2
    half = size // 2
    return {"left": cx - half, "top": cy - half, "width": size, "height": size}

def start_detector(rect: dict, out_q: "queue.Queue[List[Target]]", stop_evt: threading.Event):
    region = (rect["left"], rect["top"],
           rect["left"] + rect["width"],
           rect["top"]  + rect["height"])
    model_path = Path(CONFIG["MODEL_PATH"])
    print('model_path.with_suffix(".engine").exists():', model_path.with_suffix(".engine").exists())
    if CONFIG["USE_TENSORRT"] and model_path.with_suffix(".engine").exists():
        model = YOLO(str(model_path.with_suffix(".engine")))
    else:
        model = YOLO(str(model_path))
    cam = dxcam.create(
        output_idx=CONFIG["MON_IDX"],    # 显示器索引，跟原来 rect 属于同一台
        output_color="BGR"               # 直接给 YOLO 用的 BGR
    )

    head_id = CONFIG["HEAD_CLASS_ID"]
    show_debug = CONFIG["SHOW_DEBUG"]
    if show_debug:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("debug", CONFIG["MODEL_W"], CONFIG["MODEL_W"])

    # Stat = collections.namedtuple('Stat','grab infer total')
    while not stop_evt.is_set():
        t0 = time.perf_counter()
        # frame_bgr = np.asarray(sct.grab(rect))[:, :, :3]
        frame_bgr = cam.grab(region=region)  # numpy.ndarray, shape=(H,W,3), dtype=uint8
        if frame_bgr is None:          # ← 抓屏失败就跳过这一帧
            continue
        t_grab = time.perf_counter()
        det = model(frame_bgr, conf=CONFIG["CONF_THRES"], verbose=False)[0]
        t_infer = time.perf_counter()
        print('[DET] grab={:.3f}ms infer={:.3f}ms'.format(
            (t_grab-t0)*1000, (t_infer-t_grab)*1000)
        )
        targets: List[Target] = []
        if det.boxes is not None and det.boxes.xyxy.shape[0]:
            for xyxy, conf, cls_idx in zip(
                det.boxes.xyxy.cpu().numpy(),
                det.boxes.conf.cpu().numpy(),
                det.boxes.cls.cpu().numpy().astype(int),
            ):
                if cls_idx != head_id:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                if w < CONFIG["MIN_BOX_SIZE"] or h < CONFIG["MIN_BOX_SIZE"]:
                    continue
                targets.append(Target((x1 + x2) // 2, (y1 + y2) // 2, w, h, float(conf)))

        try:
            out_q.put_nowait(targets)
        except queue.Full:
            pass

        if show_debug:
            dbg = np.ascontiguousarray(frame_bgr.copy())
            cc = CONFIG["MODEL_W"] // 2
            cv2.drawMarker(dbg, (cc, cc), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            for t in targets:
                x1 = int(t.cx - t.w // 2)
                y1 = int(t.cy - t.h // 2)
                cv2.rectangle(dbg, (x1, y1), (x1 + t.w, y1 + t.h), (0, 255, 0), 1)
                cv2.putText(
                    dbg,
                    f"{t.conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("debug", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                stop_evt.set()
                break
    if show_debug:
        cv2.destroyAllWindows()
