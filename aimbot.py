
#!/usr/bin/env python
"""
aimbot.py – CS2 auto-aim prototype (multi-monitor, head-only, stability fixes)

v0.6.1  ▸  Debug window + full code dump
  • SHOW_DEBUG window with head boxes
  • Stability knobs and target persistence

Hotkeys
  F8  toggle aim assist ON/OFF
  ESC quit script (window or game)

Disclaimer: Using aim assist in online CS2 violates the EULA and can lead
to VAC bans. Use only for learning or on private servers.
"""
from __future__ import annotations
import math, queue, random, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np, cv2, win32api, win32con
from mss import mss
from ultralytics import YOLO

CONFIG = {
    # runtime
    "MON_IDX": 1,
    "LIST_MONITORS": False,
    "SHOW_DEBUG": True,

    # model
    "MODEL_PATH": "sunxds_0.5.6.pt",
    "USE_TENSORRT": False,
    "MODEL_W": 640,
    "CONF_THRES": 0.28,
    "HEAD_CLASS_ID": 7,

    # mapping
    "SENSITIVITY": 2.0,
    "FOV_HORIZONTAL": 90.0,
    "MOVE_SMOOTH_STEPS": 8,
    "MIN_BOX_SIZE": 24,
    "FIRE_ERROR_RADIUS_DEG": 0.3,

    # stability
    "FOV_PIX_RADIUS": 250,
    "MAX_COUNTS_STEP": 60,
    "TARGET_STICK_MS": 300,
}

@dataclass
class Target:
    cx:int; cy:int; w:int; h:int; conf:float

def grab_rect_for_center(mon:dict, size:int):
    cx = mon['left'] + mon['width']//2
    cy = mon['top'] + mon['height']//2
    half = size//2
    return {'left': cx-half,'top': cy-half,'width': size,'height': size}

def vertical_fov(h_deg:float, w:int, h:int):
    return math.degrees(2*math.atan(math.tan(math.radians(h_deg)/2)*(h/w)))

def clamp(v, mn, mx): return max(mn,min(mx,v))

def crop_px_to_mouse_counts(dx,dy,scw,sch,mw,mh,fov_h):
    dx_s,dy_s = dx*scw,dy*sch
    theta_x = dx_s * fov_h / mw
    theta_y = dy_s * vertical_fov(fov_h,mw,mh) / mh
    return int(round(theta_x/(0.022*CONFIG['SENSITIVITY']))), int(round(theta_y/(0.022*CONFIG['SENSITIVITY'])))

def send_mouse(dx,dy,click=False):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,dx,dy,0,0)
    if click:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)

def yolo_worker(box_q, stop_evt, rect):
    mp = Path(CONFIG['MODEL_PATH'])
    model = YOLO(str(mp.with_suffix('.engine'))) if CONFIG['USE_TENSORRT'] and mp.with_suffix('.engine').exists() else YOLO(str(mp))
    hid = CONFIG['HEAD_CLASS_ID']; debug = CONFIG['SHOW_DEBUG']
    if debug: cv2.namedWindow('debug',cv2.WINDOW_NORMAL); cv2.resizeWindow('debug',640,640)
    with mss() as sct:
        while not stop_evt.is_set():
            frame = np.asarray(sct.grab(rect))[:,:,:3]
            det = model(frame,conf=CONFIG['CONF_THRES'],verbose=False)[0]
            targets=[]
            if det.boxes is not None and det.boxes.xyxy.shape[0]:
                for (xyxy,conf,clsid) in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.conf.cpu().numpy(), det.boxes.cls.cpu().numpy().astype(int)):
                    if clsid!=hid: continue
                    x1,y1,x2,y2=map(int,xyxy); w,h=x2-x1,y2-y1
                    if w<CONFIG['MIN_BOX_SIZE'] or h<CONFIG['MIN_BOX_SIZE']: continue
                    targets.append(Target((x1+x2)//2,(y1+y2)//2,w,h,float(conf)))
            if debug:
                dbg = np.ascontiguousarray(frame.copy())
                cc = CONFIG['MODEL_W']//2
                cv2.drawMarker(dbg,(cc,cc),(0,0,255),cv2.MARKER_CROSS,12,1)
                for t in targets:
                    x1=int(t.cx-t.w//2);y1=int(t.cy-t.h//2)
                    cv2.rectangle(dbg,(x1,y1),(x1+t.w,y1+t.h),(0,255,0),1)
                    cv2.putText(dbg,f"{t.conf:.2f}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
                cv2.imshow('debug',dbg)
                if cv2.waitKey(1)&0xFF==27: stop_evt.set(); break
            try: box_q.put_nowait(targets)
            except queue.Full: pass
    if debug: cv2.destroyAllWindows()

def controller_worker(box_q, stop_evt, scw,sch,mw,mh):
    enabled=True; toggle=0x77; esc=0x1B; last=False; cc=CONFIG['MODEL_W']//2
    last_t=None; last_time=0
    while not stop_evt.is_set():
        cur=bool(win32api.GetAsyncKeyState(toggle))
        if cur and not last: enabled=not enabled; print(f"[Controller] {'ON' if enabled else 'OFF'}")
        last=cur
        if win32api.GetAsyncKeyState(esc): stop_evt.set(); break
        try: tgts=box_q.get(timeout=0.01)
        except queue.Empty: continue
        if not enabled or not tgts: continue
        now=time.time()*1000
        if last_t and now-last_time<CONFIG['TARGET_STICK_MS']:
            chosen=min(tgts,key=lambda t:(t.cx-last_t.cx)**2+(t.cy-last_t.cy)**2)
        else:
            chosen=min(tgts,key=lambda t:(t.cx-cc)**2+(t.cy-cc)**2)
        dx,dy=chosen.cx-cc, chosen.cy-cc
        if math.hypot(dx*scw,dy*sch)>CONFIG['FOV_PIX_RADIUS']: continue
        mx,my=crop_px_to_mouse_counts(dx,dy,scw,sch,mw,mh,CONFIG['FOV_HORIZONTAL'])
        mx=clamp(mx,-CONFIG['MAX_COUNTS_STEP']*CONFIG['MOVE_SMOOTH_STEPS'],CONFIG['MAX_COUNTS_STEP']*CONFIG['MOVE_SMOOTH_STEPS'])
        my=clamp(my,-CONFIG['MAX_COUNTS_STEP']*CONFIG['MOVE_SMOOTH_STEPS'],CONFIG['MAX_COUNTS_STEP']*CONFIG['MOVE_SMOOTH_STEPS'])
        steps=CONFIG['MOVE_SMOOTH_STEPS']
        for _ in range(steps):
            dxs=int(clamp(mx/steps,-CONFIG['MAX_COUNTS_STEP'],CONFIG['MAX_COUNTS_STEP']))
            dys=int(clamp(my/steps,-CONFIG['MAX_COUNTS_STEP'],CONFIG['MAX_COUNTS_STEP']))
            send_mouse(dxs+random.randint(-1,1),dys+random.randint(-1,1)); time.sleep(0.001)
        if math.hypot(mx,my)*0.022*CONFIG['SENSITIVITY']<CONFIG['FIRE_ERROR_RADIUS_DEG']: send_mouse(0,0,click=True)
        last_t, last_time = chosen, now

def main():
    with mss() as sct:
        if CONFIG['LIST_MONITORS']:
            for i,m in enumerate(sct.monitors):
                if i==0: continue
                print(f"{i}: {m['width']}x{m['height']} @({m['left']},{m['top']})")
            return
        idx=CONFIG['MON_IDX']
        if not 1<=idx<len(sct.monitors): raise SystemExit('Invalid MON_IDX')
        mon=sct.monitors[idx]
    print(f"[*] Capturing monitor {idx}: {mon['width']}×{mon['height']}")
    scw,sch=mon['width']/CONFIG['MODEL_W'], mon['height']/CONFIG['MODEL_W']
    rect=grab_rect_for_center(mon,CONFIG['MODEL_W'])
    box_q=queue.Queue(maxsize=2); stop_evt=threading.Event()
    threading.Thread(target=yolo_worker,args=(box_q,stop_evt,rect),daemon=True).start()
    threading.Thread(target=controller_worker,args=(box_q,stop_evt,scw,sch,mon['width'],mon['height']),daemon=True).start()
    try:
        while not stop_evt.is_set(): time.sleep(0.1)
    except KeyboardInterrupt:
        stop_evt.set()
    print('Bye.')
if __name__=='__main__': main()
