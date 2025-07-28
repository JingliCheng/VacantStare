"""
screen_capture_face.py
采集 2K 屏幕 (2560×1440) -> 缩放 -> YOLOv8‑Face 推理
实时显示 + 保存 MP4 + 输出人头框 CSV

本文件按照 *screen_capture_pose.py* 的代码结构改写，只保留“检测头部/人脸”功能，
去除所有关键点 (pose) 相关逻辑。
"""
import os
import time, csv, cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# ---------- 配置 ----------
MON_IDX = 1                         # mss 屏幕索引（0 = 全屏, 1 = 主屏，2 = 副屏...）
# MODEL_NAME = "yolov8x-face"         # 请先下载对应 pt 权重到当前目录
# MODEL_NAME = "sunxds_0.5.6"
# MODEL_NAME = "yolov10n"
MODEL_NAME = "best"
MODEL_W = MODEL_H = 960            # 推理分辨率 (正方形即可)
OUT_MP4 = "out_face.mp4"            # 输出视频文件名
OUT_CSV = "faces.csv"               # 输出 CSV 文件名
CONF_THRES = 0.28                    # 置信度阈值
USE_TENSORRT = False                # 是否导出 / 使用 TensorRT 引擎
# --------------------------------

pt_path = f"{MODEL_NAME}.pt"
engine_path = f"{MODEL_NAME}_{MODEL_W}.engine"

# === resize & letter‑box ====================================================

def letterbox_resize(image, target_size=(640, 640), color=(114, 114, 114)):
    ih, iw = image.shape[:2]
    w, h   = target_size
    scale  = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    resized   = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((h, w, 3), color, dtype=np.uint8)
    top  = (h - nh) // 2
    left = (w - nw) // 2
    new_image[top:top + nh, left:left + nw] = resized
    return new_image, scale, top, left

# === 模型加载 / TensorRT 导出 =================================================

if USE_TENSORRT:
    if not os.path.exists(engine_path):
        try:
            model = YOLO(pt_path)
        except Exception as e:
            raise FileNotFoundError(
                f"\n❌ 无法加载模型 `{pt_path}`，请确认权重已下载到本地。\n\n原始错误：{e}")
        export_res      = model.export(format="tensorrt", imgsz=MODEL_W, half=True, device=0)
        default_eng     = str(export_res)
        os.rename(default_eng, engine_path)
    model = YOLO(engine_path)
else:
    try:
        model = YOLO(pt_path)
    except Exception as e:
        raise FileNotFoundError(
            f"\n❌ 无法加载模型 `{pt_path}`，请手动下载到本地。\n\n原始错误：{e}")

# === 屏幕抓取 & 推理 =========================================================

with mss() as sct:
    mon = sct.monitors[MON_IDX]
    capture_rect = {
        "top":    mon["top"],
        "left":   mon["left"],
        "width":  mon["width"],
        "height": mon["height"],
    }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_MP4, fourcc, 30, (MODEL_W, MODEL_H))

    csv_file   = open(OUT_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "pid", "x1", "y1", "x2", "y2"])

    t0, fid = time.time(), 0
    print("⏯  Press 'q' in the video window to quit.")

    try:
        while True:
            # 1) 屏幕截图 (BGRA) -> BGR
            frame_bgr = np.asarray(sct.grab(capture_rect))[:, :, :3]

            # 2) Letter‑box 缩放到目标输入分辨率
            input_img, scale, pad_top, pad_left = letterbox_resize(frame_bgr, (MODEL_W, MODEL_H))

            # 3) 推理 (Ultralytics 接受 BGR 或 RGB; 为保持一致转 RGB)
            results = model(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB),
                             conf=CONF_THRES, verbose=False)

            output_frame = input_img.copy()
            # yolov8‑face 只有一类 (face/head)，直接遍历 boxes
            for pid, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                csv_writer.writerow([fid, pid, x1, y1, x2, y2])
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 128, 255), 2)

            # 4) 展示 & 保存
            cv2.imshow("YOLO Face (q to quit)", output_frame)
            writer.write(output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fid += 1

    except KeyboardInterrupt:
        print("❌ Interrupted")

    finally:
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"Done. Avg FPS ≈ {fid/(time.time()-t0):.1f}")
