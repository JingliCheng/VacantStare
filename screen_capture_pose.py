"""
screen_capture_pose.py
采集 2K 屏幕(2560×1440) -> 缩放 -> YOLOv11n-Pose 推理
实时显示 + 保存 MP4 + 输出关键点 CSV
"""
import os
import time, csv, cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# ---------- 配置 ----------
MON_IDX = 1
# MODEL_NAME = "yolo11m-pose"
MODEL_NAME = "yolov8x-face"
MODEL_W = MODEL_H = 1080
OUT_MP4 = "out_pose.mp4"
OUT_CSV = "keypoints.csv"
CONF_THRES = 0.4
USE_TENSORRT = False


pt_path = f"{MODEL_NAME}.pt"
engine_path = f"{MODEL_NAME}_{MODEL_W}.engine"
# --------------------------

def letterbox_resize(image, target_size=(640, 640), color=(114, 114, 114)):
    ih, iw = image.shape[:2]
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((h, w, 3), color, dtype=np.uint8)
    top = (h - nh) // 2
    left = (w - nw) // 2
    new_image[top:top + nh, left:left + nw] = resized
    return new_image, scale, top, left

if USE_TENSORRT:
    # 模型加载和导出 TensorRT
    if not os.path.exists(engine_path):
        try:
            model = YOLO(pt_path)
        except Exception as e:
            raise FileNotFoundError(
                f"\n❌ 无法加载模型 `{pt_path}`，"
                f"请确认是否手动下载到了本地。\n\n原始错误：{e}"
            )
        export_result = model.export(format="tensorrt", imgsz=MODEL_W, half=True, device=0)
        default_engine_path = str(export_result)
        os.rename(default_engine_path, engine_path)
    model = YOLO(engine_path)
else:
    try:
        model = YOLO(pt_path)
    except Exception as e:
        raise FileNotFoundError(
            f"\n❌ 无法从远程repo加载模型 `{pt_path}`，"
            f"请确认是否手动下载到了本地。\n\n原始错误：{e}"
        )

# 屏幕抓取
with mss() as sct:
    mon = sct.monitors[MON_IDX]
    capture_rect = {
        "top": mon["top"],
        "left": mon["left"],
        "width": mon["width"],
        "height": mon["height"],
    }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_MP4, fourcc, 30, (MODEL_W, MODEL_H))

    csv_file = open(OUT_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "pid"] + sum([[f"x{i}", f"y{i}"] for i in range(17)], []))

    t0, fid = time.time(), 0
    print("⏯  Press 'q' in the video window to quit.")

    try:
        while True:
            frame = np.asarray(sct.grab(capture_rect))[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            input_img, scale, pad_top, pad_left = letterbox_resize(frame, (MODEL_W, MODEL_H))
            results = model(input_img, conf=CONF_THRES, verbose=False)

            output_frame = input_img.copy()
            for person_id, kp in enumerate(results[0].keypoints.xy.cpu().numpy()):
                csv_writer.writerow([fid, person_id] + kp.reshape(-1).tolist())

                for x, y in kp:
                    cv2.circle(output_frame, (int(x), int(y)), 2, (0, 255, 0), -1)

                nose = kp[0]; l_sh, r_sh = kp[5], kp[6]
                sh_w = np.linalg.norm(l_sh - r_sh)
                r = int(sh_w * 0.5)
                nose_x, nose_y = int(nose[0]), int(nose[1])
                cv2.rectangle(output_frame, (nose_x - r, nose_y - r), (nose_x + r, nose_y + r), (0, 0, 255), 1)

                xs = [l_sh[0], r_sh[0], kp[11][0], kp[12][0]]
                ys = [l_sh[1], r_sh[1], kp[11][1], kp[12][1]]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            cv2.imshow("YOLO Pose (q to quit)", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

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
