# train_planA.py
from ultralytics import YOLO
from pathlib import Path

DATA_YAML   = r"C:/Users/fish/git_project/VacantStare/datasets/merged_yolo/cs.yaml"
MODEL_YAML  = r"scripts/bgf_yolov10n.yaml"
IMG_SIZE    = 640
BATCH       = 16
WORKERS     = 8          # ≥1 就会触发多进程

SKIP_STAGE1 = True
STAGE1_DIR = Path("runs/planA_stage1")
BEST1      = STAGE1_DIR / "weights" / "best.pt"

SKIP_STAGE2 = True
RESUME_STAGE2 = False
STAGE2_DIR = Path("runs/planA_stage2")           # ← run 目录
LAST2      = STAGE2_DIR / "weights" / "last.pt"  # 自动生成
BEST2      = STAGE2_DIR / "weights" / "best.pt"

STAGE3_DIR = "runs/planA_stage3"

def stage1():
    model = YOLO(MODEL_YAML)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=30,
        batch=BATCH,
        workers=WORKERS,
        freeze=15,
        cos_lr=True, lr0=0.003, lrf=0.12,
        mosaic=0.8,
        copy_paste=0.4,
        multi_scale=True,
        project="runs",
        name="planA_stage1",
        save=True,          # 默认就是 True
        save_period=5,      # ← 每x个 epoch 都保存  epoch001.pt、epoch002.pt …
        exist_ok=True
    )
    return Path(model.trainer.best)

def stage2(best_ckpt):
    if RESUME_STAGE2:
        print("Resume Stage 2 →", LAST2)
        model = YOLO(LAST2)          # 给目录即可
        model.train(resume=True, epochs=80)    # 继续到 80
    else:
        model = YOLO(best_ckpt)
        model.train(
            data=DATA_YAML,
            imgsz=IMG_SIZE,
            epochs=80,                  # 继续到 80
            batch=BATCH,
            workers=WORKERS,
            freeze=0,
            cos_lr=True, lr0=0.0006, lrf=0.05,
            mosaic=0.8,
            copy_paste=0.4,
            multi_scale=True,
            project="runs",
            name="planA_stage2",
            save=True,          # 默认就是 True
            save_period=5,      # ← 每x个 epoch 都保存  epoch001.pt、epoch002.pt …
            exist_ok=True
        )
    return Path(model.trainer.best)

def stage3(best_ckpt):
    model = YOLO(best_ckpt)  # 仅加载权重（无优化器），新的 run
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=160,
        batch=BATCH,
        workers=WORKERS,
        # ---- LR & Scheduler ----
        cos_lr=True,
        lr0=1e-3,
        lrf=0.05,             # 60 epoch 后降到 5e-5
        # ---- 增强策略 ----
        mosaic=0.3,
        copy_paste=0.3,
        multi_scale=True,    # 为显存+稳定关闭
        hsv_s=0.6, hsv_v=0.3, # 适当减弱颜色扰动
        # ---- 训练稳健性 ----
        freeze=0,             # 不冻结
        # ---- 保存 ----
        save_period=5,        # 每个 epoch 备份
        project="runs",
        name="planA_stage3",
        exist_ok=True
    )


if __name__ == "__main__":
    if not SKIP_STAGE1:
        best1 = stage1()
    else:
        print("Skipping stage 1")
        best1 = BEST1
    if not SKIP_STAGE2:
        best2 = stage2(best1)
    else:
        print("Skipping stage 2")
        best2 = BEST2
    best3 = stage3(best2)
    print("✓ Plan-A complete. Best:", best2)
