# anchor_cluster.py
# -------------------------------------------------------
# Re-cluster anchors for BGF-YOLOv10-n and write them
# back into bgf_yolov10n.yaml in-place.
#
# ─ Data config : datasets/merged_yolo/cs.yaml
# ─ Model cfg   : scripts/bgf_yolov10n.yaml
# ─ img size    : 960 × 960
# -------------------------------------------------------

from ultralytics import YOLO
from pathlib import Path
import yaml, re

MODEL_YAML = Path("../scripts/bgf_yolov10n.yaml")
DATA_YAML  = Path("../datasets/merged_yolo/cs.yaml")
IMG_SIZE   = 960

def main():
    # 1️⃣  Load structure-only model (weights not needed)
    model = YOLO(str(MODEL_YAML))

    # 2️⃣  Run anchor clustering
    clusters = model.anchors(data=str(DATA_YAML), imgsz=IMG_SIZE)  # returns list
    # clusters shape: [12]  ->  3 anchors × 4 scales (P2-P5)

    # 3️⃣  Pretty-print result
    a = [f"{w},{h}" for (w, h) in clusters]
    print("\nNew anchors (P2-P5, 3 each):")
    for i, s in enumerate(("P2", "P3", "P4", "P5")):
        print(f"{s}: {', '.join(a[i*3:(i+1)*3])}")

    # 4️⃣  Write back into YAML (replace existing anchors block)
    with open(MODEL_YAML, "r", encoding="utf-8") as f:
        cfg = f.read()

    new_anchor_line = "  - [" + ", ".join(a) + "]"
    cfg = re.sub(r"anchors:[\s\S]+?$", "anchors:\n" + new_anchor_line, cfg, flags=re.M)

    with open(MODEL_YAML, "w", encoding="utf-8") as f:
        f.write(cfg)

    print(f"\n✅ Anchors written back to {MODEL_YAML}")

if __name__ == "__main__":
    main()
