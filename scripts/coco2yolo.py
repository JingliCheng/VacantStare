"""coco2yolo_embedded.py

📝 Purpose
--------
Convert **multiple COCO‑format datasets** into a single YOLO directory *without* any
command‑line arguments.  All paths, label mapping, and split ratios are defined
as Python variables at the top of the file—so you can just `python coco2yolo_embedded.py`
and forget about long CLI flags.

🔧 How to use
------------
1. Open this file and edit the variables under **CONFIG SECTION**:
   - `SRC_JSONS`: list of paths to COCO annotation JSON files.
   - `DST_DIR`: where merged YOLO data will be written.
   - `LABEL_MAP`: dict that maps COCO category *names* ➔ new YOLO class IDs.
   - `VAL_SPLIT`: fraction of images used for validation.
2. Run: `python coco2yolo_embedded.py`

Dependencies: `pip install pycocotools tqdm`  (plus standard libs).

Feel free to drop this script into your project root and tweak as needed.
"""
from pathlib import Path
import shutil, random, math, json
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
from pycocotools.coco import COCO

# ============================================================
# CONFIG SECTION  (EDIT HERE)
# ============================================================

# There are some datasets only coming with body but no head.
SRC_JSONS: List[str] = [
    "../datasets/Aimbot.v2i.coco/train/_annotations.coco.json",
    # "../datasets/cs2 v1.v2i.coco/train/_annotations.coco.json",
    "../datasets/cs2-AI.v1i.coco/train/_annotations.coco.json",
    # "../datasets/CSGO videogame.v2-release.coco/train/_annotations.coco.json",
    "../datasets/keremberke/train/_annotations.coco.json",

    "../datasets/Aimbot.v2i.coco/valid/_annotations.coco.json",
    # "../datasets/cs2 v1.v2i.coco/valid/_annotations.coco.json",
    "../datasets/cs2-AI.v1i.coco/valid/_annotations.coco.json",
    # "../datasets/CSGO videogame.v2-release.coco/valid/_annotations.coco.json",
    "../datasets/keremberke/valid/_annotations.coco.json",

    "../datasets/Aimbot.v2i.coco/test/_annotations.coco.json",
    # "../datasets/cs2 v1.v2i.coco/test/_annotations.coco.json",
    # "../datasets/cs2-AI.v1i.coco/test/_annotations.coco.json",
    # "../datasets/CSGO videogame.v2-release.coco/test/_annotations.coco.json",
    "../datasets/keremberke/test/_annotations.coco.json",
    # add more paths as needed
]

DST_DIR: str = "../datasets/merged_yolo"           # Output directory (will be created)

LABEL_MAP: Dict[str, int] = {
    "CTHead": 0,
    "THead": 0,        # Aimbot
    "1": 0,            
    "3": 0,            # CS2 AI v1
    "cthead": 0,       
    "thead": 0,        # keremberke
    # "person": 0       # map whole body to 0 if desired, else remove
}

VAL_SPLIT: float = 0.1  # 10 % of images go to validation set
# ============================================================

# Derived paths
DST_IMAGES = Path(DST_DIR) / "images"
DST_LABELS = Path(DST_DIR) / "labels"

for subset in ["train", "val"]:
    (DST_IMAGES / subset).mkdir(parents=True, exist_ok=True)
    (DST_LABELS / subset).mkdir(parents=True, exist_ok=True)


def coco_to_yolo(coco_path: Path):
    """Convert a single COCO json to YOLO txt files."""
    coco = COCO(str(coco_path))
    img_ids = coco.getImgIds()
    random.shuffle(img_ids)
    split_idx = math.floor(len(img_ids) * (1 - VAL_SPLIT))

    for i, img_id in enumerate(tqdm(img_ids, desc=coco_path.name)):
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        subset = "train" if i < split_idx else "val"
        img_src = Path(coco_path).parent / img["file_name"]
        img_dst = DST_IMAGES / subset / img["file_name"]
        lbl_dst = DST_LABELS / subset / (Path(img["file_name"]).stem + ".txt")
        img_dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy image (skip if already copied from another dataset)
        if not img_dst.exists():
            shutil.copy2(img_src, img_dst)

        h, w = img["height"], img["width"]
        lines = []
        for a in anns:
            cat_name = coco.loadCats(a["category_id"])[0]["name"]
            if cat_name not in LABEL_MAP:
                continue  # skip unwanted categories
            x, y, bw, bh = a["bbox"]
            # Filter ultra‑small boxes (<4 px on either side)
            if bw < 4 or bh < 4:
                continue
            cx, cy = x + bw / 2, y + bh / 2
            cls_id = LABEL_MAP[cat_name]
            lines.append(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")

        # Write label file (even if empty — some trainers expect a file)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        with lbl_dst.open("w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    print("[coco2yolo] Starting conversion…")
    for js in SRC_JSONS:
        coco_to_yolo(Path(js))
    print("[coco2yolo] Done! YOLO dataset written to:", DST_DIR)
