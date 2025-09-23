# make_filelists.py
import os, random
from pathlib import Path

DATA_DIR = "DATA/PokemonData"   # class_index.json과 images/가 있는 루트
IMAGES_SUBDIR = "new"
MAX_PER_CLASS = 100
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)
root = Path(DATA_DIR)
images_root = root / IMAGES_SUBDIR
class_names = sorted([d.name for d in images_root.iterdir() if d.is_dir()])

name_to_idx = {name: i for i, name in enumerate(class_names)}
train_lines, val_lines = [], []

for cname in class_names:
    cdir = images_root / cname
    imgs = sorted([p for p in cdir.iterdir() if p.is_file()])
    random.shuffle(imgs)
    imgs = imgs[:MAX_PER_CLASS]
    k = int(round(len(imgs) * TRAIN_RATIO))
    train_imgs, val_imgs = imgs[:k], imgs[k:]
    lbl = name_to_idx[cname]
    for p in train_imgs:
        train_lines.append(f"{p.relative_to(root).as_posix()} {lbl}\n")
    for p in val_imgs:
        val_lines.append(f"{p.relative_to(root).as_posix()} {lbl}\n")

(root / "train.txt").write_text("".join(train_lines), encoding="utf-8")
(root / "val.txt").write_text("".join(val_lines), encoding="utf-8")

print("완료:", len(class_names), "classes",
      "| train:", len(train_lines),
      "| val:", len(val_lines))