#!/usr/bin/env python
import os, glob, math, argparse
import numpy as np
import cv2, matplotlib.pyplot as plt
from pathlib import Path

# Use the cp alias provided by minivit, which falls back to NumPy if no GPU.
from minivit import MiniViT, cp

# ---------------- CLI ---------------- #
p = argparse.ArgumentParser()
p.add_argument("--data_root",   default="./data",   help="Folder with test/")
p.add_argument("--weights_dir", default="./weights")
p.add_argument("--show", type=int, default=12)
p.add_argument("--threshold", type=float, default=0.5,
               help="Probability threshold to classify as weapon")
args = p.parse_args()

# --------- load the latest checkpoint -------- #
ckpts = sorted(Path(args.weights_dir).glob("minivit_*.npz"))
if not ckpts:
    raise FileNotFoundError(f"No checkpoints in {args.weights_dir}")
latest = ckpts[-1]
print(f"▶️  Using weights: {latest.name}")

# --- infer model dimensions from checkpoint --- #
meta = MiniViT.inspect_checkpoint(latest)
img_size = meta['img_size']
net = MiniViT(img_size=img_size, patch=meta['patch'], embed=meta['embed'],
              heads=6, ff=meta['embed']*2, layers=6,
              n_classes=2, dropout=0.0,
              pool="mean")
net.load(latest)

# --------- normalization stats --------- #
stats_files = sorted(Path(args.weights_dir).glob("stats*.npz"))
if stats_files:
    st = np.load(stats_files[-1])
    mean, std = st['mean'], st['std']
else:
    mean, std = 0.0, 1.0

# --------- load test images ---------- #
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")
test_dir = Path(args.data_root) / "test"
paths = sum([glob.glob(str(test_dir / ext)) for ext in IMG_EXTS], [])
if not paths:
    raise FileNotFoundError(f"Put images into {test_dir} for testing")
paths.sort()
if args.show:
    paths = paths[: args.show]

imgs = []
for p in paths:
    im = cv2.imread(p)
    im = cv2.cvtColor(
        cv2.resize(im, (img_size, img_size)),
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0
    im = (im - mean) / std
    imgs.append(im)

# —— forward on whichever backend (GPU or CPU) —— #
logits = net.forward(cp.asarray(imgs), {}, train=False)

# —— safe conversion back to NumPy —— #
raw   = net.softmax(logits)
probs = cp.asnumpy(raw) if hasattr(cp, "asnumpy") else raw

# -------------- print results --------------- #
for p, pr in zip(paths, probs):
    pred = "ARMA" if pr[1] >= args.threshold else "NO ARMA"
    print(f"{os.path.basename(p):25s} → {pred:7s}  p={pr[1]:.2f}")

# ----- display if requested -------------- #
if args.show:
    cols = 4
    rows = math.ceil(len(paths) / cols)
    plt.figure(figsize=(3.2 * cols, 3.2 * rows))
    for i, (p, pr) in enumerate(zip(paths, probs)):
        plt.subplot(rows, cols, i + 1)
        # muestra la imagen sin normalizar para evitar avisos de matplotlib
        img_disp = np.clip(imgs[i] * std + mean, 0, 1)
        plt.imshow(img_disp)
        plt.axis("off")
        is_weapon = pr[1] >= args.threshold
        plt.title(
            f"{os.path.basename(p)}\n"
            f"{'ARMA' if is_weapon else 'NO'}  p={pr[1]:.2f}",
            color="red" if is_weapon else "green"
        )
    plt.tight_layout()
    plt.show()
