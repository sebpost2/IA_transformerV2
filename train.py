#!/usr/bin/env python

from __future__ import annotations
import os, glob, math, argparse, time, random, urllib.request, urllib.error
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from minivit import MiniViT, now_str, cp   # tu clase MiniViT

# ----------------------------- CLI --------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--data_root", default="./data",
               help="Carpeta con weapon_detection/ y no_weapon_detection/")
p.add_argument("--epochs",   type=int,   default=150)
p.add_argument("--batch",    type=int,   default=64)
p.add_argument("--lr_base",  type=float, default=4e-4)
p.add_argument("--clip",     type=float, default=5.0)
p.add_argument("--img_size", type=int,   default=96)
p.add_argument("--pos_weight", type=float, default=3.0)
p.add_argument("--smooth", type=float, default=0.05,
               help="Label smoothing factor")
p.add_argument("--weight_decay", type=float, default=1e-4,
               help="L2 regularization factor")
p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam",
               help="Tipo de optimizador")

# ——— descargas de negativos COCO (opcionales) ———————————————— #
p.add_argument("--add_neg",   action="store_true",
               help="Añade negativos de COCO 2017 a no_weapon_detection")
p.add_argument("--neg_train", type=int, default=5000,
               help="Nº de negativos a añadir al split train")
p.add_argument("--neg_val",   type=int, default=1000,
               help="Nº de negativos a añadir al split val")

p.add_argument("--save_dir", default="./weights")
args = p.parse_args()

# ------------------- Hiperparámetros del modelo -------------------------- #
# Patch size se escala con la resolución de entrada para no perder detalle.
# Incrementamos el nº de parches por lado para capturar más detalle.
# Con img_size=96 tendremos 16 parches por lado.
PATCH = max(4, args.img_size // 16)
EMBED, HEADS, LAYERS, NUM_CLASSES = 192, 6, 8, 2   # arma / no-arma

# ------------------- Descarga de negativos “mini-COCO” ------------------- #
def fetch_coco_single(split: str, n_imgs: int, dst_img: str, dst_lbl: str,
                      seed: int = 0):
    """Baja n_imgs JPG individuales del split COCO 2017 a dst_img (negativos)."""
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    total  = 118_287 if split == "train" else 5_000
    urltpl = f"http://images.cocodataset.org/{split}2017/{{:012d}}.jpg"
    rng    = random.Random(seed)
    ids    = rng.sample(range(1, total + 1), n_imgs)

    for idx in tqdm(ids, desc=f"{split} negativos COCO"):
        fname = f"{idx:012d}.jpg"
        path  = os.path.join(dst_img, fname)
        if os.path.exists(path):   # ya descargada
            continue
        try:
            urllib.request.urlretrieve(urltpl.format(idx), path)
            # etiqueta vacía = no arma
            open(os.path.join(dst_lbl, fname[:-3] + "txt"), "w").close()
        except urllib.error.URLError:
            if os.path.exists(path):
                os.remove(path)

if args.add_neg:
    coco_root = Path(args.data_root) / "no_weapon_detection"
    fetch_coco_single("train", args.neg_train,
                      str(coco_root / "train" / "images"),
                      str(coco_root / "train" / "labels"),
                      seed=42)
    fetch_coco_single("val",   args.neg_val,
                      str(coco_root / "val"   / "images"),
                      str(coco_root / "val"   / "labels"),
                      seed=43)

# ----------------------- Helpers de dataset ------------------------------ #
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")

def load_from_root(root_dir: Path, split: str, label_value: int):
    """Carga un split desde root_dir/(train|val)/images|labels."""
    im_dir = root_dir / split / "images"
    paths  = sum([glob.glob(str(im_dir / "**" / e), recursive=True)
                  for e in IMG_EXTS], [])
    imgs, labels = [], []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_size, args.img_size),
                         interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32) / 255.0)
        labels.append(label_value)
    return imgs, labels

def load_split(split: str):
    """Fusiona positivos (weapon_detection) y negativos (no_weapon_detection)."""
    root = Path(args.data_root)
    w_imgs,  w_lbls  = load_from_root(root / "weapon_detection",     split, 1)
    nw_imgs, nw_lbls = load_from_root(root / "no_weapon_detection",  split, 0)
    imgs   = np.asarray(w_imgs + nw_imgs,  np.float32)
    labels = np.asarray(w_lbls + nw_lbls, np.int32)
    # barajar
    idx = np.random.permutation(len(labels))
    return imgs[idx], labels[idx]

def class_balance(x, y):
    """Iguala #positivos y #negativos (simple undersampling)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        print("⚠️  Split contiene solo una clase; se usará sin balancear.")
        idx = np.random.permutation(len(y))
        return x[idx], y[idx]
    n   = min(len(pos), len(neg))
    idx = np.random.permutation(np.concatenate([pos[:n], neg[:n]]))
    return x[idx], y[idx]

# ------------------- Carga y balanceo ------------------------------------ #
x_tr, y_tr = load_split("train")
x_va, y_va = load_split("val")
x_tr, y_tr = class_balance(x_tr, y_tr)
x_va, y_va = class_balance(x_va, y_va)
print(f"✅ Dataset — train={len(y_tr)} | val={len(y_va)} | device {MiniViT.device()}")

if len(y_tr) == 0:
    raise RuntimeError("El set de entrenamiento está vacío; revisa las rutas.")

# ------------------- Traslado a CuPy / NumPy ----------------------------- #
onehot  = np.eye(NUM_CLASSES, dtype=np.float32)
y_tr_oh = cp.asarray(onehot[y_tr])
y_va_oh = cp.asarray(onehot[y_va])
x_tr_cp = cp.asarray(x_tr)
x_va_cp = cp.asarray(x_va)

# ----- estadísticas para normalización -----
mean = x_tr_cp.mean(axis=(0,1,2))
std  = x_tr_cp.std(axis=(0,1,2)) + 1e-7
stats_path = Path(args.save_dir) / "stats.npz"
to_np = lambda a: cp.asnumpy(a) if hasattr(cp, "asnumpy") else a
np.savez(stats_path, mean=to_np(mean), std=to_np(std))
print(f"📊 Saved normalization stats to {stats_path}")

# ------------------- Red y métricas -------------------------------------- #
net = MiniViT(img_size=args.img_size, patch=PATCH, embed=EMBED,
              heads=HEADS, ff=EMBED*2, layers=LAYERS,
              n_classes=NUM_CLASSES, dropout=0.1)

def cross_entropy(logits, labels, smooth=args.smooth):
    """Cross entropy con suavizado y pesos de clase."""
    if smooth > 0.0:
        labels = labels * (1.0 - smooth) + smooth / labels.shape[1]
    maxl = cp.max(logits, 1, keepdims=True)
    logp = logits - maxl - cp.log(cp.sum(cp.exp(logits - maxl), 1,
                                         keepdims=True) + 1e-9)
    w = cp.asarray([1.0, args.pos_weight], dtype=cp.float32)
    return float((-(labels * w) * logp).sum() / logits.shape[0])

def accuracy(logits, labels):
    return float(cp.mean((cp.argmax(logits, 1) == cp.argmax(labels, 1))
                         .astype(cp.float32)))

# ------------------- Entrenamiento -------------------------------------- #
lr  = args.lr_base
t0  = time.time()
rng = np.random.default_rng()
best_acc = 0.0

def augment_batch(xb, rng):
    """Pequeñas augmentaciones de imagen en GPU/CPU."""
    if rng.random() < 0.5:
        xb[:] = cp.flip(xb, 2)  # horizontal flip
    if rng.random() < 0.5:
        xb[:] = xb[:, ::-1]     # vertical flip
    if rng.random() < 0.4:
        gamma = cp.random.uniform(0.7, 1.3,
                                  size=(xb.shape[0],1,1,1)).astype(cp.float32)
        xb[:] = cp.clip(xb * gamma, 0, 1)
    if rng.random() < 0.3:
        # recorte aleatorio simple via CPU (mantiene >=90% del área)
        B, H, W, _ = xb.shape
        scale = rng.uniform(0.9, 1.0)
        h = int(H * scale)
        w = int(W * scale)
        ys = rng.integers(0, H - h + 1)
        xs = rng.integers(0, W - w + 1)
        to_cpu = lambda a: cp.asnumpy(a) if hasattr(cp, "asnumpy") else np.array(a)
        for i in range(B):
            sub = to_cpu(xb[i, ys:ys+h, xs:xs+w])
            sub = cv2.resize(sub, (W, H), interpolation=cv2.INTER_AREA)
            xb[i] = cp.asarray(sub, dtype=cp.float32)


for ep in range(1, args.epochs + 1):
    if ep in (200, 300):
        lr *= 0.3

    idx   = cp.random.permutation(len(x_tr_cp))
    tloss = tacc = nb = 0

    for s in range(0, len(idx), args.batch):
        b  = idx[s:s + args.batch]
        xb = x_tr_cp[b].copy()
        augment_batch(xb, rng)
        xb = (xb - mean) / std

        yb       = y_tr_oh[b]
        cache    = {}
        logits   = net.forward(xb, cache, train=True)
        net.backward(cache, xb, yb,
                     pos_weight=args.pos_weight,
                     smooth=args.smooth)
        gnorm    = net.step(lr, args.clip, args.weight_decay,
                           adam=(args.optimizer == "adam"))

        tloss += cross_entropy(logits, yb)
        tacc  += accuracy(logits, yb)
        nb    += 1

    print(f"[{ep:03d}/{args.epochs}] loss={tloss/nb:.3f} "
          f"acc={tacc/nb:.3f}  gnorm={gnorm:.1f}")

    # —— validación ——
    vloss = vacc = nb = 0
    for s in range(0, len(x_va_cp), args.batch):
        xb = x_va_cp[s:s + args.batch]
        xb = (xb - mean) / std
        yb = y_va_oh[s:s + args.batch]
        logits = net.forward(xb, {}, train=False)
        vloss += cross_entropy(logits, yb, smooth=0.0)
        vacc  += accuracy(logits, yb)
        nb    += 1
    val_acc = vacc / nb
    print(f"          val loss={vloss/nb:.3f}  acc={val_acc:.3f}\n")
    if val_acc > best_acc:
        best_acc = val_acc
        best_ckpt = Path(args.save_dir) / "minivit_best.npz"
        net.save(best_ckpt)
        print(f"✅ Nuevo mejor modelo guardado en {best_ckpt}")

# ------------------- Guardar checkpoint --------------------------------- #
os.makedirs(args.save_dir, exist_ok=True)
ckpt = Path(args.save_dir) / f"minivit_{now_str()}.npz"
net.save(ckpt)
print(f"💾 Pesos guardados en {ckpt} | duración total {time.time()-t0:.0f}s")
