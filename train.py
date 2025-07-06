from __future__ import annotations
import os, glob, math, argparse, time
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
from minivit import MiniViT, now_str, cp   # tu clase MiniViT

# ----------------------------- CLI --------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--data_root", default="./data",
               help="Carpeta con weapon_detection/ y no_weapon_detection/")
p.add_argument("--epochs",   type=int,   default=100)
p.add_argument("--batch",    type=int,   default=8)
p.add_argument("--lr",       type=float, default=4e-4)
p.add_argument("--img_size", type=int,   default=96)

p.add_argument("--save_dir", default="./weights")
args = p.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# --- hyperparams (no CLI to keep things simple) --- #
CLIP = 5.0
POS_WEIGHT = 1.0
DROPOUT = 0.2
WARMUP_FRAC = 0.1
WEIGHT_DECAY = 1e-4

# ------------------- Hiperparámetros del modelo -------------------------- #
# Patch size se escala con la resolución de entrada para no perder detalle.
# Incrementamos el nº de parches por lado para capturar más detalle.
# Con img_size=96 tendremos 16 parches por lado.
PATCH = 8
EMBED, HEADS, LAYERS, NUM_CLASSES = 384, 6, 6, 2   # arma / no-arma

# ------------------- Descarga de negativos “mini-COCO” ------------------- #

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

def class_balance(x, y, oversample=True):
    """Iguala #positivos y #negativos."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        print("⚠️  Split contiene solo una clase; se usará sin balancear.")
        idx = np.random.permutation(len(y))
        return x[idx], y[idx]
    if oversample and len(pos) != len(neg):
        if len(pos) < len(neg):
            add = np.random.choice(pos, len(neg) - len(pos), replace=True)
            pos = np.concatenate([pos, add])
        else:
            add = np.random.choice(neg, len(pos) - len(neg), replace=True)
            neg = np.concatenate([neg, add])
    else:
        n = min(len(pos), len(neg))
        pos, neg = pos[:n], neg[:n]
    idx = np.random.permutation(np.concatenate([pos, neg]))
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
              n_classes=NUM_CLASSES, dropout=DROPOUT,
              pool="mean")

def cross_entropy(logits, labels):
    """Cross entropy con peso para la clase positiva."""
    maxl  = cp.max(logits, 1, keepdims=True)
    logp  = logits - maxl - cp.log(cp.sum(cp.exp(logits - maxl), 1,
                                          keepdims=True) + 1e-9)
    w     = cp.asarray([1.0, POS_WEIGHT], dtype=cp.float32)
    w     = cp.broadcast_to(w, labels.shape)
    return float((-(labels * w) * logp).sum() / logits.shape[0])

def accuracy(logits, labels):
    return float(cp.mean((cp.argmax(logits, 1) == cp.argmax(labels, 1))
                         .astype(cp.float32)))

# ------------------- Entrenamiento -------------------------------------- #
rng = np.random.default_rng()
t0  = time.time()

best_acc = 0.0

def augment_batch(xb, rng):
    """Pequeñas augmentaciones de imagen en GPU/CPU."""
    if rng.random() < 0.5:
        xb[:] = cp.flip(xb, 2)  # horizontal flip
    if rng.random() < 0.4:
        gamma = cp.random.uniform(0.7, 1.3,
                                  size=(xb.shape[0],1,1,1)).astype(cp.float32)
        xb[:] = cp.clip(xb * gamma, 0, 1)
    if rng.random() < 0.2:
        # ligero cambio de color por canal
        jitter = cp.random.uniform(0.8, 1.2, (1,1,1,3)).astype(cp.float32)
        xb[:] = cp.clip(xb * jitter, 0, 1)
    if rng.random() < 0.5:
        # recorte aleatorio simple via CPU (mantiene >=90% del área)
        # --- AVISO: Causa alta carga de CPU y ralentiza el entrenamiento ---
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
    if rng.random() < 0.5:
        # rotación aleatoria ligera
        # --- AVISO: Causa alta carga de CPU y ralentiza el entrenamiento ---
        B, H, W, _ = xb.shape
        angle = rng.uniform(-20.0, 20.0)
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        to_cpu = lambda a: cp.asnumpy(a) if hasattr(cp, "asnumpy") else np.array(a)
        for i in range(B):
            rot = cv2.warpAffine(to_cpu(xb[i]), M, (W, H),
                                 borderMode=cv2.BORDER_REFLECT)
            xb[i] = cp.asarray(rot, dtype=cp.float32)
    if rng.random() < 0.3:
        noise = cp.random.normal(0, 0.05, xb.shape)
        noise = noise.astype(cp.float32)
        xb[:] = cp.clip(xb + noise, 0, 1)

    if rng.random() < 0.5:
        # Random shift (GPU-based)
        B, H, W, C = xb.shape
        pad_h, pad_w = H // 8, W // 8
        padded = cp.zeros((B, H + 2*pad_h, W + 2*pad_w, C), dtype=cp.float32)
        padded[:, pad_h:pad_h+H, pad_w:pad_w+W, :] = xb
        h_off = rng.integers(0, 2*pad_h)
        w_off = rng.integers(0, 2*pad_w)
        xb[:] = padded[:, h_off:h_off+H, w_off:w_off+W, :]

def lr_schedule(ep: int) -> float:
    warm = max(1, int(args.epochs * WARMUP_FRAC))
    if ep <= warm:
        return args.lr * ep / warm
    prog = (ep - warm) / max(1, args.epochs - warm)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * prog))

for ep in range(1, args.epochs + 1):
    lr = lr_schedule(ep)

    # Bucle de entrenamiento
    idx   = cp.random.permutation(len(x_tr_cp))
    tloss = tacc = nb = 0
    net.training = True
    for s in range(0, len(idx), args.batch):
        b  = idx[s:s + args.batch]
        xb = x_tr_cp[b].copy()
        augment_batch(xb, rng)
        yb = y_tr_oh[b]
        xb = (xb - mean) / std

        cache  = {}
        logits = net.forward(xb, cache, train=True)
        net.backward(cache, xb, yb, pos_weight=POS_WEIGHT)
        gnorm  = net.step(lr, CLIP, WEIGHT_DECAY)

        tloss += cross_entropy(logits, yb)
        tacc  += accuracy(logits, yb)
        nb    += 1

    # Bucle de validación
    vloss = vacc = nb_val = 0
    net.training = False
    for s in range(0, len(x_va_cp), args.batch):
        xb = x_va_cp[s:s + args.batch]
        xb = (xb - mean) / std
        yb = y_va_oh[s:s + args.batch]
        logits = net.forward(xb, {}, train=False)
        vloss += cross_entropy(logits, yb)
        vacc  += accuracy(logits, yb)
        nb_val += 1
    val_acc = vacc / nb_val

    print(f"[{ep:03d}/{args.epochs}] loss={tloss/nb:.3f} acc={tacc/nb:.3f} | val_loss={vloss/nb_val:.3f} val_acc={val_acc:.3f} | gnorm={gnorm:.1f} lr={lr:.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_ckpt = Path(args.save_dir) / "minivit_best.npz"
        net.save(best_ckpt)
        # Silenciado para no ser tan verboso
        # print(f"  -> Nuevo mejor modelo guardado en {best_ckpt}")

# ------------------- Guardar checkpoint --------------------------------- #
os.makedirs(args.save_dir, exist_ok=True)
ckpt = Path(args.save_dir) / f"minivit_{now_str()}.npz"
net.save(ckpt)
print(f"💾 Pesos guardados en {ckpt} | duración total {time.time()-t0:.0f}s")
