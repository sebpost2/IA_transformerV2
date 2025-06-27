"""
MiniViT - Transformer miniatura para clasificación de imágenes
--------------------------------------------------------------
Implementación didáctica desde cero, sin frameworks de alto nivel como TensorFlow o PyTorch. Utiliza únicamente NumPy o CuPy para las operaciones de tensores.
Python 3.10+
CuPy (GPU NVIDIA) o NumPy (CPU puro)
"""

from __future__ import annotations
import math, time
import numpy as np

# ---------- intento de importar CuPy ----------
try:
    import cupy as cp
    # comprueba que haya un driver válido
    cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = True
except Exception:
    # o bien no está CuPy o no hay driver compatible
    cp = np                                # alias → NumPy
    GPU_AVAILABLE = False


# --------------------------- MiniViT --------------------------- #
class MiniViT:
    def __init__(self, img_size: int = 64, patch: int = 8,
                 embed: int = 96, heads: int = 4, ff: int | None = None,
                 layers: int = 6, n_classes: int = 2, dropout: float = 0.1,
                 seed: int = 42):
        """Inicializa la red y todos sus pesos."""
        ff = ff or embed * 2
        rng = cp.random.default_rng(seed)

        # Dimensiones principales
        self.P  = patch                     # tamaño del parche
        self.N  = (img_size // patch) ** 2  # nº de parches por imagen
        self.N1 = self.N + 1                # +1 para el token [CLS]
        self.E  = embed                     # dimensión de embedding
        self.H  = heads                    # nº de cabezas de atención
        self.D  = embed // heads           # dimensión por cabeza
        self.FF = ff                       # dimensión del MLP interno
        self.L  = layers                   # nº de bloques Transformer
        self.dropout = dropout             # probabilidad de dropout
        self.training = True               # modo entrenamiento/inferencia
        self.rng = rng                     # generador aleatorio

        # Xavier helper
        xavier = lambda fan_in: math.sqrt(2.0 / fan_in)

        # Proyección de parches + posición
        # Incluimos los 3 canales de color en la proyecci\u00f3n del parche
        patch_dim = patch * patch * 3
        self.Wp  = rng.standard_normal((patch_dim, embed),
                                       dtype=cp.float32) * xavier(patch_dim)
        self.bp  = cp.zeros(embed, dtype=cp.float32)
        self.pos = rng.standard_normal((1, self.N1, embed),
                                       dtype=cp.float32) * 0.01
        self.cls = rng.standard_normal((1, 1, embed),
                                       dtype=cp.float32) * 0.01

        # Bloques Transformer
        self.blk = []
        for _ in range(layers):
            fan = embed
            p = dict(
                # LN1
                g1 = cp.ones(embed, dtype=cp.float32),
                b1 = cp.zeros(embed, dtype=cp.float32),
                # QKV y salida Wo
                Wq = rng.standard_normal((embed, embed), dtype=cp.float32) * xavier(fan),
                Wk = rng.standard_normal((embed, embed), dtype=cp.float32) * xavier(fan),
                Wv = rng.standard_normal((embed, embed), dtype=cp.float32) * xavier(fan),
                Wo = rng.standard_normal((embed, embed), dtype=cp.float32) * xavier(fan),
                # LN2
                g2 = cp.ones(embed, dtype=cp.float32),
                b2 = cp.zeros(embed, dtype=cp.float32),
                # MLP interno
                W1 = rng.standard_normal((embed, ff), dtype=cp.float32) * xavier(embed),
                b1_= cp.zeros(ff, dtype=cp.float32),
                W2 = rng.standard_normal((ff, embed), dtype=cp.float32) * xavier(ff),
                b2_= cp.zeros(embed, dtype=cp.float32)
            )
            self.blk.append(p)

        # Cabeza de clasificación
        self.Wc1 = rng.standard_normal((embed, 128), dtype=cp.float32) * xavier(embed)
        self.bc1 = cp.zeros(128, dtype=cp.float32)
        self.Wc2 = rng.standard_normal((128, n_classes), dtype=cp.float32) * xavier(128)
        self.bc2 = cp.zeros(n_classes, dtype=cp.float32)

        # Gradientes
        self._alloc_grads()
        # Estados para Adam
        self.opt_m = {n: cp.zeros_like(p) for p, _, n in self._param_items()}
        self.opt_v = {n: cp.zeros_like(p) for p, _, n in self._param_items()}
        self.opt_t = 0

    # -------------------- helpers / activaciones ------------------ #
    @staticmethod
    def gelu(x):
        k, c = 0.7978845608028654, 0.044715
        return 0.5 * x * (1. + cp.tanh(k * (x + c * x**3)))

    @staticmethod
    def gelu_grad(x):
        k, c = 0.7978845608028654, 0.044715
        t = k * (x + c * x**3)
        phi = cp.tanh(t)
        return 0.5 * (1. + phi) + 0.5 * x * (1. - phi**2) * k * (1. + 3.*c*x**2)

    def dense(self, x, W, b):
        return x @ W + b

    def softmax(self, x):
        m = cp.max(x, -1, keepdims=True)
        e = cp.exp(x - m)
        return e / cp.sum(e, -1, keepdims=True)

    def _patchify(self, imgs):
        """Divide las imágenes en parches manteniendo los 3 canales."""
        B, H, W, C = imgs.shape
        s0, s1, s2, s3 = imgs.strides
        view = cp.lib.stride_tricks.as_strided(
            imgs,
            shape=(B, H // self.P, W // self.P, self.P, self.P, C),
            strides=(s0, s1 * self.P, s2 * self.P, s1, s2, s3)
        )
        return view.reshape(B, self.N, self.P * self.P * C)

    def _drop(self, x, cache, key):
        """Aplica dropout y guarda la máscara en cache."""
        if not self.training or self.dropout <= 0.0:
            return x
        mask = (self.rng.random(x.shape, dtype=cp.float32) >= self.dropout)
        cache[key] = mask
        return x * mask / (1.0 - self.dropout)

    # ----------------------------- forward ------------------------ #
    def forward(self, imgs, cache: dict | None = None, train: bool = False):
        """Realiza el forward; si train=True aplica dropout."""
        self.training = train
        cache = {} if cache is None else cache
        B = imgs.shape[0]
        patch_emb = self.dense(self._patchify(imgs), self.Wp, self.bp)
        cls_tok   = cp.repeat(self.cls, B, axis=0)
        x = cp.concatenate([
            cls_tok + self.pos[:, :1],
            patch_emb + self.pos[:, 1:]
        ], axis=1)

        for l, p in enumerate(self.blk):
            st = cache.setdefault(f"b{l}", {})

            # LN1
            mu1 = x.mean(-1, keepdims=True)
            std1 = cp.sqrt(x.var(-1, keepdims=True) + 1e-5)
            n1 = (x - mu1) / std1
            y  = p['g1'] * n1 + p['b1']

            # Atención multi-cabeza
            q = self.dense(y, p['Wq'], 0.0)
            k = self.dense(y, p['Wk'], 0.0)
            v = self.dense(y, p['Wv'], 0.0)
            split = lambda t: t.reshape(B, self.N1, self.H, self.D).transpose(0,2,1,3)
            qh, kh, vh = map(split, (q, k, v))
            w = self.softmax((qh @ kh.transpose(0,1,3,2)) / math.sqrt(self.D))
            att = (w @ vh).transpose(0,2,1,3).reshape(B, self.N1, self.E)
            z = self.dense(att, p['Wo'], 0.0)
            z = self._drop(z, st, 'mask_att')
            x = x + z  # residual

            st.update(mu1=mu1, std1=std1, n1=n1, y=y,
                      qh=qh, kh=kh, vh=vh, w=w, att=att)

            # LN2
            mu2 = x.mean(-1, keepdims=True)
            std2 = cp.sqrt(x.var(-1, keepdims=True) + 1e-5)
            n2 = (x - mu2) / std2
            y2 = p['g2'] * n2 + p['b2']

            # MLP interno
            h_raw = self.dense(y2, p['W1'], p['b1_'])
            h     = self.gelu(h_raw)
            h2    = self.dense(h, p['W2'], p['b2_'])
            h2    = self._drop(h2, st, 'mask_ff')
            x     = x + h2

            st.update(mu2=mu2, std2=std2, n2=n2, y2=y2,
                      h_raw=h_raw, h=h)

        # Cabeza de clasificación
        cls_token = x[:, 0]
        h_cls_raw = self.dense(cls_token, self.Wc1, self.bc1)
        h_cls     = self.gelu(h_cls_raw)
        h_cls     = self._drop(h_cls, cache, 'mask_cls')
        logits    = self.dense(h_cls, self.Wc2, self.bc2)
        cache.update(logits=logits, cls_token=cls_token,
                     h_cls_raw=h_cls_raw, h_cls=h_cls)
        return logits

    # ----------------------------- backward ----------------------- #
    def backward(self, cache, imgs, labels, *, pos_weight: float = 1.0,
                 smooth: float = 0.0):
        """Backprop con opción de peso positivo y label smoothing."""
        # limpia gradientes
        for g in self.grads.values():
            if isinstance(g, list):
                for gg in g: [v.fill(0.) for v in gg.values()]
            else:
                g.fill(0.)

        B = labels.shape[0]
        probs = self.softmax(cache['logits'])
        if smooth > 0.0:
            labels = labels * (1.0 - smooth) + smooth / labels.shape[1]
        # ponderación por clase en la derivada
        class_w = cp.asarray([1.0, pos_weight], dtype=cp.float32)
        dlog = (probs - labels) * class_w / B

        # cabeza MLP
        self.grads['Wc2'] += cache['h_cls'].T @ dlog
        self.grads['bc2'] += dlog.sum(0)
        dh = dlog @ self.Wc2.T
        if 'mask_cls' in cache:
            dh = dh * cache['mask_cls'] / (1.0 - self.dropout)
        dh *= self.gelu_grad(cache['h_cls_raw'])
        self.grads['Wc1'] += cache['cls_token'].T @ dh
        self.grads['bc1'] += dh.sum(0)
        dx = cp.zeros((B, self.N1, self.E), dtype=cp.float32)
        dx[:,0,:] = dh @ self.Wc1.T
        scale = 1.0 / math.sqrt(self.D)

        # bloques en reversa
        for l in reversed(range(self.L)):
            p, g, st = self.blk[l], self.grads['blk'][l], cache[f"b{l}"]

            # --- MLP back ---
            dx_res2 = dx.copy()
            if 'mask_ff' in st:
                dx = dx * st['mask_ff'] / (1.0 - self.dropout)
            g['W2']  += st['h'].reshape(-1, self.FF).T @ dx.reshape(-1, self.E)
            g['b2_'] += dx.sum((0,1))
            dh = dx @ p['W2'].T
            dh *= self.gelu_grad(st['h_raw'])
            g['W1']  += st['y2'].reshape(-1, self.E).T @ dh.reshape(-1, self.FF)
            g['b1_'] += dh.sum((0,1))
            dx = dh @ p['W1'].T + dx_res2

            # --- LN2 back ---
            dx = self._ln_back(dx, st['n2'], st['std2'],
                               p['g2'], g['g2'], g['b2'])

            # --- Atención back ---
            dx_res1 = dx.copy()
            if 'mask_att' in st:
                dx = dx * st['mask_att'] / (1.0 - self.dropout)
            g['Wo'] += st['att'].reshape(-1, self.E).T @ dx.reshape(-1, self.E)
            dattn = (dx @ p['Wo'].T).reshape(B, self.N1, self.H, self.D).transpose(0,2,1,3)
            dV = st['w'].transpose(0,1,3,2) @ dattn
            g['Wv'] += st['y'].reshape(-1, self.E).T @ dV.transpose(0,2,1,3).reshape(-1, self.E)
            dVm = dV.transpose(0,2,1,3).reshape(B, self.N1, self.E)
            dy_v = dVm @ p['Wv'].T

            dw = dattn @ st['vh'].transpose(0,1,3,2)
            ds = st['w'] * (dw - (dw * st['w']).sum(-1, keepdims=True))
            dS = ds * scale
            dQ = dS @ st['kh']
            dK = dS.transpose(0,1,3,2) @ st['qh']
            dQm = dQ.transpose(0,2,1,3).reshape(B, self.N1, self.E)
            dKm = dK.transpose(0,2,1,3).reshape(B, self.N1, self.E)
            g['Wq'] += st['y'].reshape(-1, self.E).T @ dQm.reshape(-1, self.E)
            g['Wk'] += st['y'].reshape(-1, self.E).T @ dKm.reshape(-1, self.E)
            dy_q = dQm @ p['Wq'].T
            dy_k = dKm @ p['Wk'].T

            dx = dx_res1 + dy_v + dy_q + dy_k
            dx = self._ln_back(dx, st['n1'], st['std1'],
                               p['g1'], g['g1'], g['b1'])

        # proyección final
        patch = self._patchify(imgs).reshape(-1, self.P * self.P * 3)
        self.grads['Wp'] += patch.T @ dx[:,1:,:].reshape(-1, self.E)
        self.grads['bp'] += dx[:,1:,:].sum((0,1))
        self.grads['cls'] += dx[:,0:1,:].sum(0)
        self.grads['pos'] += dx.sum(0, keepdims=True)

    # --------- backprop de LayerNorm (se había perdido) --------- #
    def _ln_back(self, dY, n, std, gamma, g_gamma, g_beta):
        g_gamma += cp.sum(dY * n, axis=(0,1))
        g_beta  += cp.sum(dY, axis=(0,1))
        std = cp.maximum(std, 1e-3)

        d_norm = dY * gamma
        d_var  = cp.sum(d_norm * n * -0.5 / std**2, axis=-1, keepdims=True)
        d_mu   = cp.sum(-d_norm / std, axis=-1, keepdims=True) \
                 + d_var * cp.mean(-2.0 * n * std, axis=-1, keepdims=True)
        B, N, E = dY.shape
        return d_norm / std + d_var * 2.0 * n * std / E + d_mu / E

    # ---------------------- paso de SGD ------------------------ #
    def step(self, lr: float = 3e-4, clip: float = 5.0, wd: float = 0.0,
             adam: bool = False):
        """Actualiza los pesos con SGD o Adam."""
        norm = 0.0
        for _, g, _ in self._param_items():
            norm += float(cp.sum(g * g))
        norm = math.sqrt(norm)
        scale = clip / (norm + 1e-6) if clip and norm > clip else 1.0

        self.opt_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8

        for p, g, name in self._param_items():
            upd = scale * g + (wd * p if name[0] not in ('b', 'g') else 0.0)
            if adam:
                m = self.opt_m[name]
                v = self.opt_v[name]
                m[:] = b1 * m + (1 - b1) * upd
                v[:] = b2 * v + (1 - b2) * (upd ** 2)
                m_hat = m / (1 - b1 ** self.opt_t)
                v_hat = v / (1 - b2 ** self.opt_t)
                p -= lr * m_hat / (cp.sqrt(v_hat) + eps)
            else:
                p -= lr * upd
            g.fill(0)
        return norm

    def _param_list(self):
        """Devuelve una lista ordenada con todos los tensores del modelo."""
        params = [v for v in self.__dict__.values()
                  if isinstance(v, cp.ndarray)]
        for blk in self.blk:
            params.extend(blk[k] for k in blk)
        return params

    def _param_items(self):
        """Itera (param, grad, nombre) sobre todos los tensores."""
        yield self.Wp, self.grads['Wp'], 'Wp'
        yield self.bp, self.grads['bp'], 'bp'
        yield self.pos, self.grads['pos'], 'pos'
        yield self.cls, self.grads['cls'], 'cls'
        yield self.Wc1, self.grads['Wc1'], 'Wc1'
        yield self.bc1, self.grads['bc1'], 'bc1'
        yield self.Wc2, self.grads['Wc2'], 'Wc2'
        yield self.bc2, self.grads['bc2'], 'bc2'
        for i, blk in enumerate(self.blk):
            g = self.grads['blk'][i]
            for k in blk:
                yield blk[k], g[k], f'b{i}_{k}'

    # -------------------- Guardar checkpoint ---------------------------
    def save(self, path: str | Path):
        """Guarda los parámetros en formato arr_0, arr_1, … (npz)."""
        arrays = [p.get() if hasattr(p, 'get') else p
                  for p in self._param_list()]  # CuPy ➜ NumPy
        np.savez_compressed(str(path), *arrays)

    # -------------------- Cargar checkpoint ----------------------------
     # -------------------- Cargar checkpoint ----------------------------
    def load(self, path: str | Path):
        """
        Carga un .npz:
        • Si las claves son arr_0, arr_1… → las ordena numéricamente.
        • En cualquier otro caso usa el orden original de data.files.
        """
        data  = np.load(path, allow_pickle=False)

        # --- Determina estrategia de orden ---
        def is_arr_k(name):
            return name.startswith("arr_") and name[4:].isdigit()

        if all(is_arr_k(k) for k in data.files):             # Caso A
            files = sorted(data.files, key=lambda k: int(k[4:]))
        else:                                                # Caso B
            files = list(data.files)   # orden de inserción (np.savez)

        arrays = [data[k] for k in files]
        params = self._param_list()

        if len(arrays) != len(params):
            raise ValueError(
                f"Checkpoint tiene {len(arrays)} tensores, "
                f"pero la red espera {len(params)}."
            )

        for p, a in zip(params, arrays):
            p[...] = cp.asarray(a)  # copia 
    # ------------------------ utils --------------------------- #
    def _alloc_grads(self):
        self.grads = {
            'Wp': cp.zeros_like(self.Wp), 'bp': cp.zeros_like(self.bp),
            'pos': cp.zeros_like(self.pos),
            'cls': cp.zeros_like(self.cls),
            'Wc1': cp.zeros_like(self.Wc1), 'bc1': cp.zeros_like(self.bc1),
            'Wc2': cp.zeros_like(self.Wc2), 'bc2': cp.zeros_like(self.bc2),
            'blk': [{k: cp.zeros_like(v) for k, v in b.items()} for b in self.blk]
        }

    @staticmethod
    def device():
        return "GPU" if GPU_AVAILABLE else "CPU"

    @staticmethod
    def inspect_checkpoint(path: str | Path):
        """Lee metadatos básicos (img_size, patch, embed) de un .npz."""
        data = np.load(path, allow_pickle=False)

        def is_arr_k(name):
            return name.startswith("arr_") and name[4:].isdigit()

        files = (sorted(data.files, key=lambda k: int(k[4:]))
                 if all(is_arr_k(k) for k in data.files)
                 else list(data.files))

        Wp  = data[files[0]]          # (P*P*3, E)
        pos = data[files[2]]          # (1, N+1, E)

        patch  = int(round(math.sqrt(Wp.shape[0] / 3)))
        embed  = int(Wp.shape[1])
        n1     = int(pos.shape[1])
        img_sz = patch * int(math.sqrt(n1 - 1))
        return dict(img_size=img_sz, patch=patch, embed=embed)


# -------------- helper para timestamp en checkpoints ---------- #
def now_str():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
