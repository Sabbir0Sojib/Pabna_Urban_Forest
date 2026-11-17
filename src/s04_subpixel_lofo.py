"""Step 4. Coarse-to-fine sub-pixel model (following He et al., 2022) in the independent field test.

A gradient-boosting regressor predicts the canopy fraction of each 1 m cell from the 10 spectral
values (8 bands, NDVI, NDRE) of the 5 x 5 neighbourhood of 3 m PlanetScope pixels around it, plus
the cell's position (one of 3 x 3) inside its 3 m pixel. Reference: 1 m SegFormer canopy fraction.

For each held-out field k (shift, model and threshold from the two training fields only):
  F1 at 1 m (cell = canopy if reference fraction >= 0.5), R2, and the area bias
  (predicted / reference canopy fraction, summed over the field). The three area biases are used
  in step 5 to correct the city-wide map.

Outputs: outputs/subpixel_lofo.json, outputs/subpixel_pred_1m_F{k}.npy
"""
import gc
import sys

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import f1_score, precision_score, r2_score, recall_score

from utils import best_shift, load_config, load_scene, nd, out, p, save_json

FIELDS = (1, 2, 3)


def spectral_stack(img, sh):
    x = np.stack([ndimage.shift(b, sh, order=1, mode="nearest") for b in img]) / 10000.0
    return np.stack([x[i] for i in range(8)] + [nd(x[7], x[5]), nd(x[7], x[6])]).astype(np.float32)


def neighbourhood(F, R, C, half=2):
    return np.concatenate([F[:, R + dr, C + dc] for dr in range(-half, half + 1) for dc in range(-half, half + 1)], 0).T


def build_rows(F, frac3, frac1, o1):
    """Rows for every 1 m cell inside the fully covered 3 m pixels of one field."""
    R, C = np.where(np.isfinite(frac3))
    base = neighbourhood(F, R, C)
    X, Y, grp = [], [], []
    for i in range(3):
        for j in range(3):
            rr = 3 * R + i - o1[0]
            cc = 3 * C + j - o1[1]
            inb = (rr >= 0) & (rr < frac1.shape[0]) & (cc >= 0) & (cc < frac1.shape[1])
            y = np.full(len(R), np.nan)
            y[inb] = frac1[rr[inb], cc[inb]]
            pos = np.zeros((len(R), 9), np.float32)
            pos[:, 3 * i + j] = 1
            X.append(np.hstack([base, pos]))
            Y.append(y)
            grp.append(np.arange(len(R)))
    X, Y, grp = np.vstack(X).astype(np.float32), np.concatenate(Y), np.concatenate(grp)
    m = np.isfinite(Y)
    return X[m], Y[m], grp[m]


def hgb(cfg):
    return HistGradientBoostingRegressor(**cfg["subpixel"]["hgb"])


def run_fold(cfg, k, img, Z):
    sp = cfg["subpixel"]
    cands = [(dy, dx) for dy in sp["shift_candidates_dy"] for dx in sp["shift_candidates_dx"]]
    Fr3 = {j: Z[f"frac3_{j}"] for j in FIELDS}
    tr = [j for j in FIELDS if j != k]
    sh = best_shift(nd(img[7], img[5]), Fr3, tr, cands)
    F = spectral_stack(img, sh)
    D = {j: build_rows(F, Fr3[j], Z[f"frac1_{j}"], Z[f"o1_{j}"]) for j in FIELDS}
    Xtr = np.vstack([D[j][0] for j in tr])
    Ytr = np.concatenate([D[j][1] for j in tr])
    sel = np.random.default_rng(0).choice(len(Ytr), min(len(Ytr), sp["max_train_rows"]), replace=False)
    m = hgb(cfg).fit(Xtr[sel], Ytr[sel])
    # decision threshold on predicted fraction: inner leave-one-field-out on the training fields
    a, b = tr
    pb = hgb(cfg).fit(D[a][0], D[a][1]).predict(D[b][0])
    pa = hgb(cfg).fit(D[b][0], D[b][1]).predict(D[a][0])
    pin, yin = np.concatenate([pa, pb]), np.concatenate([D[a][1], D[b][1]]) >= 0.5
    T = np.round(np.arange(0.05, 0.9, 0.01), 2)
    t = float(T[np.argmax([f1_score(yin, pin >= x, zero_division=0) for x in T])])
    X, Y, _ = D[k]
    pred = m.predict(X)
    yt, yp = Y >= 0.5, pred >= t
    res = dict(test_field=k, shift=sh, threshold=t, F1_1m=round(f1_score(yt, yp), 3),
               precision_1m=round(precision_score(yt, yp), 3), recall_1m=round(recall_score(yt, yp), 3),
               R2_1m=round(r2_score(Y, np.clip(pred, 0, 1)), 3),
               area_bias=round(float(np.clip(pred, 0, 1).sum() / Y.sum()), 3),
               reference_cover=round(float(Y.mean()), 3), predicted_cover=round(float(np.clip(pred, 0, 1).mean()), 3))
    print(res, flush=True)

    # full 1 m prediction over the field window (for Figure 5)
    f1, o1 = Z[f"frac1_{k}"], Z[f"o1_{k}"]
    Hh, Ww = f1.shape
    Rg, Cg = np.meshgrid(np.arange(o1[0] // 3 - 1, (o1[0] + Hh) // 3 + 2), np.arange(o1[1] // 3 - 1, (o1[1] + Ww) // 3 + 2), indexing="ij")
    Rg, Cg = Rg.ravel(), Cg.ravel()
    base = neighbourhood(F, Rg, Cg)
    full = np.full((Hh, Ww), np.nan, np.float32)
    for i in range(3):
        for j in range(3):
            pos = np.zeros((len(Rg), 9), np.float32)
            pos[:, 3 * i + j] = 1
            q = np.clip(m.predict(np.hstack([base, pos])), 0, 1)
            rr, cc = 3 * Rg + i - o1[0], 3 * Cg + j - o1[1]
            ok = (rr >= 0) & (rr < Hh) & (cc >= 0) & (cc < Ww)
            full[rr[ok], cc[ok]] = q[ok]
    np.save(out(cfg, f"subpixel_pred_1m_F{k}.npy"), full)
    del D, F
    gc.collect()
    return res


def main():
    cfg = load_config()
    img, _ = load_scene(p(cfg, "scene_main"))
    Z = np.load(out(cfg, "reference_fractions.npz"))
    folds = [int(a) for a in sys.argv[1:]] or list(FIELDS)
    res = [run_fold(cfg, k, img, Z) for k in folds]
    if len(res) == 3:
        res.append(dict(mean_F1_1m=round(float(np.mean([r["F1_1m"] for r in res])), 3),
                        mean_area_bias=round(float(np.mean([r["area_bias"] for r in res])), 3)))
    save_json(res, out(cfg, "subpixel_lofo.json"))


if __name__ == "__main__":
    main()
