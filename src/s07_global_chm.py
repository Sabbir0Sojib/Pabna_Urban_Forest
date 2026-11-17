"""Step 7. Global canopy height maps (Meta/WRI v1 and CHMv2) in the independent field test.

Each map is first resampled to a 1 m grid aligned with the study grid (see data/README.md).
For each held-out field the height threshold (0.5-4 m) and a shift (up to 12 m, in 2 m steps)
are chosen on the two training fields; the held-out field is scored once against the 1 m
SegFormer canopy fraction (canopy if >= 0.5).

Output: outputs/global_chm_lofo.json
"""
import numpy as np
from scipy import ndimage
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import load_config, load_scene, out, p, read_raster, save_json

FIELDS = (1, 2, 3)
SHIFTS = [(dy, dx) for dy in range(-12, 13, 2) for dx in range(-12, 13, 2)]
HEIGHTS = [0.5, 1, 2, 3, 4]


def evaluate(chm, gc, G, Z):
    dr = int(round(gc["y0"] - G["y0"]))     # CHM row of study-grid row 0
    dc = int(round(G["x0"] - gc["x0"]))     # CHM column of study-grid column 0
    win = {}
    for k in FIELDS:
        f, o = Z[f"frac1_{k}"], Z[f"o1_{k}"]
        r0, c0 = o[0] + dr, o[1] + dc
        Hh, Ww = f.shape
        win[k] = (f, chm[r0 - 20:r0 + Hh + 20, c0 - 20:c0 + Ww + 20])

    def pairs(k, sh):
        f, c = win[k]
        Hh, Ww = f.shape
        c = ndimage.shift(c, sh, order=1, mode="nearest") if sh != (0, 0) else c
        c = c[20:20 + Hh, 20:20 + Ww]
        ok = np.isfinite(f)
        return f[ok], c[ok]

    res = []
    for k in FIELDS:
        tr = [j for j in FIELDS if j != k]
        best = None
        for sh in SHIFTS:
            P = {j: pairs(j, sh) for j in tr}
            for h in HEIGHTS:
                s = np.mean([f1_score(P[j][0] >= 0.5, P[j][1] >= h) for j in tr])
                if best is None or s > best[0]:
                    best = (s, sh, h)
        f, c = pairs(k, best[1])
        yt, yp = f >= 0.5, c >= best[2]
        res.append(dict(test_field=k, shift_m=best[1], height_m=best[2], F1=round(f1_score(yt, yp), 3),
                        precision=round(precision_score(yt, yp), 3), recall=round(recall_score(yt, yp), 3),
                        area_ratio=round(float(yp.sum() / yt.sum()), 2)))
        print(res[-1], flush=True)
    return res


def main():
    cfg = load_config()
    _, G = load_scene(p(cfg, "scene_main"))
    Z = np.load(out(cfg, "reference_fractions.npz"))
    allres = {}
    for name in cfg["paths"]["global_chm"]:
        chm, gc = read_raster(p(cfg, "global_chm", name))
        chm = np.nan_to_num(chm.astype(np.float32))
        print(name, flush=True)
        r = evaluate(chm, gc, G, Z)
        allres[name] = dict(folds=r, mean_F1=round(float(np.mean([x["F1"] for x in r])), 3))
    save_json(allres, out(cfg, "global_chm_lofo.json"))


if __name__ == "__main__":
    main()
