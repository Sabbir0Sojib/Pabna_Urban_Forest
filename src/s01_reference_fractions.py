"""Step 1. UAV canopy as reference fractions on the satellite grids.

  * 3 m: share of UAV reference-mask pixels labelled tree inside each PlanetScope pixel
    (used to train and test the 3 m classifiers; label = tree if fraction >= rule)
  * 1 m: share of SegFormer canopy-mask pixels labelled tree inside each 1 m cell of the same
    grid subdivided 3 x 3 (used to train and test the sub-pixel model)
Only cells covered by at least 95 % UAV pixels are kept; the rest are NaN.

Output: outputs/reference_fractions.npz
  frac3_{k}: (H, W) on the 3 m grid of the 12 January scene
  frac1_{k}: 1 m window for field k, with its upper-left offset o1_{k} = (row, col) on the 1 m grid
"""
import numpy as np

from utils import load_config, out, p, read_raster


def fraction_3m(mask, g, G, H, W, full_share):
    h, w = mask.shape
    cols = g["x0"] + (np.arange(w) + 0.5) * g["sx"]
    rows = g["y0"] - (np.arange(h) + 0.5) * g["sy"]
    ci = np.floor((cols - G["x0"]) / G["sx"]).astype(int)
    ri = np.floor((G["y0"] - rows) / G["sy"]).astype(int)
    s = np.zeros((H, W))
    n = np.zeros((H, W))
    for r in range(h):
        np.add.at(s, (ri[r], ci), mask[r])
        np.add.at(n, (ri[r], ci), 1)
    with np.errstate(invalid="ignore"):
        f = s / n
    full = (G["sx"] / g["sx"]) * (G["sy"] / g["sy"])
    f[n < full_share * full] = np.nan
    return f.astype(np.float32)


def fraction_1m(mask, g, G, full_share):
    """1 m grid = the 3 m grid subdivided 3 x 3 (same origin)."""
    h, w = mask.shape
    ys = g["y0"] - (np.arange(h) + 0.5) * g["sy"]
    xs = g["x0"] + (np.arange(w) + 0.5) * g["sx"]
    R = np.floor(G["y0"] - ys).astype(int)
    C = np.floor(xs - G["x0"]).astype(int)
    r0, r1, c0, c1 = R.min(), R.max() + 1, C.min(), C.max() + 1
    tree = np.zeros((r1 - r0, c1 - c0))
    val = np.zeros_like(tree)
    cnt = np.zeros_like(tree)
    Rr, Cc = R - r0, C - c0
    for i in range(h):
        row = mask[i]
        ok = row != 255
        np.add.at(cnt[Rr[i]], Cc, 1)
        np.add.at(val[Rr[i]], Cc[ok], 1)
        np.add.at(tree[Rr[i]], Cc[ok & (row == 1)], 1)
    f = np.where(val > full_share * cnt.max(), tree / np.maximum(val, 1), np.nan).astype(np.float32)
    return f, (int(r0), int(c0))


def main():
    cfg = load_config()
    share = cfg["grid"]["full_cover_share"]
    scene, G = read_raster(p(cfg, "scene_main"))
    H, W = scene.shape[1:]
    del scene
    outd = {}
    for k in (1, 2, 3):
        m3, g3 = read_raster(p(cfg, "reference_mask_3m", k))
        f3 = fraction_3m(m3.astype(np.float32), g3, G, H, W, share)
        m1, g1 = read_raster(p(cfg, "canopy_mask", k))
        f1, o1 = fraction_1m(m1, g1, G, share)
        outd[f"frac3_{k}"], outd[f"frac1_{k}"], outd[f"o1_{k}"] = f3, f1, np.array(o1)
        v = f3[np.isfinite(f3)]
        print(f"Field {k}: 3 m cells {v.size} ({v.size * 9 / 1e4:.2f} ha), mean fraction {v.mean():.3f}; "
              f"1 m cells {np.isfinite(f1).sum()}, cover {np.nanmean(f1):.3f}", flush=True)
    np.savez_compressed(out(cfg, "reference_fractions.npz"), **outd)


if __name__ == "__main__":
    main()
