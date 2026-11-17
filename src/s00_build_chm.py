"""Step 0. Canopy height model (Equation 1): CHM = UAV DSM - UAV ground-only DTM.

For each field, on a 0.25 m grid that covers the SegFormer canopy mask:
  * DSM: maximum of the 5.3 cm DSM values in each cell (crown tops are kept)
  * DTM: bilinear sample of the 10 cm ground-only DTM at the cell centre
  * canopy fraction: mean of the SegFormer canopy mask (1 tree, 0 non-tree) in each cell
Both surfaces come from the same Metashape project, so there is no date, resolution or
vertical-datum mismatch between them (the 12.5 m ALOS PALSAR DEM is no longer used).

Output: outputs/chm_F{k}.npz with chm_max, chm_mean, frac, X0, Y0, RES.
"""
import sys

import numpy as np
from scipy import ndimage

from utils import load_config, out, p, read_raster


def bin_to_grid(a, g, X0, Y0, RES, H, W, want_max=False):
    """Mean (and optionally maximum) of fine raster values inside each coarse cell."""
    h, w = a.shape
    ys = g["y0"] - (np.arange(h) + 0.5) * g["sy"]
    xs = g["x0"] + (np.arange(w) + 0.5) * g["sx"]
    R = np.floor((Y0 - ys) / RES).astype(int)
    C = np.floor((xs - X0) / RES).astype(int)
    okc = (C >= 0) & (C < W)
    s = np.zeros((H, W))
    n = np.zeros((H, W))
    mx = np.full((H, W), -np.inf, np.float32)
    for i in range(h):
        r = R[i]
        if r < 0 or r >= H:
            continue
        v = a[i]
        ok = okc & np.isfinite(v)
        np.add.at(s[r], C[ok], v[ok])
        np.add.at(n[r], C[ok], 1)
        if want_max:
            np.maximum.at(mx[r], C[ok], v[ok])
    mean = (s / np.where(n > 0, n, np.nan)).astype(np.float32)
    return mean, n, mx


def build(cfg, k):
    RES = cfg["grid"]["chm_resolution_m"]
    dsm, gd = read_raster(p(cfg, "dsm", k))
    dsm = dsm.astype(np.float32)
    dsm[dsm < -1000] = np.nan
    dtm, gt = read_raster(p(cfg, "dtm", k))
    dtm = dtm.astype(np.float32)
    dtm[dtm < -1000] = np.nan
    m, gm = read_raster(p(cfg, "canopy_mask", k))

    X0, Y0 = gm["x0"], gm["y0"]
    W = int(m.shape[1] * gm["sx"] / RES)
    H = int(m.shape[0] * gm["sy"] / RES)

    dsm_mean, nd, dsm_max = bin_to_grid(dsm, gd, X0, Y0, RES, H, W, want_max=True)
    del dsm

    yy = Y0 - (np.arange(H) + 0.5) * RES
    xx = X0 + (np.arange(W) + 0.5) * RES
    ri = (gt["y0"] - yy) / gt["sy"] - 0.5
    ci = (xx - gt["x0"]) / gt["sx"] - 0.5
    coords = np.meshgrid(ri, ci, indexing="ij")
    valid_dtm = ndimage.map_coordinates(np.isfinite(dtm).astype(np.float32), coords, order=0, cval=0) > 0
    dtm_f = np.where(np.isfinite(dtm), dtm, np.nanmedian(dtm))
    dtm_r = ndimage.map_coordinates(dtm_f, coords, order=1, cval=np.nan).astype(np.float32)

    mm = np.where(m == 255, np.nan, m.astype(np.float32))
    del m
    frac, _, _ = bin_to_grid(mm, gm, X0, Y0, RES, H, W)

    chm = dsm_max - dtm_r
    chm[~valid_dtm | ~np.isfinite(chm) | (nd == 0)] = np.nan
    chm_mean = dsm_mean - dtm_r
    chm_mean[~valid_dtm] = np.nan
    np.savez_compressed(out(cfg, f"chm_F{k}.npz"), chm_max=chm.astype(np.float32),
                        chm_mean=chm_mean.astype(np.float32), frac=frac, X0=X0, Y0=Y0, RES=RES)
    tree = frac >= 0.5
    print(f"Field {k}: grid {H} x {W}; CHM percentiles in canopy cells (5,25,50,75,95):",
          np.nanpercentile(chm[tree], [5, 25, 50, 75, 95]).round(2), flush=True)


if __name__ == "__main__":
    cfg = load_config()
    for k in ([int(a) for a in sys.argv[1:]] or [1, 2, 3]):
        build(cfg, k)
