"""Step 3. Individual crowns by marker-controlled watershed on the UAV canopy height model.

  * canopy = 0.25 m cells with SegFormer canopy fraction >= 0.5 and CHM >= 2 m
  * the CHM is Gaussian-smoothed (sigma), local maxima at least `min_distance` apart are markers
  * watershed of the negative smoothed CHM inside the canopy gives one segment per crown
  * segments < 1 m2 are dropped; crowns with p95 height > 25 m are flagged and removed
    (they fell on building edges or orthomosaic gaps when inspected)

The marker distance and smoothing are chosen against the 1,320 manually delineated reference
crowns (share of reference crowns matched at IoU >= 0.5). The reference crowns are used only for
this choice and for validation, never for training.

Outputs: outputs/crown_param_scan.json, outputs/crown_validation.json, outputs/crowns_all_fields.csv
"""
import json

import numpy as np
from scipy import ndimage
from skimage.draw import polygon as skpoly
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils import load_config, out, p, save_json, to_utm45


def load_chm(cfg, k):
    z = np.load(out(cfg, f"chm_F{k}.npz"))
    return z["chm_max"], z["frac"], float(z["X0"]), float(z["Y0"]), float(z["RES"])


def reference_raster(cfg, k, X0, Y0, RES, shape):
    G = json.load(open(p(cfg, "reference_crowns"), encoding="utf-8"))
    lab = np.zeros(shape, np.int32)
    n = 0
    for f in G["features"]:
        if f["properties"]["field"] != f"Field{k}":
            continue
        ring = np.array(f["geometry"]["coordinates"][0])
        x, y = to_utm45(ring[:, 0], ring[:, 1])
        rr, cc = skpoly((Y0 - y) / RES, (x - X0) / RES, shape)
        n += 1
        lab[rr, cc] = n
    return lab, n


def segment(chm, frac, RES, min_dist_m, sigma_m, cc):
    can = (frac >= cc["canopy_fraction_min"]) & (chm >= cc["min_height_m"])
    s = ndimage.gaussian_filter(np.nan_to_num(chm), sigma_m / RES)
    s[~can] = 0
    pk = peak_local_max(s, min_distance=max(1, int(round(min_dist_m / RES))), threshold_abs=cc["min_height_m"],
                        labels=can.astype(int))
    markers = np.zeros(chm.shape, np.int32)
    markers[pk[:, 0], pk[:, 1]] = np.arange(1, len(pk) + 1)
    return watershed(-s, markers, mask=can)


def best_iou(lab, ref, nref):
    """Best IoU of each reference crown with any predicted crown, and the matched crown id."""
    g = ref.ravel()
    q = lab.ravel()
    sel = (g > 0) & (q > 0)
    u, cnt = np.unique(np.stack([g[sel], q[sel]], 1), axis=0, return_counts=True)
    ga = np.bincount(g, minlength=nref + 1)
    qa = np.bincount(q)
    best = np.zeros(nref + 1)
    who = np.zeros(nref + 1, int)
    for (gi, qi), c in zip(u, cnt):
        iou = c / (ga[gi] + qa[qi] - c)
        if iou > best[gi]:
            best[gi], who[gi] = iou, qi
    return best[1:], who[1:], ga[1:], qa


def crown_table(cfg, k, chm, frac, X0, Y0, RES):
    cc = cfg["crowns"]
    lab = segment(chm, frac, RES, cc["marker_min_distance_m"], cc["smoothing_sigma_m"], cc)
    idx = np.arange(1, lab.max() + 1)
    area = ndimage.sum(np.ones_like(chm), lab, idx) * RES * RES
    h95 = np.array(ndimage.labeled_comprehension(np.nan_to_num(chm), lab, idx,
                                                 lambda v: np.percentile(v, cc["height_percentile"]), float, 0))
    cm = np.array(ndimage.center_of_mass(np.ones_like(chm), lab, idx))
    cx = X0 + (cm[:, 1] + 0.5) * RES
    cy = Y0 - (cm[:, 0] + 0.5) * RES
    flagged = (area >= cc["min_area_m2"]) & (h95 > cc["max_height_m"])
    keep = (area >= cc["min_area_m2"]) & (h95 <= cc["max_height_m"])
    return lab, dict(x=cx[keep], y=cy[keep], area=area[keep], cd=2 * np.sqrt(area[keep] / np.pi), h95=h95[keep],
                     n_flagged=int(flagged.sum()))


def main():
    cfg = load_config()
    cc = cfg["crowns"]
    D = {}
    for k in (1, 2, 3):
        chm, frac, X0, Y0, RES = load_chm(cfg, k)
        ref, n = reference_raster(cfg, k, X0, Y0, RES, chm.shape)
        D[k] = (chm, frac, X0, Y0, RES, ref, n)

    # parameter scan against the reference crowns
    scan = {}
    for md in cc["scan_min_distance_m"]:
        for sg in cc["scan_sigma_m"]:
            row = []
            for k in (1, 2, 3):
                chm, frac, X0, Y0, RES, ref, n = D[k]
                b, _, _, _ = best_iou(segment(chm, frac, RES, md, sg, cc), ref, n)
                row.append(round(float((b >= cc["iou_match"]).mean()), 3))
            scan[f"{md}_{sg}"] = row
            print("min_distance", md, "sigma", sg, "share of reference crowns matched (F1, F2, F3):", row, flush=True)
    save_json(scan, out(cfg, "crown_param_scan.json"))

    # crowns with the chosen parameters + validation
    val = {}
    rows = []
    for k in (1, 2, 3):
        chm, frac, X0, Y0, RES, ref, n = D[k]
        lab, t = crown_table(cfg, k, chm, frac, X0, Y0, RES)
        b, who, ga, qa = best_iou(lab, ref, n)
        m = b >= cc["iou_match"]
        dg = 2 * np.sqrt(ga[m] * RES * RES / np.pi)
        dq = 2 * np.sqrt(qa[who[m]] * RES * RES / np.pi)
        val[k] = dict(crowns=int(len(t["h95"])), flagged_removed=t["n_flagged"],
                      reference_matched_share=round(float(m.mean()), 3),
                      diameter_r=round(float(np.corrcoef(dg, dq)[0, 1]), 3),
                      diameter_bias_m=round(float((dq - dg).mean()), 2),
                      diameter_rmse_m=round(float(np.sqrt(((dq - dg) ** 2).mean())), 2),
                      mean_height_m=round(float(t["h95"].mean()), 2), max_height_m=round(float(t["h95"].max()), 2),
                      mean_crown_diameter_m=round(float(t["cd"].mean()), 2))
        print(f"Field {k}:", val[k], flush=True)
        rows.append(np.c_[np.full(len(t["h95"]), k), t["x"], t["y"], t["area"], t["cd"], t["h95"]])
    save_json(val, out(cfg, "crown_validation.json"))
    np.savetxt(out(cfg, "crowns_all_fields.csv"), np.vstack(rows), delimiter=",", comments="", fmt="%.3f",
               header="field,x_utm45n,y_utm45n,area_m2,crown_diam_m,h95_m")


if __name__ == "__main__":
    main()
