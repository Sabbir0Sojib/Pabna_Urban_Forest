"""Step 5. Indicative city-wide canopy map and ward statistics.

  * The sub-pixel model of step 4 is refitted on all three fields (130,000 random 1 m rows per
    field; shift chosen on all fields) and applied to every 1 m cell of the 15 wards.
  * 3.12 ha of Ward 6 are missing in the 12 January scene; they are filled from a full-coverage
    composite of the same date before prediction.
  * City and ward cover are corrected with the area bias of each held-out field from step 4:
    middle = predicted / mean bias, low = predicted / largest bias, high = predicted / smallest bias.
    One factor is applied to all wards, so the ranges describe the cover level, not the ward ranking.
  * Optional robustness check with another date:  python src/s05_city_map.py --check

Outputs: outputs/city_canopy_fraction_1m.tif (percent, 255 = outside), outputs/city_canopy_corrected.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from s04_subpixel_lofo import neighbourhood, spectral_stack
from utils import best_shift, load_config, load_polygons, load_scene, nd, out, p, rasterize, save_json, write_raster

FIELDS = (1, 2, 3)


def main():
    cfg = load_config()
    sp = cfg["subpixel"]
    check = "--check" in sys.argv
    img, G = load_scene(p(cfg, "scene_main"))
    H, W = img.shape[1:]
    if check:
        img, _ = load_scene(p(cfg, "scene_check"), G, (H, W))
    else:
        fill, _ = load_scene(p(cfg, "scene_fill"), G, (H, W))
        gap = (img == 0).all(0)
        img[:, gap] = fill[:, gap]
        print("pixels filled from the composite:", int(gap.sum()), flush=True)

    Z = np.load(out(cfg, "reference_fractions.npz"))
    Fr3 = {k: Z[f"frac3_{k}"] for k in FIELDS}
    cands = [(dy, dx) for dy in sp["shift_candidates_dy"] for dx in sp["shift_candidates_dx"]]
    sh = best_shift(nd(img[7], img[5]), Fr3, list(FIELDS), cands)
    F = spectral_stack(img, sh)

    rng = np.random.default_rng(0)
    XS, YS = [], []
    n = sp["city_rows_per_field"]
    for k in FIELDS:
        f1, o1 = Z[f"frac1_{k}"], Z[f"o1_{k}"]
        R, C = np.where(np.isfinite(Fr3[k]))
        base = neighbourhood(F, R, C).astype(np.float32)
        pi, pp = rng.integers(0, len(R), n), rng.integers(0, 9, n)
        i, j = pp // 3, pp % 3
        rr, cc = 3 * R[pi] + i - o1[0], 3 * C[pi] + j - o1[1]
        inb = (rr >= 0) & (rr < f1.shape[0]) & (cc >= 0) & (cc < f1.shape[1])
        y = np.full(n, np.nan)
        y[inb] = f1[rr[inb], cc[inb]]
        pos = np.zeros((n, 9), np.float32)
        pos[np.arange(n), pp] = 1
        Xk = np.hstack([base[pi], pos])
        ok = np.isfinite(y)
        XS.append(Xk[ok])
        YS.append(y[ok])
    model = HistGradientBoostingRegressor(**sp["hgb"]).fit(np.vstack(XS), np.concatenate(YS))
    del XS, YS

    ward = np.zeros((H, W), np.uint8)
    for props, rings in load_polygons(p(cfg, "wards")):
        ward[rasterize(rings, G, (H, W))] = int(props["id"])
    sub = np.full((H * 3, W * 3), np.nan, np.float32)
    rows = np.where((ward > 0).any(1))[0]
    for r0 in range(rows.min(), rows.max() + 1, 4):
        r1 = min(r0 + 4, rows.max() + 1)
        R, C = np.where(ward[r0:r1] > 0)
        if len(R) == 0:
            continue
        R = np.clip(R + r0, 2, H - 3)
        C = np.clip(C, 2, W - 3)
        base = neighbourhood(F, R, C)
        for i in range(3):
            for j in range(3):
                pos = np.zeros((len(R), 9), np.float32)
                pos[:, 3 * i + j] = 1
                sub[3 * R + i, 3 * C + j] = np.clip(model.predict(np.hstack([base, pos])), 0, 1)

    w1 = np.repeat(np.repeat(ward, 3, 0), 3, 1)
    inside = w1 > 0
    tag = "_check" if check else ""
    g1 = dict(x0=G["x0"], y0=G["y0"], sx=1.0, sy=1.0)
    write_raster(out(cfg, f"city_canopy_fraction_1m{tag}.tif"),
                 np.where(inside & np.isfinite(sub), np.round(100 * sub), 255).astype(np.uint8), g1, nodata=255)

    lofo = json.load(open(out(cfg, "subpixel_lofo.json")))
    bias = [r["area_bias"] for r in lofo if "area_bias" in r]
    b_mid, b_lo, b_hi = float(np.mean(bias)), max(bias), min(bias)
    city = 100 * float(np.nanmean(sub[inside]))
    area = inside.sum() / 1e4
    res = dict(shift=sh, area_bias=bias,
               city=dict(area_ha=round(area, 2), predicted_pct=round(city, 1), predicted_ha=round(city / 100 * area),
                         middle_pct=round(city / b_mid, 1), low_pct=round(city / b_lo, 1), high_pct=round(city / b_hi, 1),
                         middle_ha=round(city / b_mid / 100 * area), low_ha=round(city / b_lo / 100 * area),
                         high_ha=round(city / b_hi / 100 * area)),
               wards={})
    for w in range(1, cfg["study_area"]["n_wards"] + 1):
        s = sub[w1 == w]
        pc = 100 * float(np.nanmean(s))
        res["wards"][w] = dict(area_ha=round(s.size / 1e4, 2), predicted_pct=round(pc, 1), middle_pct=round(pc / b_mid, 1),
                               low_pct=round(pc / b_lo, 1), high_pct=round(pc / b_hi, 1))
    wa = [res["wards"][w]["area_ha"] for w in res["wards"]]
    wc = [res["wards"][w]["predicted_pct"] for w in res["wards"]]
    rho, pv = spearmanr(wa, wc)
    res["spearman_cover_vs_area"] = dict(rho=round(float(rho), 2), p=round(float(pv), 2))
    save_json(res, out(cfg, f"city_canopy_corrected{tag}.json"))
    print(json.dumps(res["city"], indent=1))
    for w in sorted(res["wards"], key=lambda w: -res["wards"][w]["predicted_pct"]):
        print(w, res["wards"][w])


if __name__ == "__main__":
    main()
