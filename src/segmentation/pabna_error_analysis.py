"""Pabna UAV tree segmentation: error analysis + ExG thresholding baseline (CPU only, no training).

Uses ONLY saved test predictions (test_pred_masks) and the raw annotated tiles. Nothing is re-tuned on test data.

1. Rebuilds ground truth exactly as pabna_train.py (category 1 polygons/RLE, padding excluded via pad_h/pad_w)
   and checks TP/FP/FN per tile against each run's per_tile_test.csv (guards against a reconstruction error).
2. Attributes every false-positive and false-negative pixel to one cause, in this priority order:
     FP: crown edge (within 3 px = 8 cm of a ground-truth crown) > shadow > green non-tree vegetation > other
     FN: crown edge > shadow > small crown (< 5 m2) > other
   Heuristics (stated in the paper as such): shadow = luminance Y < 60/255;
   green vegetation = chromatic excess green 2g - r - b > 0.05 (g = G/(R+G+B), Woebbecke et al. 1995).
3. Crown-size analysis: per ground-truth crown (8-connected component), pixel recall and detection
   (recall >= 0.5) by size class.
4. ExG baseline (no deep learning), same ExG as the model input: ((2G - R - B)/255 + 1) / 3.
     a) per-tile Otsu threshold (unsupervised, uses no labels);
     b) one global threshold per fold chosen on that fold's VALIDATION tiles only, then applied once to test.
5. Figures: random examples (seeded, tiles with > 5% tree) and, separately, diagnostic worst-tile figures.
"""
import os, glob, json, argparse, collections
import numpy as np, pandas as pd, cv2
from pycocotools import mask as mask_utils
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--runs_root", required=True, nargs="+")
p.add_argument("--splits_dir", required=True)
p.add_argument("--raw_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--gsd", type=float, default=0.026)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
SEED = 20261106
EDGE_PX, SHADOW_Y, GREEN_EXG, SMALL_M2 = 3, 60, 0.05, 5.0
SMALL_PX = int(round(SMALL_M2 / args.gsd ** 2))
SIZE_BINS = [(0, 5), (5, 20), (20, 50), (50, 1e9)]          # m2
LOG = []
def log(m):
    print(m, flush=True); LOG.append(str(m))

# ------------------------------------------------------------------ raw file index
raw_index = {os.path.basename(f): f for f in glob.glob(os.path.join(args.raw_dir, "**", "*.jpg"), recursive=True)}
log(f"raw images found locally: {len(raw_index)}")

def load_split(fold, subset):
    coco = json.load(open(os.path.join(args.splits_dir, f"S2_fold{fold}_{subset}.json")))
    by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == 1:
            by_img[a["image_id"]].append(a)
    out = []
    for im in coco["images"]:
        out.append(dict(tile_id=im["tile_id"], field=im["field"], r=im["r"], c=im["c"],
                        path=raw_index[os.path.basename(im["file_name"])], H=im["height"], W=im["width"],
                        pad_w=im.get("pad_w", 0), pad_h=im.get("pad_h", 0), anns=by_img.get(im["id"], [])))
    return out

def decode(seg, H, W):                                        # identical to pabna_train.py
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, H, W)
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode()
        m = mask_utils.decode(rle)
    else:
        polys = [q for q in seg if len(q) >= 6]
        if not polys:
            return np.zeros((H, W), np.uint8)
        m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polys, H, W)))
    return (m.max(axis=2) if m.ndim == 3 else m).astype(np.uint8)

def load_item(it):                                            # identical to pabna_train.py
    im = cv2.cvtColor(cv2.imread(it["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (it["H"], it["W"]):
        im = cv2.resize(im, (it["W"], it["H"]), interpolation=cv2.INTER_AREA)
    gt = np.zeros((it["H"], it["W"]), np.uint8)
    for a in it["anns"]:
        gt |= decode(a["segmentation"], it["H"], it["W"])
    valid = np.zeros_like(gt, dtype=bool)
    valid[: it["H"] - it["pad_h"], : it["W"] - it["pad_w"]] = True
    return im, gt.astype(bool), valid

def exg_model(im):
    f = im.astype(np.float32) / 255.0
    return ((2 * f[..., 1] - f[..., 0] - f[..., 2]) + 1.0) / 3.0

def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")

# ------------------------------------------------------------------ runs (LOFO only)
runs = {}
for root in args.runs_root:
    for f in glob.glob(os.path.join(root, "**", "test_summary.json"), recursive=True):
        s = json.load(open(f))
        if s.get("smoke") or not s["split"].startswith("S2"):
            continue
        d = os.path.dirname(f)
        runs[s["run_id"]] = dict(dir=d, fold=int(s["split"][-1]), input=s["input"],
                                 seed_idx=json.load(open(os.path.join(d, "config.json")))["seed_idx"],
                                 tiles=pd.read_csv(os.path.join(d, "per_tile_test.csv")).set_index("tile_id"))
log(f"LOFO runs: {len(runs)}: {sorted(runs)}")

# ------------------------------------------------------------------ main loop over test tiles
feat_rows, attr_rows, crown_rows, check_bad = [], [], [], 0
otsu_conf = collections.defaultdict(lambda: np.zeros(3, np.int64))   # fold -> TP, FP, FN for per-tile Otsu
test_cache_hist = {}
bins = np.linspace(0.0, 1.0, 501)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_PX + 1, 2 * EDGE_PX + 1))
for fold in (1, 2, 3):
    fold_runs = {rid: r for rid, r in runs.items() if r["fold"] == fold}
    hist_tree = np.zeros(len(bins) - 1); hist_non = np.zeros(len(bins) - 1)
    test_items = load_split(fold, "test")
    for it in test_items:
        im, gt, va = load_item(it)
        Y = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
        s = im.astype(np.float32).sum(2) + 1e-6
        chrom_exg = (2 * im[..., 1] - im[..., 0] - im[..., 2]) / s
        shadow = Y < SHADOW_Y
        green = chrom_exg > GREEN_EXG
        edge_out = cv2.dilate(gt.astype(np.uint8), k3).astype(bool) & ~gt
        edge_in = gt & ~cv2.erode(gt.astype(np.uint8), k3).astype(bool)
        ncc, lab, stats, _ = cv2.connectedComponentsWithStats((gt & va).astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        small_mask = np.isin(lab, 1 + np.where(areas < SMALL_PX)[0])
        exg = exg_model(im)
        # per-tile Otsu baseline (unsupervised)
        e8 = np.clip(exg * 255, 0, 255).astype(np.uint8)
        thr_o, _ = cv2.threshold(e8[va].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        po = (e8 > thr_o) & va
        otsu_conf[fold] += [np.sum(po & gt), np.sum(po & ~gt), np.sum(~po & gt & va)]
        # store test ExG histograms for the validation-tuned global threshold (applied later, once)
        ht, _ = np.histogram(exg[gt & va], bins); hn, _ = np.histogram(exg[~gt & va], bins)
        test_cache_hist.setdefault(fold, [np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)])
        test_cache_hist[fold][0] += ht; test_cache_hist[fold][1] += hn
        nva = va.sum()
        feat_rows.append(dict(tile_id=it["tile_id"], field=it["field"], r=it["r"], c=it["c"], valid_px=int(nva),
                              tree_frac=(gt & va).sum() / nva, shadow_frac=(shadow & va).sum() / nva,
                              green_nontree_frac=(green & ~gt & va).sum() / nva,
                              n_crowns=int(ncc - 1), mean_crown_m2=float(areas.mean() * args.gsd ** 2) if len(areas) else 0.0,
                              crowns_per_ha=(ncc - 1) / (nva * args.gsd ** 2 / 1e4)))
        for rid, r in fold_runs.items():
            pr = cv2.imread(os.path.join(r["dir"], "test_pred_masks", f"{it['tile_id']}.png"), 0) > 0
            tp, fp, fn = np.sum(pr & gt & va), np.sum(pr & ~gt & va), np.sum(~pr & gt & va)
            ref = r["tiles"].loc[it["tile_id"]]
            if (tp, fp, fn) != (ref.TP, ref.FP, ref.FN):
                check_bad += 1
            FPm, FNm = pr & ~gt & va, ~pr & gt & va
            fp_edge = FPm & edge_out; rest = FPm & ~edge_out
            fp_sh = rest & shadow; rest &= ~shadow
            fp_gr = rest & green; fp_ot = rest & ~green
            fn_edge = FNm & edge_in; rest = FNm & ~edge_in
            fn_sh = rest & shadow; rest &= ~shadow
            fn_sm = rest & small_mask; fn_ot = rest & ~small_mask
            attr_rows.append(dict(run=rid, fold=fold, input=r["input"], seed_idx=r["seed_idx"], tile_id=it["tile_id"],
                                  TP=int(tp), FP=int(fp), FN=int(fn),
                                  FP_edge=int(fp_edge.sum()), FP_shadow=int(fp_sh.sum()), FP_green_nontree=int(fp_gr.sum()),
                                  FP_other=int(fp_ot.sum()), FN_edge=int(fn_edge.sum()), FN_shadow=int(fn_sh.sum()),
                                  FN_small_crown=int(fn_sm.sum()), FN_other=int(fn_ot.sum())))
            hit = np.bincount(lab[pr], minlength=ncc)[1:]
            for a, h in zip(areas, hit):
                crown_rows.append(dict(run=rid, fold=fold, input=r["input"], area_m2=a * args.gsd ** 2, recall=h / a))
    # validation histograms for the global ExG threshold of this fold
    for it in load_split(fold, "valid"):
        im, gt, va = load_item(it)
        exg = exg_model(im)
        hist_tree += np.histogram(exg[gt & va], bins)[0]; hist_non += np.histogram(exg[~gt & va], bins)[0]
    tp_c = np.cumsum(hist_tree[::-1])[::-1]; fp_c = np.cumsum(hist_non[::-1])[::-1]; fn_c = hist_tree.sum() - tp_c
    f1s = 2 * tp_c / np.maximum(2 * tp_c + fp_c + fn_c, 1)
    k = int(np.argmax(f1s)); thr = bins[k]
    ht, hn = test_cache_hist[fold]
    TP = ht[k:].sum(); FP = hn[k:].sum(); FN = ht[:k].sum()
    otsu = otsu_conf[fold]
    test_cache_hist[fold] = dict(fold=fold, valid_tuned_threshold=float(thr), valid_F1_at_threshold=float(f1s[k]),
                                 test_F1_valid_tuned=f1(TP, FP, FN), test_IoU_valid_tuned=TP / (TP + FP + FN),
                                 test_cover_true=(TP + FN) / (ht.sum() + hn.sum()), test_cover_pred_valid_tuned=(TP + FP) / (ht.sum() + hn.sum()),
                                 test_F1_per_tile_otsu=f1(*otsu), test_IoU_per_tile_otsu=otsu[0] / otsu.sum())
    log(f"fold {fold}: done; ExG baseline {test_cache_hist[fold]}")

log(f"GT reconstruction check: {check_bad} tile-run mismatches against per_tile_test.csv (must be 0)")
feat = pd.DataFrame(feat_rows); feat.to_csv(os.path.join(args.out, "tile_features.csv"), index=False)
attr = pd.DataFrame(attr_rows); attr.to_csv(os.path.join(args.out, "error_attribution_per_tile.csv"), index=False)
base = pd.DataFrame([test_cache_hist[k] for k in (1, 2, 3)]); base.to_csv(os.path.join(args.out, "exg_baseline.csv"), index=False)
log(base.round(4).to_string(index=False))

# ------------------------------------------------------------------ summaries
cols_fp = ["FP_edge", "FP_shadow", "FP_green_nontree", "FP_other"]; cols_fn = ["FN_edge", "FN_shadow", "FN_small_crown", "FN_other"]
g = attr.groupby(["fold", "input"])[["TP", "FP", "FN"] + cols_fp + cols_fn].sum()
summ = g.copy()
for c in cols_fp:
    summ[c + "_share"] = g[c] / g.FP
for c in cols_fn:
    summ[c + "_share"] = g[c] / g.FN
summ["FP_minus_FN_pct_of_valid"] = (g.FP - g.FN) / feat.groupby("field").valid_px.sum().reindex(g.index.get_level_values(0)).values
summ = summ.reset_index(); summ.to_csv(os.path.join(args.out, "error_attribution_summary.csv"), index=False)
log(summ[["fold", "input"] + [c + "_share" for c in cols_fp + cols_fn]].round(3).to_string(index=False))

cr = pd.DataFrame(crown_rows)
cr["size_class"] = pd.cut(cr.area_m2, [b[0] for b in SIZE_BINS] + [1e9], right=False,
                          labels=[f"{a}-{b if b < 1e9 else 'inf'} m2" for a, b in SIZE_BINS])
cs = cr.groupby(["fold", "input", "size_class"], observed=True).agg(n=("recall", "size"), mean_pixel_recall=("recall", "mean"),
                                                                    detected_share=("recall", lambda x: float((x >= 0.5).mean()))).reset_index()
cs.to_csv(os.path.join(args.out, "crown_size_recall.csv"), index=False)
log(cs.groupby(["input", "size_class"], observed=True)[["n", "mean_pixel_recall", "detected_share"]].mean().round(3).to_string())

# per-tile F1 (mean over runs) vs tile features: Spearman, tiles with >= 1% tree
pt = attr.groupby("tile_id")[["TP", "FP", "FN"]].sum()
pt["F1"] = 2 * pt.TP / (2 * pt.TP + pt.FP + pt.FN).replace(0, np.nan)
m = feat.set_index("tile_id").join(pt.F1)
m = m[m.tree_frac >= 0.01]
corr = m[["F1", "tree_frac", "shadow_frac", "green_nontree_frac", "mean_crown_m2", "crowns_per_ha"]].corr(method="spearman")["F1"]
corr.to_csv(os.path.join(args.out, "tile_F1_feature_spearman.csv"))
log(f"Spearman correlation of per-tile F1 with tile features (n={len(m)} tiles with >= 1% tree):\n{corr.round(3).to_string()}")

# ------------------------------------------------------------------ figures
def panel(axrow, it, rid, title_extra=""):
    im, gt, va = load_item(it)
    pr = cv2.imread(os.path.join(runs[rid]["dir"], "test_pred_masks", f"{it['tile_id']}.png"), 0) > 0
    err = np.zeros((*gt.shape, 3)); err[gt & pr] = (0.8, 0.8, 0.8); err[~gt & pr & va] = (1, 0, 0); err[gt & ~pr & va] = (0, 0.35, 1)
    t = runs[rid]["tiles"].loc[it["tile_id"]]
    axrow[0].imshow(im); axrow[0].set_title(f"{it['tile_id']} (Field {it['field']} held out){title_extra}", fontsize=9)
    axrow[1].imshow(gt, cmap="Greens"); axrow[1].set_title("ground truth", fontsize=9)
    axrow[2].imshow(pr, cmap="Greens"); axrow[2].set_title(f"prediction (F1 {f1(t.TP, t.FP, t.FN):.2f})", fontsize=9)
    axrow[3].imshow(err); axrow[3].set_title("grey TP, red FP, blue FN", fontsize=9)
    for a in axrow: a.axis("off")

items_by_tile = {it["tile_id"]: it for k in (1, 2, 3) for it in load_split(k, "test")}
rgbexg_s0 = {r["fold"]: rid for rid, r in runs.items() if r["input"] == "rgbexg" and r["seed_idx"] == 0}
rng = np.random.default_rng(SEED)
picks = []
for fold in (1, 2, 3):
    cand = feat[(feat.field == fold) & (feat.tree_frac > 0.05)].tile_id.tolist()
    picks += [(t, rgbexg_s0[fold]) for t in rng.choice(cand, min(3, len(cand)), replace=False)] if fold in rgbexg_s0 else []
if picks:
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.6 * len(picks)), squeeze=False)
    for row, (t, rid) in zip(axes, picks):
        panel(row, items_by_tile[t], rid)
    fig.suptitle("Random test tiles (seeded; tiles with > 5% tree), RGB+ExG seed 0, each field held out", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.985]); plt.savefig(os.path.join(args.out, "fig_examples_random_LOFO.png"), dpi=110); plt.close()
    log(f"random example tiles: {[t for t, _ in picks]}")

for fold, kind in ((1, "FP"), (2, "FN"), (3, "FP")):
    if fold not in rgbexg_s0:
        continue
    rid = rgbexg_s0[fold]
    a = attr[attr.run == rid].copy()
    a["px"] = a[kind]
    worst = a.sort_values("px", ascending=False).head(4).tile_id.tolist()
    fig, axes = plt.subplots(len(worst), 4, figsize=(14, 3.6 * len(worst)), squeeze=False)
    for row, t in zip(axes, worst):
        panel(row, items_by_tile[t], rid, f"  {kind} px {int(a.set_index('tile_id').loc[t, kind])}")
    fig.suptitle(f"DIAGNOSTIC (not representative): Field {fold} tiles with the most {kind} pixels, RGB+ExG seed 0", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(os.path.join(args.out, f"fig_diagnostic_worst_{kind}_field{fold}.png"), dpi=110); plt.close()
    log(f"diagnostic worst {kind} Field {fold}: {worst}")

with open(os.path.join(args.out, "error_analysis_log.md"), "w") as f:
    f.write("# Error analysis + ExG baseline log\n\n```\n" + "\n".join(LOG) + "\n```\n")
log("done")
