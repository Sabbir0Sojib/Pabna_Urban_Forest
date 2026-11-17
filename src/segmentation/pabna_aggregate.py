"""Pabna UAV tree segmentation: Phase 5 (metrics, uncertainty, paired comparison) + Phase 6 (outputs).

Reads finished runs (folders with per_tile_test.csv, test_summary.json, tuning.json, config.json) and writes
results_summary.csv, per_tile_metrics.csv, lofo_summary.csv, paired_comparison.csv, final_table.csv,
the example figure and a zip. Uses only test predictions already saved; never re-tunes anything.

Uncertainty: spatial BLOCK bootstrap. Neighbouring tiles overlap by 64 px and are spatially correlated, so
tiles are grouped into blocks of 3 x 3 tile positions (about 50 m x 50 m) and whole blocks are resampled
(1,000 resamples, fixed seed). A plain tile bootstrap is also reported for comparison.
"""
import os, sys, json, glob, argparse, zipfile, platform, datetime
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--runs_root", required=True, nargs="+")
p.add_argument("--splits_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--block", type=int, default=3)
# old leaky copy-level split (S0), test F1 from HF repos Sabbir12345/P6_S3_SegFormer_RGB_100ep (0.9439)
# and P6_S4_SegFormer_RGBExG_100ep (0.9449); see archive/hf_models/HF_OLD_MODELS_REPORT.md
p.add_argument("--old_leaky_f1_rgb", type=float, default=0.9439)
p.add_argument("--old_leaky_f1_rgbexg", type=float, default=0.9449)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
BOOT_SEED = 20261106
LOG = []
def log(m):
    print(m, flush=True); LOG.append(m)

# ------------------------------------------------------------------ load runs
runs = {}
for root in args.runs_root:
    for f in glob.glob(os.path.join(root, "**", "test_summary.json"), recursive=True):
        d = os.path.dirname(f)
        s = json.load(open(f))
        if s.get("smoke"):
            continue
        assert os.path.exists(os.path.join(d, "TEST_DONE.flag")), f"{d}: no TEST_DONE flag"
        if s["run_id"] in runs:
            log(f"WARNING duplicate run {s['run_id']} in {d}; keeping the first one")
            continue
        runs[s["run_id"]] = dict(dir=d, summary=s, tuning=json.load(open(os.path.join(d, "tuning.json"))),
                                 config=json.load(open(os.path.join(d, "config.json"))),
                                 tiles=pd.read_csv(os.path.join(d, "per_tile_test.csv")))
log(f"runs found: {len(runs)}: {sorted(runs)}")
assert runs, "no finished runs found"

# consistency checks: identical hyper-parameters across runs except input/split/seed
KEYS = ["pretrained", "img", "batch", "lr", "head_lr_mult", "wd", "total_steps", "eval_every", "warmup_frac", "gsd"]
ref = next(iter(runs.values()))["config"]
for rid, r in runs.items():
    diff = {k: (ref.get(k), r["config"].get(k)) for k in KEYS if ref.get(k) != r["config"].get(k)}
    assert not diff, f"{rid}: hyper-parameters differ from other runs: {diff}"
    t = r["tuning"]; t2 = dict(t); h = t2.pop("sha256")
    import hashlib
    assert hashlib.sha256(json.dumps(t2, sort_keys=True).encode()).hexdigest() == h, f"{rid}: tuning.json was modified after test"
log("hyper-parameters identical across all runs; tuning.json files untouched since the test")

# ------------------------------------------------------------------ helpers
def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
def iou(tp, fp, fn):
    return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
def kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 0.0
def objf1(m, g, pr):
    P = m / pr if pr else 0.0; R = m / g if g else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0
def stats(df):
    s = df[["TP", "FP", "FN", "TN", "matched_objects", "gt_objects", "pred_objects"]].sum()
    return dict(F1=f1(s.TP, s.FP, s.FN), IoU=iou(s.TP, s.FP, s.FN), objF1=objf1(s.matched_objects, s.gt_objects, s.pred_objects))

def add_block(df):
    df = df.copy()
    df["block"] = "F" + df.field.astype(str) + "_" + (df.r // args.block).astype(str) + "_" + (df.c // args.block).astype(str)
    return df

def boot_idx(units, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, len(units), len(units)) for _ in range(n)]

def ci(df, unit="block"):
    """95% percentile CI of pooled F1 / IoU / objF1, resampling blocks (or tiles) with replacement."""
    df = add_block(df)
    cols = ["TP", "FP", "FN", "TN", "matched_objects", "gt_objects", "pred_objects"]
    g = df.groupby(unit)[cols].sum()
    arr = g.values
    out = {"F1": [], "IoU": [], "objF1": []}
    for idx in boot_idx(g.index, args.n_boot, BOOT_SEED):
        s = arr[idx].sum(0)
        out["F1"].append(f1(s[0], s[1], s[2])); out["IoU"].append(iou(s[0], s[1], s[2])); out["objF1"].append(objf1(s[4], s[5], s[6]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}

# ------------------------------------------------------------------ per run table
rows, tiles_all = [], []
for rid, r in sorted(runs.items()):
    s, t, tl = r["summary"], r["tuning"], r["tiles"]
    cb, ct = ci(tl, "block"), ci(tl, "tile_id")
    fold = s["split"].replace("S2_fold", "") if s["split"].startswith("S2") else ""
    rows.append(dict(run=rid, model="SegFormer MiT-B2", input="RGB+ExG" if s["input"] == "rgbexg" else "RGB",
                     split="S1 tile-random" if s["split"] == "S1" else "S2 LOFO", fold=fold,
                     test_field=f"Field{fold}" if fold else "mixed", seed=s["seed"], seed_idx=r["config"]["seed_idx"], n_tiles=s["n_tiles"],
                     prevalence=s["prevalence"], TP=s["TP"], FP=s["FP"], FN=s["FN"], TN=s["TN"],
                     precision=s["precision"], recall=s["recall"], F1=s["F1"], IoU=s["IoU"], OA=s["OA"],
                     kappa=kappa(s["TP"], s["FP"], s["FN"], s["TN"]),
                     cover_true=(s["TP"] + s["FN"]) / (s["TP"] + s["FP"] + s["FN"] + s["TN"]),
                     cover_pred=(s["TP"] + s["FP"]) / (s["TP"] + s["FP"] + s["FN"] + s["TN"]),
                     F1_CI_low=cb["F1"][0], F1_CI_high=cb["F1"][1], IoU_CI_low=cb["IoU"][0], IoU_CI_high=cb["IoU"][1],
                     objF1=s["obj_F1"], objF1_CI_low=cb["objF1"][0], objF1_CI_high=cb["objF1"][1],
                     F1_CI_low_tileboot=ct["F1"][0], F1_CI_high_tileboot=ct["F1"][1],
                     obj_precision=s["obj_precision"], obj_recall=s["obj_recall"], gt_objects=s["gt_objects"],
                     pred_objects=s["pred_objects"], matched_objects=s["matched_objects"],
                     threshold=t["threshold"], A_min=t["A_min_m2"], TTA=t["tta"], best_epoch=s["best_old_epoch"],
                     valid_F1_tuned=t["valid_F1_tuned"], raw_F1_thr0p5=s["raw_F1_thr0p5"]))
    tiles_all.append(tl.assign(input=s["input"], split=s["split"], seed=s["seed"], seed_idx=r["config"]["seed_idx"]))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(args.out, "results_summary.csv"), index=False)
per_tile = pd.concat(tiles_all, ignore_index=True)
per_tile["F1"] = [f1(a, b, c) for a, b, c in zip(per_tile.TP, per_tile.FP, per_tile.FN)]
per_tile["IoU"] = [iou(a, b, c) for a, b, c in zip(per_tile.TP, per_tile.FP, per_tile.FN)]
per_tile.to_csv(os.path.join(args.out, "per_tile_metrics.csv"), index=False)
log(res[["run", "input", "split", "fold", "seed", "F1", "F1_CI_low", "F1_CI_high", "IoU", "objF1"]].to_string(index=False))

# ------------------------------------------------------------------ LOFO summaries
lofo = res[res.split == "S2 LOFO"]
summ_rows, final = [], []
if len(lofo):
    for inp, g in lofo.groupby("input"):
        per_fold = g.groupby("fold")[["F1", "IoU", "objF1"]].mean()          # mean over seeds within a fold
        d = dict(input=inp, n_runs=len(g), n_folds=len(per_fold), n_seeds=g.seed_idx.nunique(),
                 F1_mean=per_fold.F1.mean(), F1_sd_folds=per_fold.F1.std(ddof=1) if len(per_fold) > 1 else np.nan,
                 IoU_mean=per_fold.IoU.mean(), IoU_sd_folds=per_fold.IoU.std(ddof=1) if len(per_fold) > 1 else np.nan,
                 objF1_mean=per_fold.objF1.mean(), F1_min_fold=per_fold.F1.min(), F1_max_fold=per_fold.F1.max(),
                 F1_sd_seeds_within_fold=g.groupby("fold").F1.std(ddof=1).mean() if g.seed_idx.nunique() > 1 else np.nan)
        for k, row in per_fold.iterrows():
            d[f"F1_fold{k}"] = row.F1
        # pooled over all LOFO test tiles (each tile tested exactly once per seed), block CI per seed then averaged
        key = "rgbexg" if inp == "RGB+ExG" else "rgb"
        pooled = per_tile[(per_tile.input == key) & per_tile.split.str.startswith("S2")]
        ps = [stats(x)["F1"] for _, x in pooled.groupby("seed_idx")]
        d["F1_pooled_LOFO"] = float(np.mean(ps))
        summ_rows.append(d)
    lofo_summary = pd.DataFrame(summ_rows)
    lofo_summary.to_csv(os.path.join(args.out, "lofo_summary.csv"), index=False)
    log(lofo_summary.to_string(index=False))

# ------------------------------------------------------------------ paired comparison RGB+ExG minus RGB (S2)
pair_rows = []
pt = per_tile[per_tile.split.str.startswith("S2")]
if set(pt.input) == {"rgbexg", "rgb"}:
    for (split, seed), g in pt.groupby(["split", "seed"]):
        a, b = g[g.input == "rgbexg"], g[g.input == "rgb"]
        if len(a) and len(b):
            assert set(a.tile_id) == set(b.tile_id), "paired runs must share test tiles"
            sa, sb = stats(a), stats(b)
            pair_rows.append(dict(split=split, seed=seed, F1_rgbexg=sa["F1"], F1_rgb=sb["F1"], dF1=sa["F1"] - sb["F1"],
                                  IoU_rgbexg=sa["IoU"], IoU_rgb=sb["IoU"], dIoU=sa["IoU"] - sb["IoU"],
                                  dObjF1=sa["objF1"] - sb["objF1"]))
    pairs = pd.DataFrame(pair_rows)
    # block bootstrap of the pooled-LOFO difference (mean over seeds); same resampled blocks for both inputs
    pt_b = add_block(pt)
    cols = ["TP", "FP", "FN"]
    tabs = {(inp, sd): pt_b[(pt_b.input == inp) & (pt_b.seed_idx == sd)].groupby("block")[cols].sum()
            for inp in ("rgbexg", "rgb") for sd in sorted(pt_b.seed_idx.unique())}
    blocks = sorted(set.intersection(*[set(t.index) for t in tabs.values()]))
    diffs_f1, diffs_iou = [], []
    seed_list = sorted(pt_b.seed_idx.unique())
    for idx in boot_idx(blocks, args.n_boot, BOOT_SEED + 1):
        sel = [blocks[i] for i in idx]
        dF, dI = [], []
        for sd in seed_list:
            A = tabs[("rgbexg", sd)].loc[sel].values.sum(0); B = tabs[("rgb", sd)].loc[sel].values.sum(0)
            dF.append(f1(*A) - f1(*B)); dI.append(iou(*A) - iou(*B))
        diffs_f1.append(np.mean(dF)); diffs_iou.append(np.mean(dI))
    point_F = np.mean([f1(*tabs[("rgbexg", sd)].loc[blocks].values.sum(0)) - f1(*tabs[("rgb", sd)].loc[blocks].values.sum(0)) for sd in seed_list])
    point_I = np.mean([iou(*tabs[("rgbexg", sd)].loc[blocks].values.sum(0)) - iou(*tabs[("rgb", sd)].loc[blocks].values.sum(0)) for sd in seed_list])
    comp = dict(comparison="RGB+ExG minus RGB, S2 LOFO pooled (mean over seeds)", n_pairs=len(pairs),
                n_pairs_ExG_better=int((pairs.dF1 > 0).sum()), mean_dF1_over_pairs=float(pairs.dF1.mean()),
                sd_dF1_over_pairs=float(pairs.dF1.std(ddof=1)) if len(pairs) > 1 else np.nan,
                pooled_dF1=float(point_F), pooled_dF1_CI_low=float(np.percentile(diffs_f1, 2.5)),
                pooled_dF1_CI_high=float(np.percentile(diffs_f1, 97.5)),
                share_boot_dF1_gt0=float(np.mean(np.array(diffs_f1) > 0)),
                pooled_dIoU=float(point_I), pooled_dIoU_CI_low=float(np.percentile(diffs_iou, 2.5)),
                pooled_dIoU_CI_high=float(np.percentile(diffs_iou, 97.5)))
    pairs.to_csv(os.path.join(args.out, "paired_comparison_per_fold_seed.csv"), index=False)
    pd.DataFrame([comp]).to_csv(os.path.join(args.out, "paired_comparison.csv"), index=False)
    log(pairs.to_string(index=False)); log(json.dumps(comp, indent=1))

# ------------------------------------------------------------------ final table
final = [dict(row="S0 old leaky copy-level split (HF P6 repos)", input=inp, F1=v,
               note="NOT an independent test: augmented copies of the same tile in train and test")
         for inp, v in (("RGB", args.old_leaky_f1_rgb), ("RGB+ExG", args.old_leaky_f1_rgbexg))]
s1 = res[res.split == "S1 tile-random"]
for inp, g in s1.groupby("input"):
    final.append(dict(row="S1 tile-grouped random 80/10/10", input=inp, F1=g.F1.mean(), IoU=g.IoU.mean(),
                      objF1=g.objF1.mean(), F1_CI=f"{g.F1_CI_low.mean():.3f}-{g.F1_CI_high.mean():.3f}",
                      note="optimistic: neighbouring tiles share a 64 px strip with training tiles"))
for d in summ_rows:
    final.append(dict(row="S2 leave-one-field-out (main)", input=d["input"], F1=d["F1_mean"], F1_SD=d["F1_sd_folds"],
                      IoU=d["IoU_mean"], IoU_SD=d["IoU_sd_folds"], objF1=d["objF1_mean"],
                      note=f"mean of {d['n_folds']} folds (each the mean of {d['n_seeds']} seed(s)) +/- SD across folds"))
pd.DataFrame(final).to_csv(os.path.join(args.out, "final_table.csv"), index=False)
log(pd.DataFrame(final).to_string(index=False))

# ------------------------------------------------------------------ example figure (not cherry-picked)
try:
    coco_by_tile = {}
    for k in (1, 2, 3):
        fj = os.path.join(args.splits_dir, f"S2_fold{k}_test.json")
        if os.path.exists(fj):
            for im in json.load(open(fj))["images"]:
                coco_by_tile[im["tile_id"]] = im
    show = per_tile[(per_tile.input == "rgbexg") & per_tile.split.str.startswith("S2") & (per_tile.seed_idx == 0)]
    show = show[(show.TP + show.FN) / show.valid_px > 0.05]          # tiles with >5% tree, then random (seeded)
    picks = []
    rng = np.random.default_rng(BOOT_SEED)
    for k, g in show.groupby("field"):
        picks += list(g.iloc[rng.choice(len(g), min(3, len(g)), replace=False)].itertuples())
    run_dirs = {r["summary"]["split"] + "|" + r["summary"]["input"] + "|" + str(r["summary"]["seed"]): r["dir"] for r in runs.values()}
    if picks:
        fig, axes = plt.subplots(len(picks), 4, figsize=(16, 4 * len(picks)), squeeze=False)
        for row, t in zip(axes, picks):
            im = coco_by_tile[t.tile_id]
            img = cv2.cvtColor(cv2.imread(im["file_name"]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (im["width"], im["height"])) if img.shape[:2] != (im["height"], im["width"]) else img
            from pycocotools import mask as mu
            gt = np.zeros(img.shape[:2], np.uint8)
            js = json.load(open(os.path.join(args.splits_dir, f"S2_fold{t.field}_test.json")))
            for a in js["annotations"]:
                if a["image_id"] == im["id"] and a["category_id"] == 1:
                    seg = a["segmentation"]
                    if isinstance(seg, dict):
                        rle = dict(seg); rle["counts"] = rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
                        m = mu.decode(rle if isinstance(rle["counts"], bytes) else mu.frPyObjects(rle, *img.shape[:2]))
                    else:
                        m = mu.decode(mu.merge(mu.frPyObjects([q for q in seg if len(q) >= 6], *img.shape[:2])))
                    gt |= (m.max(2) if m.ndim == 3 else m).astype(np.uint8)
            d = run_dirs[f"{t.split}|rgbexg|{t.seed}"]
            pr = cv2.imread(os.path.join(d, "test_pred_masks", f"{t.tile_id}.png"), 0) > 0
            g_ = gt.astype(bool)
            err = np.zeros((*g_.shape, 3)); err[g_ & pr] = (0.85, 0.85, 0.85); err[~g_ & pr] = (1, 0, 0); err[g_ & ~pr] = (0, 0.3, 1)
            row[0].imshow(img); row[0].set_title(f"{t.tile_id} (Field{t.field} held out)")
            row[1].imshow(g_, cmap="Greens"); row[1].set_title("ground truth")
            row[2].imshow(pr, cmap="Greens"); row[2].set_title(f"prediction  F1 {t.F1:.2f}")
            row[3].imshow(err); row[3].set_title("errors: FP red, FN blue, TP grey")
            for a_ in row: a_.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig_examples_LOFO.png"), dpi=100); plt.close()
        log(f"example figure: {[t.tile_id for t in picks]} (random among tiles with >5% tree, seed {BOOT_SEED})")
except Exception as e:
    import traceback; log("example figure FAILED: " + traceback.format_exc())

with open(os.path.join(args.out, "environment_aggregate.txt"), "w") as f:
    f.write(f"python {platform.python_version()}\nnumpy {np.__version__}\npandas {pd.__version__}\nopencv {cv2.__version__}\n")
with open(os.path.join(args.out, "aggregate_log.md"), "w") as f:
    f.write("# Phase 5 aggregate log\n\n```\n" + "\n".join(LOG) + "\n```\n")
zp = os.path.join(args.out, "phase5_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for fpath in glob.glob(os.path.join(args.out, "*")):
        if not fpath.endswith(".zip"):
            z.write(fpath, os.path.basename(fpath))
log(f"done: {zp}")
