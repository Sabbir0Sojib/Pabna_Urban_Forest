"""Explain check (a): compare, on the annotated test tiles, the frozen per-seed LOFO tile masks with the
stitched ensemble mosaic, separately for the tile interior and the 64 px overlap band at tile edges.
Local, CPU only, reads only files that already exist (no new predictions)."""
import os, sys, json, glob
import numpy as np, cv2, rasterio
from pycocotools import mask as mu

ROOT = sys.argv[1]
OUTD = os.path.join(ROOT, "archive", "canopy_maps", "kaggle_v1", "canopy_maps")
SPL = os.path.join(ROOT, "archive", "kaggle_run_v4", "results", "splits")
TILE, STEP, BAND = 640, 576, 64

def decode(seg, H, W):
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle["counts"], list):
            rle = mu.frPyObjects(rle, H, W)
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode()
        m = mu.decode(rle)
    else:
        polys = [q for q in seg if len(q) >= 6]
        if not polys:
            return np.zeros((H, W), np.uint8)
        m = mu.decode(mu.merge(mu.frPyObjects(polys, H, W)))
    return (m.max(axis=2) if m.ndim == 3 else m).astype(np.uint8)

def f1(c):
    tp, fp, fn = c
    return 2 * tp / max(2 * tp + fp + fn, 1)

rows = []
for k in (1, 2, 3):
    coco = json.load(open(os.path.join(SPL, f"S2_fold{k}_test.json")))
    by = {}
    for a in coco["annotations"]:
        if a["category_id"] == 1:
            by.setdefault(a["image_id"], []).append(a)
    with rasterio.open(os.path.join(OUTD, f"Field{k}_canopy_mask.tif")) as d:
        mos = d.read(1)
    seed_dirs = sorted(glob.glob(os.path.join(ROOT, "archive", "kaggle_train_*", "runs", f"S2_fold{k}_rgb_s*")))
    assert len(seed_dirs) == 3, seed_dirs
    acc = {n: {"interior": np.zeros(3, np.int64), "edge_band": np.zeros(3, np.int64), "all": np.zeros(3, np.int64)}
           for n in ["stitched"] + [os.path.basename(s) for s in seed_dirs]}
    for im in coco["images"]:
        H, W = im["height"], im["width"]
        h, w = H - im.get("pad_h", 0), W - im.get("pad_w", 0)
        gt = np.zeros((H, W), np.uint8)
        for a in by.get(im["id"], []):
            gt |= decode(a["segmentation"], H, W)
        g = gt[:h, :w].astype(bool)
        y, x = im["r"] * STEP, im["c"] * STEP
        band = np.zeros((h, w), bool)
        band[:BAND, :] = band[-BAND:, :] = band[:, :BAND] = band[:, -BAND:] = True
        preds = {"stitched": mos[y:y + h, x:x + w] == 1}
        for s in seed_dirs:
            p = cv2.imread(os.path.join(s, "test_pred_masks", f"{im['tile_id']}.png"), cv2.IMREAD_GRAYSCALE)
            assert p is not None, (s, im["tile_id"])
            preds[os.path.basename(s)] = p[:h, :w] > 0
        for n, p in preds.items():
            for reg, sel in (("interior", ~band), ("edge_band", band), ("all", np.ones_like(band))):
                acc[n][reg] += [(p & g & sel).sum(), (p & ~g & sel).sum(), (~p & g & sel).sum()]
    for n, d in acc.items():
        rows.append(dict(field=k, prediction=n, F1_all=f1(d["all"]), F1_interior=f1(d["interior"]), F1_edge_band=f1(d["edge_band"])))
import csv
out = os.path.join(ROOT, "archive", "canopy_maps", "check_a_decomposition.csv")
with open(out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
for r in rows:
    print(f"Field {r['field']}  {r['prediction']:22s}  all {r['F1_all']:.4f}  interior {r['F1_interior']:.4f}  edge band {r['F1_edge_band']:.4f}")
print("wrote", out)
