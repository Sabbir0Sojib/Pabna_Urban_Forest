"""Pabna UAV: full-field canopy maps from the RGB leave-one-field-out (LOFO) SegFormer MiT-B2 models.
INFERENCE ONLY. Nothing is trained, tuned or re-selected here.

For field k (k = 1, 2, 3):
  * models = the 3 RGB LOFO runs of fold k (S2_fold{k}_rgb_s0/s1/s2); they never saw field k
  * every tile of Field{k}_Ortho is rebuilt exactly as in the original tiling notebook
    (640 px, 64 px overlap -> step 576, origin top-left, right/bottom edge tiles black-padded,
    JPEG quality 95 round trip), so all tiles are predicted, not only the annotated ones
  * per model: same preprocessing and TTA as pabna_train.py predict_prob (resize to 512, ImageNet norm,
    flips if TTA was chosen in that model's tuning.json); probabilities of the 3 models are averaged
  * stitching: tile probabilities are averaged in overlaps (plain mean)
  * mask: prob >= mean of the 3 tuned thresholds, then components smaller than the mean tuned
    A_min (pixels) are removed (8-connectivity), on the stitched mosaic
  * outputs on the ortho grid (same CRS, transform, size): Field{k}_canopy_prob.tif (float32, nodata -1)
    and Field{k}_canopy_mask.tif (uint8: 1 tree, 0 non-tree, 255 nodata)
Checks: tile geometry against the annotated Roboflow tiles, reproduction of each model's frozen LOFO
test F1, ensemble F1 on the original tiles, F1 of the stitched mosaic on the annotated test tiles,
canopy cover, quicklooks.
"""
import os, sys, glob, json, io, time, hashlib, datetime, platform, zipfile, subprocess, traceback
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
try:
    import rasterio
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)
    import rasterio
from rasterio.windows import Window
from pycocotools import mask as mask_utils

CFG = dict(
    input_root="/kaggle/input", out="/kaggle/working/canopy_maps", fields=[1, 2, 3],
    tile=640, overlap=64, img=512, jpeg_q=95, batch=12, pretrained="nvidia/mit-b2", tiny_model=False,
    ortho_size={"1": [15844, 14657], "2": [16060, 16930], "3": [16738, 15157]},   # W, H from tiling notebook log
    n_tiles={"1": 728, "2": 840, "3": 783}, expected_epsg=32645, check_sha=True, quick_px=3000,
)
if os.environ.get("CANOPY_CFG"):                      # local smoke test only
    CFG.update(json.load(open(os.environ["CANOPY_CFG"])))
if os.environ.get("CANOPY_EXPECTED_JSON"):           # local smoke test only
    EXPECTED = json.load(open(os.environ["CANOPY_EXPECTED_JSON"]))
else:
    EXPECTED = EXPECTED_RUNS  # noqa: F821  (archived tuning sha256 / thresholds / frozen test F1, prepended by the notebook)

TILE, OVERLAP, IMG = CFG["tile"], CFG["overlap"], CFG["img"]
STEP = TILE - OVERLAP
OUT = CFG["out"]
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"
LOGF = os.path.join(OUT, "canopy_log.md")
def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(f"- {line}\n")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()

def find(pattern):
    return sorted(glob.glob(os.path.join(CFG["input_root"], pattern), recursive=True))

env = dict(started=datetime.datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
           torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__, rasterio=rasterio.__version__,
           gdal=rasterio.__gdal_version__, device=str(DEV),
           gpu=torch.cuda.get_device_name(0) if AMP else "cpu", cfg=CFG)
try:
    import transformers
    env["transformers"] = transformers.__version__
except Exception:
    pass
json.dump(env, open(os.path.join(OUT, "environment.json"), "w"), indent=1)
log(f"START canopy maps; device {env['gpu']}; tile {TILE}, overlap {OVERLAP}, step {STEP}; models RGB LOFO only")

# ---------------------------------------------------------------- model (identical to pabna_train.py)
from transformers import SegformerForSemanticSegmentation, SegformerConfig
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
def to_tensor(rgb_u8):
    f = rgb_u8.astype(np.float32) / 255.0
    return torch.from_numpy(((f - MEAN) / STD).transpose(2, 0, 1).copy())

def build_model():
    if CFG["tiny_model"]:
        conf = SegformerConfig(num_labels=2, hidden_sizes=[16, 32, 48, 64], depths=[1, 1, 1, 1],
                               decoder_hidden_size=32, num_attention_heads=[1, 2, 3, 4])
        return SegformerForSemanticSegmentation(conf)
    return SegformerForSemanticSegmentation.from_pretrained(CFG["pretrained"], num_labels=2,
                                                            ignore_mismatched_sizes=True)

def load_run(rid):
    exp = EXPECTED[rid]
    hits = find(f"**/runs/{rid}/best.pt")
    assert len(hits) == 1, f"{rid}: expected exactly one best.pt, found {hits}"
    rdir = os.path.dirname(hits[0])
    tun = json.load(open(os.path.join(rdir, "tuning.json")))
    if CFG["check_sha"]:
        assert tun["sha256"] == exp["sha256"], f"{rid}: tuning.json sha256 differs from the archived run"
    for k in ("threshold", "A_min_px", "tta"):
        assert tun[k] == exp[k], f"{rid}: tuning {k} differs"
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)
    assert int(ck["step"]) == int(tun["checkpoint_step"]), f"{rid}: checkpoint step {ck['step']} != tuning {tun['checkpoint_step']}"
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["model"].items()}
    m = build_model()
    m.load_state_dict(sd, strict=True)
    m.to(DEV).eval()
    info = dict(run_id=rid, session=exp.get("session"), ckpt=hits[0], ckpt_sha256=sha256_file(hits[0]),
                ckpt_step=int(ck["step"]), ckpt_valid_F1=float(ck.get("valid_F1", float("nan"))),
                tuning_sha256=tun["sha256"], threshold=tun["threshold"], A_min_px=tun["A_min_px"],
                A_min_m2=tun["A_min_m2"], tta=tun["tta"], frozen_test_F1=exp["test_F1"])
    log(f"loaded {rid} ({info['session']}): step {info['ckpt_step']}, thr {info['threshold']}, "
        f"A_min {info['A_min_px']} px, TTA {info['tta']}, tuning sha256 matches archive")
    return m, info

@torch.no_grad()
def predict_batch(model, tiles_u8, tta):
    """Batched version of pabna_train.predict_prob (same resize, normalisation, TTA, upsampling)."""
    x = torch.stack([to_tensor(cv2.resize(t, (IMG, IMG), interpolation=cv2.INTER_LINEAR)) for t in tiles_u8]).to(DEV)
    views = [x] + ([torch.flip(x, [3]), torch.flip(x, [2])] if tta else [])
    probs = []
    for k, v in enumerate(views):
        with torch.autocast("cuda", enabled=AMP):
            out = model(pixel_values=v).logits
            lg = F.interpolate(out, size=v.shape[-2:], mode="bilinear", align_corners=False)
        pr = torch.softmax(lg.float(), 1)[:, 1:2]
        if k == 1: pr = torch.flip(pr, [3])
        if k == 2: pr = torch.flip(pr, [2])
        probs.append(pr)
    pr = torch.stack(probs).mean(0)
    pr = F.interpolate(pr, size=(TILE, TILE), mode="bilinear", align_corners=False)
    return pr[:, 0].float().cpu().numpy()

def ensemble_prob(models, tiles):
    acc = None
    for m, info in models:
        p = predict_batch(m, tiles, info["tta"])
        acc = p if acc is None else acc + p
    return acc / len(models)

def remove_small(pred, amin_px):
    if amin_px <= 0:
        return pred
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= amin_px
    return keep[lab]

def decode(seg, H, W):
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

def load_test_items(k):
    hits = find(f"**/splits/S2_fold{k}_test.json")
    assert hits, f"split file S2_fold{k}_test.json not found"
    hs = {sha256_file(h) for h in hits}
    assert len(hs) == 1, f"different copies of S2_fold{k}_test.json: {hits}"
    coco = json.load(open(hits[0]))
    by = {}
    for a in coco["annotations"]:
        if a["category_id"] == 1:
            by.setdefault(a["image_id"], []).append(a)
    items = []
    for im in coco["images"]:
        fp = im["file_name"]
        if not os.path.exists(fp):
            h = find(f"**/{os.path.basename(fp)}")
            assert len(h) == 1, f"cannot locate {fp}"
            fp = h[0]
        items.append(dict(tile_id=im["tile_id"], field=im["field"], r=im["r"], c=im["c"], path=fp,
                          H=im["height"], W=im["width"], pad_w=im.get("pad_w", 0), pad_h=im.get("pad_h", 0),
                          anns=by.get(im["id"], [])))
    return items, hits[0]

def load_item(it):
    im = cv2.cvtColor(cv2.imread(it["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (it["H"], it["W"]):
        im = cv2.resize(im, (it["W"], it["H"]), interpolation=cv2.INTER_AREA)
    gt = np.zeros((it["H"], it["W"]), np.uint8)
    for a in it["anns"]:
        gt |= decode(a["segmentation"], it["H"], it["W"])
    valid = np.zeros_like(gt, dtype=bool)
    valid[: it["H"] - it["pad_h"], : it["W"] - it["pad_w"]] = True
    return im, gt, valid

def conf(tp, fp, fn, tn=0):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn), precision=p, recall=r,
                F1=2 * p * r / max(p + r, 1e-12), IoU=tp / max(tp + fp + fn, 1))

def tile_tensor(img, y, x):
    """Exactly the tiling notebook: crop, paste into a black 640x640 canvas, save JPEG q95."""
    H, W = img.shape[:2]
    t = np.zeros((TILE, TILE, 3), np.uint8)
    c = img[y:min(y + TILE, H), x:min(x + TILE, W)]
    t[:c.shape[0], :c.shape[1]] = c
    buf = io.BytesIO()
    Image.fromarray(t).save(buf, format="JPEG", quality=CFG["jpeg_q"])
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))

def write_tif(path, arr, ref, dtype, nodata, predictor):
    prof = dict(driver="GTiff", width=ref.width, height=ref.height, count=1, dtype=dtype, nodata=nodata,
                crs=ref.crs, transform=ref.transform, tiled=True, blockxsize=512, blockysize=512,
                compress="DEFLATE", predictor=predictor, ZLEVEL=6, BIGTIFF="IF_SAFER", NUM_THREADS="ALL_CPUS")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)

summary_rows = []
for k in CFG["fields"]:
    t_field = time.time()
    rep = dict(field=k)
    try:
        log(f"===== FIELD {k} =====")
        # ------------------------------------------------ models of fold k (never saw field k)
        rids = [f"S2_fold{k}_rgb_s{j}" for j in range(3)]
        models = [load_run(r) for r in rids]
        infos = [i for _, i in models]
        thr = float(np.mean([i["threshold"] for i in infos]))
        amin_px = int(round(np.mean([i["A_min_px"] for i in infos])))
        amin_m2 = float(np.mean([i["A_min_m2"] for i in infos]))
        rep.update(models=infos, threshold_mean=thr, A_min_px_mean=amin_px, A_min_m2_mean=amin_m2,
                   tta_per_model=[i["tta"] for i in infos])
        log(f"field {k}: ensemble threshold {thr:.4f} (mean of {[i['threshold'] for i in infos]}), "
            f"A_min {amin_px} px = {amin_m2:.3f} m2 (mean), TTA {[i['tta'] for i in infos]}")

        # ------------------------------------------------ ortho
        hits = find(f"**/Field{k}_Ortho.tif")
        assert len(hits) >= 1, f"Field{k}_Ortho.tif not found"
        opath = hits[0]
        ds = rasterio.open(opath)
        W, H = ds.width, ds.height
        assert [W, H] == CFG["ortho_size"][str(k)], f"ortho size {W}x{H} != tiling log {CFG['ortho_size'][str(k)]}"
        epsg = ds.crs.to_epsg() if ds.crs else None
        rep.update(ortho=opath, ortho_W=W, ortho_H=H, bands=ds.count, dtypes=list(ds.dtypes),
                   crs=ds.crs.to_string() if ds.crs else None, epsg=epsg, transform=list(ds.transform)[:6],
                   pixel_size_x=ds.transform.a, pixel_size_y=-ds.transform.e,
                   georeferenced=ds.crs is not None and not ds.transform.is_identity,
                   epsg_matches_expected=(epsg == CFG["expected_epsg"]))
        log(f"ortho {os.path.basename(opath)}: {W}x{H}, {ds.count} bands {ds.dtypes}, CRS {rep['crs']} "
            f"(EPSG {epsg}), pixel {ds.transform.a:.4f} x {-ds.transform.e:.4f}, georeferenced {rep['georeferenced']}")
        if not rep["georeferenced"]:
            log("WARNING: ortho has no CRS/transform; outputs are written on the pixel grid only (GEOREF_MISSING)")
        if epsg != CFG["expected_epsg"]:
            log(f"WARNING: ortho EPSG {epsg} != expected {CFG['expected_epsg']}; outputs keep the ortho's own grid/CRS (no reprojection)")
        t0 = time.time()
        img = np.asarray(Image.open(opath).convert("RGB"))        # exactly how the tiling notebook read it
        assert img.shape[:2] == (H, W)
        datamask = ds.dataset_mask() > 0                          # alpha / nodata / internal mask
        rep["valid_data_fraction"] = float(datamask.mean())
        if ds.count >= 3 and ds.dtypes[0] == "uint8":
            w = Window(W // 2, H // 2, 256, 256)
            rb = np.transpose(ds.read([1, 2, 3], window=w), (1, 2, 0))
            rep["pil_equals_rasterio_rgb"] = bool(np.array_equal(rb, img[H // 2:H // 2 + 256, W // 2:W // 2 + 256]))
        log(f"ortho read in {time.time() - t0:.0f}s; valid-data fraction {rep['valid_data_fraction']:.4f}; "
            f"PIL RGB == rasterio bands 1-3: {rep.get('pil_equals_rasterio_rgb')}")

        # ------------------------------------------------ grid (tiling notebook) + geometry verification
        ys = list(range(0, H - OVERLAP, STEP)); xs = list(range(0, W - OVERLAP, STEP))
        assert len(ys) * len(xs) == CFG["n_tiles"][str(k)], f"{len(ys)}x{len(xs)} tiles != tiling log {CFG['n_tiles'][str(k)]}"
        rep.update(grid_rows=len(ys), grid_cols=len(xs), n_tiles=len(ys) * len(xs))
        items, split_file = load_test_items(k)
        assert all(it["field"] == k for it in items)
        rep.update(split_file=split_file, n_annotated_test_tiles=len(items))
        geo_mae, pad_ok = [], 0
        for it in items:
            y, x = it["r"] * STEP, it["c"] * STEP
            h, w = TILE - it["pad_h"], TILE - it["pad_w"]
            pad_ok += int(min(TILE, H - y) == h and min(TILE, W - x) == w)
            rf = cv2.cvtColor(cv2.imread(it["path"]), cv2.COLOR_BGR2RGB)
            if rf.shape[:2] != (TILE, TILE):
                continue
            geo_mae.append(float(np.abs(rf[:h, :w].astype(np.int16) - img[y:y + h, x:x + w].astype(np.int16)).mean()))
        rng = np.random.default_rng(20261107 + k)
        full = [it for it in items if it["pad_w"] == 0 and it["pad_h"] == 0]
        sub = [full[i] for i in rng.choice(len(full), size=min(25, len(full)), replace=False)]
        shifts = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
        shift_sum = np.zeros((len(shifts), len(shifts))); shift_n = np.zeros_like(shift_sum)
        alt = {"step_576_overlap_64 (used)": [], "step_640_no_overlap": [], "step_512_overlap_128": []}
        for it in sub:
            rf = cv2.cvtColor(cv2.imread(it["path"]), cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = lambda y0, x0: cv2.cvtColor(np.ascontiguousarray(img[y0:y0 + 512, x0:x0 + 512]), cv2.COLOR_RGB2GRAY).astype(np.float32)
            y, x = it["r"] * STEP, it["c"] * STEP
            for a, dy in enumerate(shifts):
                for b, dx in enumerate(shifts):
                    yy, xx = y + 64 + dy, x + 64 + dx
                    if 0 <= yy and 0 <= xx and yy + 512 <= H and xx + 512 <= W:
                        shift_sum[a, b] += np.abs(rf[64:576, 64:576] - g(yy, xx)).mean(); shift_n[a, b] += 1
            for name, st in (("step_576_overlap_64 (used)", 576), ("step_640_no_overlap", 640), ("step_512_overlap_128", 512)):
                yy, xx = it["r"] * st + 64, it["c"] * st + 64
                if yy + 512 <= H and xx + 512 <= W:
                    alt[name].append(float(np.abs(rf[64:576, 64:576] - g(yy, xx)).mean()))
        shift_tab = np.where(shift_n > 0, shift_sum / np.maximum(shift_n, 1), np.nan)
        best = np.unravel_index(np.nanargmin(shift_tab), shift_tab.shape)
        nz = shift_tab.copy(); nz[shifts.index(0), shifts.index(0)] = np.nan
        rep["geometry"] = dict(
            mean_abs_diff_roboflow_vs_ortho_at_rc_offset_median=float(np.median(geo_mae)),
            mean_abs_diff_max=float(np.max(geo_mae)), n_tiles_compared=len(geo_mae),
            padding_consistent_tiles=pad_ok, padding_total=len(items),
            best_shift_dy_dx=[shifts[best[0]], shifts[best[1]]],
            mad_at_zero_shift=float(shift_tab[shifts.index(0), shifts.index(0)]),
            mad_best_nonzero_shift=float(np.nanmin(nz)), n_tiles_shift_test=len(sub),
            alternative_grids_median_mad={n: (float(np.median(v)) if v else None) for n, v in alt.items()})
        np.savetxt(os.path.join(OUT, f"Field{k}_geometry_shift_mad.csv"), shift_tab, delimiter=",", fmt="%.3f",
                   header="rows dy, cols dx = " + " ".join(map(str, shifts)))
        log(f"geometry: Roboflow tile vs ortho crop at (r*{STEP}, c*{STEP}): median MAD {rep['geometry']['mean_abs_diff_roboflow_vs_ortho_at_rc_offset_median']:.2f} DN "
            f"(max {rep['geometry']['mean_abs_diff_max']:.2f}); best shift {rep['geometry']['best_shift_dy_dx']}; "
            f"padding consistent {pad_ok}/{len(items)}; other grids {rep['geometry']['alternative_grids_median_mad']}")
        assert rep["geometry"]["best_shift_dy_dx"] == [0, 0], "tile geometry check failed"
        assert pad_ok == len(items), "padding check failed"

        # ------------------------------------------------ reproduction + ensemble on the ORIGINAL annotated tiles
        loaded = [load_item(it) for it in items]
        per_seed = [np.zeros(3, np.int64) for _ in models]
        ens = np.zeros(4, np.int64)
        for s in range(0, len(items), CFG["batch"]):
            chunk = loaded[s:s + CFG["batch"]]
            ims = [c[0] for c in chunk]
            probs_m = [predict_batch(m, ims, i["tta"]) for m, i in models]
            for j, (im, gt, va) in enumerate(chunk):
                g = gt.astype(bool) & va
                for mi, (_, inf) in enumerate(models):
                    pr = remove_small((probs_m[mi][j] >= inf["threshold"]) & va, inf["A_min_px"])
                    per_seed[mi] += [(pr & g).sum(), (pr & ~g & va).sum(), (~pr & g).sum()]
                pe = remove_small((np.mean([p[j] for p in probs_m], 0) >= thr) & va, amin_px)
                ens += [(pe & g).sum(), (pe & ~g & va).sum(), (~pe & g).sum(), (~pe & ~g & va).sum()]
        rep["reproduction"] = []
        for (_, inf), c in zip(models, per_seed):
            f1 = conf(*c)["F1"]
            rep["reproduction"].append(dict(run_id=inf["run_id"], frozen_test_F1=inf["frozen_test_F1"], reproduced_F1=f1,
                                            diff=f1 - inf["frozen_test_F1"]))
            log(f"reproduction {inf['run_id']}: frozen LOFO test F1 {inf['frozen_test_F1']:.4f}, re-computed {f1:.4f} (diff {f1 - inf['frozen_test_F1']:+.4f})")
        rep["ensemble_on_original_tiles"] = conf(*ens)
        lofo_mean = float(np.mean([i["frozen_test_F1"] for i in infos]))
        rep["lofo_test_F1_mean_of_3_seeds"] = lofo_mean
        log(f"ensemble (3 seeds, mean thr/A_min) on original test tiles: F1 {rep['ensemble_on_original_tiles']['F1']:.4f} (LOFO mean {lofo_mean:.4f})")

        # ------------------------------------------------ full-field inference + stitching
        t0 = time.time()
        acc = np.zeros((H, W), np.float32)
        cnt = np.zeros((H, W), np.uint8)
        pos = [(y, x) for y in ys for x in xs]
        for s in range(0, len(pos), CFG["batch"]):
            pb = pos[s:s + CFG["batch"]]
            tiles = [tile_tensor(img, y, x) for y, x in pb]
            P = ensemble_prob(models, tiles)
            for j, (y, x) in enumerate(pb):
                h, w = min(TILE, H - y), min(TILE, W - x)
                acc[y:y + h, x:x + w] += P[j, :h, :w]
                cnt[y:y + h, x:x + w] += 1
            if (s // CFG["batch"]) % 20 == 0:
                log(f"field {k}: tiles {min(s + CFG['batch'], len(pos))}/{len(pos)} ({time.time() - t0:.0f}s)")
        assert cnt.min() >= 1, "some ortho pixels not covered by any tile"
        rep["overlap_count_hist"] = {int(v): int(n) for v, n in zip(*np.unique(cnt, return_counts=True))}
        np.divide(acc, cnt, out=acc)
        prob = acc; del cnt
        log(f"field {k}: {len(pos)} tiles predicted and stitched in {time.time() - t0:.0f}s; overlap counts {rep['overlap_count_hist']}")
        mask = remove_small((prob >= thr) & datamask, amin_px)
        mask_u8 = np.where(datamask, mask.astype(np.uint8), np.uint8(255))
        prob_out = np.where(datamask, np.round(prob, 3), np.float32(-1)).astype(np.float32)
        p_prob = os.path.join(OUT, f"Field{k}_canopy_prob.tif")
        p_mask = os.path.join(OUT, f"Field{k}_canopy_mask.tif")
        write_tif(p_prob, prob_out, ds, "float32", -1.0, 3)
        write_tif(p_mask, mask_u8, ds, "uint8", 255, 2)
        del prob_out
        chk = {}
        for p in (p_prob, p_mask):
            with rasterio.open(p) as o:
                chk[os.path.basename(p)] = dict(same_grid=(o.width, o.height, o.transform, o.crs) == (ds.width, ds.height, ds.transform, ds.crs),
                                                dtype=o.dtypes[0], nodata=o.nodata, MB=os.path.getsize(p) / 1e6)
        rep["outputs"] = chk
        log(f"written {chk}")

        # ------------------------------------------------ check (a): stitched mosaic on annotated test tiles
        st = np.zeros(4, np.int64)
        foot = np.zeros((H, W), bool); gtm = np.zeros((H, W), bool)
        nod_in = 0
        for (im_, gt, va), it in zip(loaded, items):
            y, x = it["r"] * STEP, it["c"] * STEP
            h, w = TILE - it["pad_h"], TILE - it["pad_w"]
            pr = mask_u8[y:y + h, x:x + w] == 1
            nod_in += int((mask_u8[y:y + h, x:x + w] == 255).sum())
            g = gt[:h, :w].astype(bool)
            st += [(pr & g).sum(), (pr & ~g).sum(), (~pr & g).sum(), (~pr & ~g).sum()]
            foot[y:y + h, x:x + w] = True
            gtm[y:y + h, x:x + w] |= g
        rep["stitched_mosaic_on_test_tiles"] = conf(*st)
        f1s = rep["stitched_mosaic_on_test_tiles"]["F1"]
        rep["check_a_diff_vs_lofo_mean"] = f1s - lofo_mean
        rep["check_a_pass_within_0p01"] = abs(f1s - lofo_mean) <= 0.01
        rep["nodata_px_inside_annotated_tiles"] = nod_in
        log(f"CHECK (a) field {k}: stitched-mosaic F1 on {len(items)} annotated test tiles {f1s:.4f} vs LOFO mean {lofo_mean:.4f} "
            f"(diff {f1s - lofo_mean:+.4f}) -> {'PASS' if rep['check_a_pass_within_0p01'] else 'FAIL'}")
        # ------------------------------------------------ check (b): canopy cover
        tree_px = int((mask_u8 == 1).sum()); valid_px = int(datamask.sum())
        fp_ = foot & datamask
        rep["cover"] = dict(field_mask_cover=tree_px / valid_px, field_canopy_area_ha=tree_px * ds.transform.a * -ds.transform.e / 1e4,
                            field_valid_area_ha=valid_px * ds.transform.a * -ds.transform.e / 1e4,
                            annotated_footprint_px=int(fp_.sum()),
                            annotated_cover_mask=float((mask_u8[fp_] == 1).mean()), annotated_cover_gt=float(gtm[fp_].mean()),
                            annotated_cover_mask_tile_pooled=(st[0] + st[1]) / max(st.sum(), 1),
                            annotated_cover_gt_tile_pooled=(st[0] + st[2]) / max(st.sum(), 1))
        log(f"CHECK (b) field {k}: whole-field canopy cover {rep['cover']['field_mask_cover'] * 100:.2f}% "
            f"({rep['cover']['field_canopy_area_ha']:.2f} ha of {rep['cover']['field_valid_area_ha']:.2f} ha); annotated footprint: "
            f"mask {rep['cover']['annotated_cover_mask'] * 100:.2f}% vs ground truth {rep['cover']['annotated_cover_gt'] * 100:.2f}%")
        gt_u8 = np.where(foot, gtm.astype(np.uint8), np.uint8(255))
        write_tif(os.path.join(OUT, f"Field{k}_annotated_ground_truth.tif"), gt_u8, ds, "uint8", 255, 2)
        del gt_u8, gtm, fp_
        # ------------------------------------------------ check (c): quicklooks
        s_ = CFG["quick_px"] / max(H, W)
        sz = (int(round(W * s_)), int(round(H * s_)))
        ql = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
        ms = cv2.resize(mask.astype(np.uint8) * 255, sz, interpolation=cv2.INTER_AREA) > 127
        dm = cv2.resize(datamask.astype(np.uint8) * 255, sz, interpolation=cv2.INTER_NEAREST) > 0
        ql[~dm] = (ql[~dm] * 0.4 + 60).astype(np.uint8)
        cs, _ = cv2.findContours(ms.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ql, cs, -1, (255, 230, 0), 1)
        zr = np.random.default_rng(1000 + k); boxes = []; Z = min(1500, H // 3, W // 3)
        for _ in range(500):
            yy, xx = int(zr.integers(0, H - Z)), int(zr.integers(0, W - Z))
            win = mask_u8[yy:yy + Z:10, xx:xx + Z:10]
            if (win != 255).mean() > 0.95 and 0.15 < (win == 1).mean() < 0.6 and all(abs(yy - b[0]) > Z or abs(xx - b[1]) > Z for b in boxes):
                boxes.append((yy, xx))
            if len(boxes) == 3:
                break
        for n, (yy, xx) in enumerate(boxes):
            cv2.rectangle(ql, (int(xx * s_), int(yy * s_)), (int((xx + Z) * s_), int((yy + Z) * s_)), (0, 255, 255), 2)
            cv2.putText(ql, str(n + 1), (int(xx * s_) + 4, int(yy * s_) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        txt = (f"Field {k}: canopy mask outline (yellow), RGB LOFO ensemble of 3 seeds, thr {thr:.3f}, A_min {amin_m2:.2f} m2; "
               f"cover {rep['cover']['field_mask_cover'] * 100:.1f}%")
        cv2.rectangle(ql, (0, 0), (sz[0], 34), (0, 0, 0), -1)
        cv2.putText(ql, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        Image.fromarray(ql).save(os.path.join(OUT, f"Field{k}_quicklook.png"))
        panels = []
        for n, (yy, xx) in enumerate(boxes):
            crop = img[yy:yy + Z, xx:xx + Z].copy()
            cz, _ = cv2.findContours((mask_u8[yy:yy + Z, xx:xx + Z] == 1).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(crop, cz, -1, (255, 230, 0), 3)
            crop = cv2.resize(crop, (1000, 1000), interpolation=cv2.INTER_AREA)
            cv2.putText(crop, f"zoom {n + 1}: {Z * ds.transform.a:.0f} m wide", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            panels.append(crop)
        if panels:
            Image.fromarray(np.hstack(panels)).save(os.path.join(OUT, f"Field{k}_quicklook_zoom.png"))
        rep["zoom_boxes_yx_px"] = boxes
        del img, acc, prob, mask, mask_u8, datamask, foot, loaded
        ds.close()
        rep["seconds"] = time.time() - t_field
        rep["status"] = "OK"
    except Exception as e:
        rep["status"] = f"FAILED: {type(e).__name__}: {e}"
        log(f"FIELD {k} FAILED:\n{traceback.format_exc()}")
    json.dump(rep, open(os.path.join(OUT, f"Field{k}_report.json"), "w"), indent=1, default=str)
    summary_rows.append(rep)
    torch.cuda.empty_cache() if AMP else None

# ---------------------------------------------------------------- summary
import csv
cols = ["field", "status", "ortho_W", "ortho_H", "epsg", "pixel_size_x", "n_tiles", "threshold_mean", "A_min_px_mean",
        "A_min_m2_mean", "lofo_test_F1_mean_of_3_seeds", "ens_F1_original_tiles", "stitched_F1_test_tiles",
        "check_a_diff_vs_lofo_mean", "check_a_pass_within_0p01", "field_cover_pct", "annot_cover_mask_pct", "annot_cover_gt_pct",
        "canopy_area_ha", "valid_area_ha"]
with open(os.path.join(OUT, "canopy_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in summary_rows:
        c = r.get("cover", {})
        w.writerow([r.get("field"), r.get("status"), r.get("ortho_W"), r.get("ortho_H"), r.get("epsg"), r.get("pixel_size_x"),
                    r.get("n_tiles"), r.get("threshold_mean"), r.get("A_min_px_mean"), r.get("A_min_m2_mean"),
                    r.get("lofo_test_F1_mean_of_3_seeds"), r.get("ensemble_on_original_tiles", {}).get("F1"),
                    r.get("stitched_mosaic_on_test_tiles", {}).get("F1"), r.get("check_a_diff_vs_lofo_mean"),
                    r.get("check_a_pass_within_0p01"), c.get("field_mask_cover", float("nan")) * 100,
                    c.get("annotated_cover_mask", float("nan")) * 100, c.get("annotated_cover_gt", float("nan")) * 100,
                    c.get("field_canopy_area_ha"), c.get("field_valid_area_ha")])
try:
    import shutil
    shutil.copy(os.path.abspath(sys.argv[0]), os.path.join(OUT, "pabna_canopy_maps_used.py"))
except Exception:
    pass
log("summary written; zipping")
zp = os.path.join(os.path.dirname(OUT), "canopy_maps_results.zip")
with zipfile.ZipFile(zp, "w", allowZip64=True) as z:
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        z.write(p, f"canopy_maps/{fn}", compress_type=zipfile.ZIP_STORED if fn.endswith((".tif", ".png")) else zipfile.ZIP_DEFLATED)
log(f"DONE: {zp} ({os.path.getsize(zp) / 1e6:.0f} MB); statuses {[r.get('status') for r in summary_rows]}")
