# %% [markdown]
# # Pabna UAV tree segmentation - Phase 0 (inspect & verify) + Phase 1 (leakage-free splits)
#
# **No training happens in this notebook. CPU is enough (Accelerator: None).**
#
# Inputs to attach (Add Input):
# 1. `sabbirsojib/pabna-tree-coco-raw` - the 1,305 annotated tiles **before** augmentation (Roboflow export, auto-orient only). **This is the data used for all new runs.**
# 2. `sabbirsojib/pabna-tree-ortho` - the 3 orthomosaic GeoTIFFs, used only to verify that the raw tiles are untouched originals and that row/col indices are correct.
# 3. `sabbirsojib/pabna-tree-augmented` - the old 5,220-copy dataset, used only to reproduce the old leakage numbers (S0, transparency).
#
# Facts taken from your earlier notebooks (verified again below where possible):
# * Tiling ("Stage 3 - Tiling Notebook"): 640×640 px tiles, **64 px overlap** (step 576), names `FieldK_Ortho_rRRRR_cCCCC.jpg`, edge tiles black-padded.
# * Merge ("Merge + Augment + Split"): fields merged in order Field1, Field2, Field3 → ids 1..1305; **all 4 copies `aug_ID_0..3` were augmented** (no untouched copy); split was done per copy (leakage).
#
# Outputs go to `/kaggle/working/results/`. The notebook STOPS after Phase 0 checks (cell 9 banner) and after Phase 1 (last cell). Read both reports before any training.

# %%
# -- Cell 1 - Setup -----------------------------------------------------------
# Why: one place for every fixed constant, the seed, and a decision log (run_log.md)
import os, re, sys, json, math, glob, random, hashlib, platform, datetime, collections, zipfile
try:
    from pycocotools import mask as mask_utils
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True)
    from pycocotools import mask as mask_utils
import pycocotools
import numpy as np
import pandas as pd
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_ROOT = os.environ.get("PABNA_INPUT", "/kaggle/input")
WORK       = os.environ.get("PABNA_WORK", "/kaggle/working")
OUT        = os.path.join(WORK, "results")
SPLIT_DIR  = os.path.join(OUT, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

SEED       = 20261106                 # fixed seed from the protocol
TILE, OVERLAP = 640, 64               # from the tiling notebook
STEP       = TILE - OVERLAP           # 576
GSD_M      = 0.026                    # orthomosaic GSD (m/px), Agisoft report (user)
VALID_FRAC = 0.15                     # LOFO validation share of training-field tiles
# Ortho sizes printed by the tiling notebook (used only if the GeoTIFFs are not attached)
ORTHO_SIZE_FALLBACK = {1: (15844, 14657), 2: (16060, 16930), 3: (16738, 15157)}

REPORT  = {}   # machine-readable report (phase0_report.json / phase1_report.json)
RUN_LOG = []   # plain factual list of settings and decisions
def log(msg):
    print(msg)
    RUN_LOG.append(msg)

log(f"Notebook run started {datetime.datetime.now().isoformat(timespec='seconds')}")
log(f"SEED={SEED}; TILE={TILE}; OVERLAP={OVERLAP}; STEP={STEP}; GSD={GSD_M} m/px; VALID_FRAC={VALID_FRAC}")
log(f"python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}; "
    f"opencv {cv2.__version__}; pycocotools {getattr(pycocotools, '__version__', 'n/a')}")

# %%
# -- Cell 2 - Phase 0.1: list mounted inputs and find the files ----------------
# Why: never trust a hard-coded path; show exactly what is attached
def walk_files(root):
    for r, _, fs in os.walk(root):
        for f in fs:
            yield os.path.join(r, f)

ALL_FILES = list(walk_files(INPUT_ROOT))
print(f"{len(ALL_FILES)} files under {INPUT_ROOT}")

# folder summary (depth-limited) with file-type counts
summary = collections.defaultdict(collections.Counter)
for p in ALL_FILES:
    rel = os.path.relpath(p, INPUT_ROOT).split(os.sep)
    key = "/".join(rel[:min(len(rel) - 1, 5)])
    summary[key][os.path.splitext(p)[1].lower() or "(none)"] += 1
for k in sorted(summary):
    print(f"  {k:<85} {dict(summary[k])}")

# raw COCO jsons (one per field)
raw_jsons = {}
for p in ALL_FILES:
    m = re.search(r"Field(\d)_COCO_raw", p)
    if m and os.path.basename(p) == "_annotations.coco.json":
        k = int(m.group(1))
        assert k not in raw_jsons, f"two raw jsons for Field{k}: {raw_jsons[k]} and {p}"
        raw_jsons[k] = p
assert sorted(raw_jsons) == [1, 2, 3], f"STOP: expected raw jsons for Field1-3, found {raw_jsons}"

# old augmented jsons (train/valid/test)
aug_jsons = collections.defaultdict(list)
for p in ALL_FILES:
    if os.path.basename(p) == "_annotations.coco.json" and "COCO_raw" not in p:
        parent = os.path.basename(os.path.dirname(p)).lower()
        if parent in ("train", "valid", "test"):
            aug_jsons[parent].append(p)
for s in list(aug_jsons):
    assert len(aug_jsons[s]) == 1, f"STOP: several '{s}' jsons found: {aug_jsons[s]}"
aug_jsons = {s: v[0] for s, v in aug_jsons.items()}
HAVE_AUG = sorted(aug_jsons) == ["test", "train", "valid"]

AUG_IMG = {os.path.basename(p): p for p in ALL_FILES if re.fullmatch(r"aug_\d{6}_\d\.jpg", os.path.basename(p))}

orthos = {}
for p in ALL_FILES:
    m = re.search(r"Field(\d)_Ortho\.tiff?$", os.path.basename(p), re.I)
    if m:
        orthos[int(m.group(1))] = p
HAVE_ORTHO = sorted(orthos) == [1, 2, 3]

log(f"Raw jsons: {raw_jsons}")
log(f"Augmented jsons: {aug_jsons if HAVE_AUG else 'NOT ATTACHED'}; aug images found: {len(AUG_IMG)}")
log(f"Orthomosaics: {orthos if HAVE_ORTHO else 'NOT ATTACHED (row/col & originality check will be skipped)'}")

# %%
# -- Cell 3 - Phase 0.1 (cont.): what is inside each COCO json ----------------
# Why: categories, annotation type (polygon/RLE) and image sizes decide how masks are built
def seg_type(seg):
    if isinstance(seg, dict):
        return "RLE-compressed" if isinstance(seg.get("counts"), (str, bytes)) else "RLE-uncompressed"
    if isinstance(seg, list):
        if len(seg) == 0:
            return "empty"
        if all(isinstance(s, (list, tuple)) for s in seg):
            return "polygon"
        if all(isinstance(s, (int, float)) for s in seg):
            return "polygon-flat"
        if isinstance(seg[0], str):
            return "string"
    return "other"

def describe_coco(path, label):
    with open(path) as f:
        coco = json.load(f)
    imgs, anns = coco["images"], coco["annotations"]
    types = collections.Counter(seg_type(a.get("segmentation")) for a in anns)
    sizes = collections.Counter((i["width"], i["height"]) for i in imgs)
    cats  = collections.Counter(a["category_id"] for a in anns)
    with_ann = {a["image_id"] for a in anns}
    print(f"\n[{label}] {path}")
    print(f"  categories : {coco.get('categories')}")
    print(f"  images     : {len(imgs)}  (without any annotation: {sum(i['id'] not in with_ann for i in imgs)})")
    print(f"  annotations: {len(anns)}  by category_id: {dict(cats)}")
    print(f"  seg types  : {dict(types)}")
    print(f"  image sizes: {dict(sizes)}")
    print(f"  file names : {[i['file_name'] for i in imgs[:3]]}")
    return coco, dict(n_images=len(imgs), n_annotations=len(anns), seg_types=dict(types),
                      sizes={f'{w}x{h}': n for (w, h), n in sizes.items()},
                      categories=coco.get("categories"), ann_by_category=dict(cats))

RAW = {}
REPORT["raw_json"] = {}
for k in (1, 2, 3):
    RAW[k], REPORT["raw_json"][f"Field{k}"] = describe_coco(raw_jsons[k], f"RAW Field{k}")
if HAVE_AUG:
    AUG = {}
    REPORT["aug_json"] = {}
    for s in ("train", "valid", "test"):
        AUG[s], REPORT["aug_json"][s] = describe_coco(aug_jsons[s], f"AUG {s}")

# Category check: binary task -> every annotation must be a 'tree'-type class
used_cats = set()
for k in (1, 2, 3):
    names = {c["id"]: c["name"] for c in RAW[k]["categories"]}
    for a in RAW[k]["annotations"]:
        used_cats.add((k, a["category_id"], names.get(a["category_id"], "?")))
log(f"Raw category ids actually used (field, id, name): {sorted(used_cats)}")
# Decision needed from the user for any class that is not called 'tree'.
# Until the user decides, such annotations are FLAGGED, kept in the split files with their original
# class name, and NOT painted into the tree masks used for the Phase 0 statistics.
# Set e.g. CATEGORY_DECISION = {(2, "h"): "tree"} or {(2, "h"): "drop"} after looking at the figure.
CATEGORY_DECISION = {(2, "h"): "drop"}   # user decision 2026-11-06: stray brush stroke, delete
NON_TREE = {(k, cid): name for (k, cid, name) in used_cats if "tree" not in name.lower()}
for (k, cid), name in NON_TREE.items():
    n_ = sum(a["category_id"] == cid for a in RAW[k]["annotations"])
    log(f"FLAG: Field{k} class id {cid} '{name}' has {n_} annotation(s); decision = "
        f"{CATEGORY_DECISION.get((k, name), 'PENDING (excluded from masks until the user decides)')}")
def is_tree(k, cid):
    if (k, cid) not in NON_TREE:
        return True
    return CATEGORY_DECISION.get((k, NON_TREE[(k, cid)])) == "tree"

# %%
# -- Cell 4 - Phase 0.2: build one tile pool with safe re-indexing -------------
# Why: one row per physical tile; tile_id = field + row + col from the ORIGINAL tile name,
#      so a tile can never be counted twice. Masks are decoded strictly (failures are counted,
#      never skipped silently as in the old merge notebook).
NAME_RE = re.compile(r"Field(\d)_Ortho_r(\d{4})_c(\d{4})")

def parse_tile_name(img):
    for cand in (img.get("file_name", ""), (img.get("extra") or {}).get("name", "")):
        m = NAME_RE.search(cand or "")
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None

def ann_to_mask(seg, H, W):
    """Strict COCO decoding (pycocotools rasterisation). Returns None if undecodable."""
    t = seg_type(seg)
    try:
        if t == "polygon":
            polys = [p for p in seg if len(p) >= 6]
            if not polys:
                return None
            m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polys, H, W)))
        elif t == "polygon-flat":
            if len(seg) < 6:
                return None
            m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects([seg], H, W)))
        elif t == "RLE-uncompressed":
            m = mask_utils.decode(mask_utils.frPyObjects(seg, H, W))
        elif t == "RLE-compressed":
            rle = dict(seg)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode()
            m = mask_utils.decode(rle)
        else:
            return None
    except Exception:
        return None
    if m.ndim == 3:
        m = m.max(axis=2)
    return m.astype(np.uint8)

def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None, f"cannot read {path}"
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Two raw files are stored at 1024x1024 although their json (and their annotations) say 640x640.
# Every raw tile is therefore read through read_raw(), which resizes such files to 640x640 so that
# image and annotation share one pixel grid. Cell 6 verifies this against the orthomosaic.
RESIZED = {}
def read_raw(path):
    im = read_rgb(path)
    if im.shape[:2] != (TILE, TILE):
        RESIZED[path] = im.shape[:2]
        im = cv2.resize(im, (TILE, TILE), interpolation=cv2.INTER_AREA)
    return im

# ortho sizes (W, H) -> grid size and padding
ORTHO_SIZE = {}
for k in (1, 2, 3):
    if HAVE_ORTHO:
        with Image.open(orthos[k]) as im:
            ORTHO_SIZE[k] = im.size
        if ORTHO_SIZE[k] != ORTHO_SIZE_FALLBACK[k]:
            log(f"NOTE Field{k}: ortho size {ORTHO_SIZE[k]} differs from tiling-notebook print {ORTHO_SIZE_FALLBACK[k]}")
    else:
        ORTHO_SIZE[k] = ORTHO_SIZE_FALLBACK[k]
GRID = {k: (math.ceil((H - OVERLAP) / STEP), math.ceil((W - OVERLAP) / STEP)) for k, (W, H) in ORTHO_SIZE.items()}
log(f"Ortho sizes (W,H): {ORTHO_SIZE}; tile grid (rows, cols): {GRID}")

rows, unparsed, problems, FLAGGED = [], [], [], []
MASKS = {}           # tile_id -> union mask (uint8, HxW), kept in RAM (1305 x 640x640 ≈ 0.5 GB)
merged_id = 0
for k in (1, 2, 3):
    coco = RAW[k]
    img_dir = os.path.dirname(raw_jsons[k])
    anns_by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    for img in coco["images"]:                       # json order == merge order of the old notebook
        merged_id += 1
        parsed = parse_tile_name(img)
        if parsed is None:
            unparsed.append((k, img["file_name"]))
            continue
        fk, r, c = parsed
        if fk != k:
            problems.append(f"name field {fk} != folder field {k}: {img['file_name']}")
        H, W = img["height"], img["width"]
        path = os.path.join(img_dir, img["file_name"])
        rgb0 = read_rgb(path)
        if rgb0.shape[:2] != (H, W):
            problems.append(f"size mismatch json {W}x{H} vs file {rgb0.shape[1]}x{rgb0.shape[0]} "
                            f"(resized to json size for all use): {path}")
        assert (H, W) == (TILE, TILE), f"STOP: json size {W}x{H} is not {TILE}x{TILE}: {path}"
        rgb = read_raw(path)
        union = np.zeros((H, W), np.uint8)
        count = np.zeros((H, W), np.uint8)
        n_fail, areas = 0, []
        for a in anns_by_img[img["id"]]:
            if not is_tree(k, a["category_id"]):
                FLAGGED.append(dict(field=k, tile=f"F{k}_r{r:04d}_c{c:04d}", path=path, ann=a,
                                    category=NON_TREE[(k, a["category_id"])]))
                continue
            m = ann_to_mask(a.get("segmentation"), H, W)
            if m is None or m.sum() == 0:
                n_fail += 1
                continue
            union |= m
            count += m
            areas.append(int(m.sum()))
        tile_id = f"F{k}_r{r:04d}_c{c:04d}"
        Wo, Ho = ORTHO_SIZE[k]
        pad_w = max(0, c * STEP + TILE - Wo)
        pad_h = max(0, r * STEP + TILE - Ho)
        rows.append(dict(
            tile_id=tile_id, field=k, r=r, c=c, merged_id=merged_id,
            raw_image_id=img["id"], raw_file=img["file_name"], path=path, width=W, height=H,
            n_ann=len(anns_by_img[img["id"]]), n_ann_failed=n_fail,
            tree_px=int(union.sum()), overlap_px=int((count > 1).sum()),
            ann_area_px_median=float(np.median(areas)) if areas else np.nan,
            pad_w=pad_w, pad_h=pad_h, valid_px=(TILE - pad_w) * (TILE - pad_h),
            in_grid=(r < GRID[k][0] and c < GRID[k][1]),
            md5=hashlib.md5(rgb.tobytes()).hexdigest(),
        ))
        MASKS[tile_id] = union

if unparsed:
    print("Unparsed names (first 20):", unparsed[:20])
    raise RuntimeError(f"STOP: {len(unparsed)} raw images have no FieldK_Ortho_rRRRR_cCCCC in their name - ask the user")
pool = pd.DataFrame(rows)
pool["tree_frac"] = pool.tree_px / (pool.width * pool.height)

dup_ids = pool.tile_id[pool.tile_id.duplicated(keep=False)].tolist()
dup_md5 = pool[pool.md5.duplicated(keep=False)].sort_values("md5")[["tile_id", "raw_file"]]
log(f"Pool: {len(pool)} tiles; merged ids 1..{merged_id}; per field {pool.field.value_counts().sort_index().to_dict()}")
log(f"Duplicate tile_ids: {len(dup_ids)}; identical-pixel images: {len(dup_md5)}; out-of-grid tiles: {(~pool.in_grid).sum()}; "
    f"name/size problems: {len(problems)}")
if problems:
    print("\n".join(problems[:20]))
if len(dup_md5):
    print(dup_md5.to_string(index=False))
assert not dup_ids, f"STOP: duplicate tile ids {dup_ids[:10]}"
assert pool.in_grid.all(), "STOP: row/col outside the tiling grid - tile names do not match the orthos"

ann_tot = int(pool.n_ann.sum()); fail_tot = int(pool.n_ann_failed.sum())
per_field = pool.groupby("field").agg(
    tiles=("tile_id", "size"), annotations=("n_ann", "sum"), failed=("n_ann_failed", "sum"),
    empty_tiles=("tree_px", lambda s: int((s == 0).sum())),
    tree_frac_pooled=("tree_px", "sum"), padded_tiles=("valid_px", lambda s: int((s < TILE * TILE).sum())),
)
per_field["tree_frac_pooled"] = per_field.tree_frac_pooled / (per_field.tiles * TILE * TILE)
print("\nPer field:\n", per_field.to_string())
REPORT["pool"] = dict(n_tiles=len(pool), per_field=per_field.reset_index().to_dict("records"),
                      annotations=ann_tot, annotations_failed=fail_tot,
                      duplicate_pixel_images=dup_md5.tile_id.tolist(), problems=problems[:50])
log(f"Annotations: {ann_tot}, undecodable/empty: {fail_tot}")

# %%
# -- Cell 5 - Phase 0.2 (cont.): what do the polygons represent? --------------
# Why: object-level metrics need to know if one polygon = one crown
areas_m2 = []
n_poly_tile = pool.n_ann
for k in (1, 2, 3):
    for a in RAW[k]["annotations"]:
        if "area" in a:
            areas_m2.append(a["area"] * GSD_M ** 2)
areas_m2 = np.array(areas_m2)
overlap_share = pool.overlap_px.sum() / max(pool.tree_px.sum(), 1)
q = np.percentile(areas_m2, [5, 25, 50, 75, 95]) if len(areas_m2) else []
print(f"Polygons per tile: median {n_poly_tile.median():.0f}, max {n_poly_tile.max()}")
print(f"Polygon area (m², from COCO 'area'): p5/p25/p50/p75/p95 = {np.round(q, 2)}")
print(f"Share of tree pixels covered by >1 polygon: {overlap_share:.4f}")
REPORT["polygons"] = dict(per_tile_median=float(n_poly_tile.median()), area_m2_percentiles=[float(x) for x in q],
                          overlap_share=float(overlap_share))
log("Interpretation rule: small overlap share + crown-sized areas (a few to tens of m2) => polygons are "
    "individual crowns; very large areas => merged canopy. Judge from the numbers above and the figure in Cell 8.")

# Flagged (non-'tree') annotations: show them so the user can decide
if FLAGGED:
    fig, axes = plt.subplots(1, len(FLAGGED), figsize=(6 * len(FLAGGED), 6), squeeze=False)
    for ax, fl in zip(axes[0], FLAGGED):
        rgb = read_raw(fl["path"])
        m = ann_to_mask(fl["ann"].get("segmentation"), rgb.shape[0], rgb.shape[1])
        ax.imshow(rgb)
        if m is not None:
            cn, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cc in cn:
                ax.plot(cc[:, 0, 0], cc[:, 0, 1], c="red", lw=1.5)
            area = int(m.sum())
        else:
            area = 0
        ax.set_title(f"{fl['tile']}  class '{fl['category']}'  area {area * GSD_M**2:.1f} m2"); ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase0_flagged_annotations.png"), dpi=120); plt.close()
    log(f"Saved phase0_flagged_annotations.png ({len(FLAGGED)} flagged annotation(s)) for the user's decision")

# Tiles with NO tree annotation: true negatives or never annotated? Rank them by greenness (ExG).
def exg_green_frac(rgb):
    f = rgb.astype(np.float32) / 255.0
    exg = 2 * f[..., 1] - f[..., 0] - f[..., 2]
    return float((exg > 0.1).mean())
pool["exg_green_frac"] = [exg_green_frac(read_raw(p)) for p in pool.path]
empty = pool[pool.tree_px == 0].sort_values("exg_green_frac", ascending=False)
annot = pool[pool.tree_px > 0]
print(f"Tiles without any tree annotation: {len(empty)} "
      f"(F1 {sum(empty.field == 1)}, F2 {sum(empty.field == 2)}, F3 {sum(empty.field == 3)})")
print(f"Green-pixel share (ExG>0.1): empty tiles median {empty.exg_green_frac.median():.3f}, "
      f"annotated tiles median {annot.exg_green_frac.median():.3f}")
if len(empty):
    show = empty.head(12)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, t) in zip(axes.ravel(), show.iterrows()):
        ax.imshow(read_raw(t.path)); ax.set_title(f"{t.tile_id}  green {t.exg_green_frac:.1%}")
    plt.suptitle("Tiles with NO tree annotation, greenest first (check: really no trees, or not annotated?)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase0_empty_tiles_greenest.png"), dpi=100); plt.close()
REPORT["empty_tiles"] = dict(n=len(empty), per_field=empty.field.value_counts().sort_index().to_dict(),
                             green_median_empty=float(empty.exg_green_frac.median()) if len(empty) else None,
                             green_median_annotated=float(annot.exg_green_frac.median()))
log(f"Empty (no tree annotation) tiles: {len(empty)}; greenest 12 saved to phase0_empty_tiles_greenest.png "
    f"for the user to judge whether they are true negatives")

# %%
# -- Cell 6 - Phase 0.3: are the raw tiles untouched originals? ---------------
# Why: the protocol requires val/test on un-augmented images. Compare each sampled raw tile with
#      the crop at (row, col) of its orthomosaic: identity must match best; a shifted crop is the control.
DIHEDRAL = {
    "id": lambda a: a, "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3), "flipH": lambda a: a[:, ::-1], "flipV": lambda a: a[::-1],
    "transpose": lambda a: np.swapaxes(a, 0, 1), "antitranspose": lambda a: np.rot90(a, 2).swapaxes(0, 1),
}
def mse(a, b, valid=None):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if valid is not None:
        d = d[:valid[0], :valid[1]]
    return float(d.mean())

def ortho_crop(im, k, r, c):
    W, H = ORTHO_SIZE[k]
    x, y = c * STEP, r * STEP
    crop = im.crop((x, y, min(x + TILE, W), min(y + TILE, H))).convert("RGB")
    canvas = Image.new("RGB", (TILE, TILE), (0, 0, 0))
    canvas.paste(crop, (0, 0))
    return np.asarray(canvas)

orig_rows, ODD_FIG = [], []
rng0 = np.random.default_rng(SEED)
if HAVE_ORTHO:
    for k in (1, 2, 3):
        sub = pool[pool.field == k]
        sample = sub.iloc[rng0.choice(len(sub), size=min(30, len(sub)), replace=False)]
        odd = sub[sub.path.isin(RESIZED)]                  # always check the odd-size files too
        sample = pd.concat([sample, odd[~odd.tile_id.isin(sample.tile_id)]])
        with Image.open(orthos[k]) as im:
            im.load()                                  # one full read per field (~1 GB RAM)
            for _, t in sample.iterrows():
                raw = read_raw(t.path)
                valid = (TILE - t.pad_h, TILE - t.pad_w)
                ref = ortho_crop(im, k, t.r, t.c)
                d = {n: mse(f(ref), raw) for n, f in DIHEDRAL.items()}
                ctrl_c = t.c + 1 if t.c + 1 < GRID[k][1] else t.c - 1
                ctrl = mse(ortho_crop(im, k, t.r, ctrl_c), raw, valid)
                best = min(d, key=d.get)
                orig_rows.append(dict(tile_id=t.tile_id, field=k, mse_identity=mse(ref, raw, valid),
                                      best_transform=best, mse_shifted_control=ctrl,
                                      file_was_resized=t.path in RESIZED))
                if t.path in RESIZED:
                    ODD_FIG.append((t.tile_id, raw, ref, MASKS[t.tile_id], RESIZED[t.path]))
            del im
    orig = pd.DataFrame(orig_rows)
    orig["verified"] = (orig.best_transform == "id") & (orig.mse_identity < 0.1 * orig.mse_shifted_control)
    orig.to_csv(os.path.join(OUT, "phase0_originality_check.csv"), index=False)
    print(orig.groupby("field").agg(n=("tile_id", "size"), verified=("verified", "sum"),
                                    mse_id_median=("mse_identity", "median"),
                                    mse_ctrl_median=("mse_shifted_control", "median")).to_string())
    print("Best transform counts:", orig.best_transform.value_counts().to_dict())
    REPORT["originality"] = dict(n=len(orig), verified=int(orig.verified.sum()),
                                 mse_identity_median=float(orig.mse_identity.median()),
                                 mse_control_median=float(orig.mse_shifted_control.median()))
    log(f"Originality check vs orthomosaic: {int(orig.verified.sum())}/{len(orig)} sampled raw tiles match their "
        f"(row, col) crop with no flip/rotation (median MSE {orig.mse_identity.median():.1f} vs shifted control "
        f"{orig.mse_shifted_control.median():.1f}; residual = JPEG re-encoding)")
    if ODD_FIG:
        fig, axes = plt.subplots(len(ODD_FIG), 3, figsize=(15, 5 * len(ODD_FIG)), squeeze=False)
        for row, (tid, raw, ref, m, shp) in zip(axes, ODD_FIG):
            ov = raw.astype(np.float32) / 255
            ov[m.astype(bool)] = 0.5 * ov[m.astype(bool)] + 0.5 * np.array([1.0, 0.1, 0.9])
            row[0].imshow(raw); row[0].set_title(f"{tid}: file {shp[1]}x{shp[0]} resized to 640")
            row[1].imshow(ref); row[1].set_title("orthomosaic crop at (row, col)")
            row[2].imshow(ov); row[2].set_title("annotation on resized file")
            for a_ in row:
                a_.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase0_oddsize_tiles.png"), dpi=100); plt.close()
        od = orig[orig.file_was_resized]
        log(f"Odd-size raw files {list(od.tile_id)}: after resizing to 640, MSE vs ortho crop {list(od.mse_identity.round(1))} "
            f"(control {list(od.mse_shifted_control.round(1))}); verified = {list(od.verified)}. "
            f"Figure phase0_oddsize_tiles.png")
    if orig.verified.mean() < 0.95:
        print("!!! STOP: fewer than 95% of sampled raw tiles verified as untouched originals - report to user")
else:
    log("Originality check skipped: orthomosaics not attached")

# %%
# -- Cell 7 - Phase 0.4: map old augmented copies to raw tiles + reproduce the leakage -----
# Why: needed only for the S0 transparency row. The id→tile mapping follows the merge order;
#      it is verified for EVERY old tile by image matching (8 flips/rotations, z-scored 32x32 gray).
def feat(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(cv2.GaussianBlur(g, (5, 5), 0), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return g

def z(v):
    v = v.reshape(-1).astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-6)

field_map = pool[["tile_id", "field", "r", "c", "raw_file", "merged_id"]].copy()
if HAVE_AUG:
    # raw features in 8 orientations
    raw_ids = pool.tile_id.tolist()
    R = np.stack([np.stack([z(f(feat(read_raw(p)))) for f in DIHEDRAL.values()]) for p in pool.path])  # N x 8 x 1024
    Rf = R.reshape(-1, R.shape[-1])                                                                  # (N*8) x 1024
    by_mid = dict(zip(pool.merged_id, pool.tile_id))
    aug_rows = []
    for s in ("train", "valid", "test"):
        for img in AUG[s]["images"]:
            b = os.path.basename(img["file_name"])
            m = re.fullmatch(r"aug_(\d{6})_(\d)\.jpg", b)
            aug_rows.append(dict(split=s, aug_file=b, mid=int(m.group(1)), copy=int(m.group(2))))
    augdf = pd.DataFrame(aug_rows)
    odd = [(s, i["file_name"], i["width"], i["height"]) for s in AUG for i in AUG[s]["images"]
           if (i["width"], i["height"]) != (TILE, TILE)]
    if odd:
        log(f"NOTE: {len(odd)} old augmented images are not 640x640 in the old json: {odd}")
    missing = augdf[~augdf.aug_file.isin(AUG_IMG)]
    assert missing.empty, f"STOP: {len(missing)} augmented images listed in json but not found"
    A = np.stack([z(feat(read_rgb(AUG_IMG[b]))) for b in augdf.aug_file])                           # M x 1024
    corr = (A @ Rf.T) / A.shape[1]                                                                 # M x (N*8)
    corr = corr.reshape(len(A), len(raw_ids), 8).max(axis=2)                                       # best orientation
    augdf["corr_row"] = np.arange(len(augdf))
    res = []
    for mid, g in augdf.groupby("mid"):
        score = corr[g.corr_row.values].mean(axis=0)            # average over the copies of this old tile
        order = np.argsort(-score)
        best, second = order[0], order[1]
        expected = by_mid.get(mid)
        res.append(dict(merged_id=mid, n_copies=len(g), splits=",".join(sorted(set(g.split))),
                        expected_tile=expected, best_tile=raw_ids[best],
                        best_dist=float(1 - score[best]), second_dist=float(1 - score[second])))
    match = pd.DataFrame(res)
    match["match_ok"] = match.expected_tile == match.best_tile
    match["ambiguous"] = (match.second_dist - match.best_dist) < 0.05
    field_map = field_map.merge(match.rename(columns={"expected_tile": "tile_id"}).drop(columns=["best_tile"]),
                                on=["merged_id", "tile_id"], how="left")
    field_map["match_ok"] = field_map.merged_id.map(dict(zip(match.merged_id, match.match_ok)))
    print(f"Old ids matched to the raw tile predicted by merge order: {match.match_ok.sum()}/{len(match)}; "
          f"ambiguous (margin<0.05): {match.ambiguous.sum()}")
    if (~match.match_ok).any():
        print(match[~match.match_ok].head(20).to_string(index=False))
    plt.figure(figsize=(6, 3.5))
    plt.hist(match.best_dist, bins=40, alpha=.7, label="best")
    plt.hist(match.second_dist, bins=40, alpha=.7, label="second best")
    plt.xlabel("1 - correlation (old augmented copies vs raw tile)"); plt.ylabel("old tiles"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase0_match_hist.png"), dpi=150); plt.close()

    # is any old copy an untouched original? (pixel MSE after best flip/rotation, full resolution, 30 tiles)
    sample_mid = rng0.choice(match.merged_id.values, size=min(30, len(match)), replace=False)
    copy_mse = []
    for mid in sample_mid:
        raw = read_raw(pool.loc[pool.merged_id == mid, "path"].iloc[0])
        for _, g in augdf[augdf.mid == mid].iterrows():
            a = read_rgb(AUG_IMG[g.aug_file])
            if a.shape != raw.shape:          # a few old copies were saved at another size
                a = cv2.resize(a, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
            copy_mse.append(dict(merged_id=mid, copy=g.copy, min_mse=min(mse(f(raw), a) for f in DIHEDRAL.values())))
    copy_mse = pd.DataFrame(copy_mse)
    n_ident = int((copy_mse.min_mse < 5).sum())
    print(f"Old copies near-identical to their raw tile (pixel MSE<5 after best flip/rot; i.e. the random "
          f"augmentation happened to change almost nothing): {n_ident}/{len(copy_mse)}; "
          f"median MSE {copy_mse.min_mse.median():.1f}")

    # reproduce the leakage of the old copy-level split (S0)
    sets = augdf.groupby("mid").split.apply(set)
    multi = int((sets.apply(len) > 1).sum())
    test_in_train = int(sets.apply(lambda s: "test" in s and "train" in s).sum())
    n_test_tiles = int(sets.apply(lambda s: "test" in s).sum())
    log(f"S0 (old split): {len(sets)} old tiles; {multi} ({multi/len(sets):.1%}) have copies in >1 split; "
        f"{test_in_train} of {n_test_tiles} test tiles also have copies in train "
        f"(user's figures: 785 / 439). Old copies identical to originals: {n_ident}/{len(copy_mse)}")
    REPORT["S0"] = dict(old_tiles=len(sets), tiles_in_multiple_splits=multi, test_tiles=n_test_tiles,
                        test_tiles_also_in_train=test_in_train, id_match_ok=int(match.match_ok.sum()),
                        id_match_ambiguous=int(match.ambiguous.sum()), copies_identical_to_raw=n_ident,
                        copies_checked=len(copy_mse))
    augdf = augdf.merge(match[["merged_id", "expected_tile"]].rename(columns={"merged_id": "mid", "expected_tile": "tile_id"}),
                        on="mid", how="left")
    augdf.drop(columns=["corr_row"]).to_csv(os.path.join(SPLIT_DIR, "S0_old_split_by_copy.csv"), index=False)
    del R, Rf, A, corr
field_map.to_csv(os.path.join(OUT, "field_map.csv"), index=False)
print("Saved field_map.csv")

# %%
# -- Cell 8 - Phase 0.5: row/col indices and overlap between neighbours -------
# Why: 64 px overlap means neighbouring tiles share pixels -> must be handled when splitting
key = {(t.field, t.r, t.c): t for t in pool.itertuples()}
def pair_stats(direction):
    out = []
    for (k, r, c), a in key.items():
        nb = (k, r, c + 1) if direction == "h" else (k, r + 1, c)
        if nb not in key:
            continue
        b = key[nb]
        ia, ib = read_raw(a.path), read_raw(b.path)
        ma, mb = MASKS[a.tile_id], MASKS[b.tile_id]
        if direction == "h":
            sa, sb, xa, xb = ia[:, STEP:], ib[:, :OVERLAP], ma[:, STEP:], mb[:, :OVERLAP]
        else:
            sa, sb, xa, xb = ia[STEP:], ib[:OVERLAP], ma[STEP:], mb[:OVERLAP]
        inter, union = int((xa & xb).sum()), int((xa | xb).sum())
        out.append(dict(a=a.tile_id, b=b.tile_id, dir=direction, mse=mse(sa, sb),
                        label_iou=inter / union if union else np.nan))
    return out
pairs = pd.DataFrame(pair_stats("h") + pair_stats("v"))
# control: same strips from random non-adjacent tile pairs
ctrl = []
ids = pool.tile_id.values
for _ in range(min(300, len(pool))):
    i, j = rng0.choice(len(pool), 2, replace=False)
    ctrl.append(mse(read_raw(pool.path.iloc[i])[:, STEP:], read_raw(pool.path.iloc[j])[:, :OVERLAP]))
n8 = 0
for (k, r, c) in key:
    if any((k, r + dr, c + dc) in key for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
        n8 += 1
pairs.to_csv(os.path.join(OUT, "phase0_neighbour_pairs.csv"), index=False)
print(f"Annotated neighbour pairs sharing a 64-px strip: {len(pairs)} (h={sum(pairs.dir=='h')}, v={sum(pairs.dir=='v')})")
print(f"Strip MSE: adjacent median {pairs.mse.median():.1f} vs random-pair control median {np.median(ctrl):.1f}")
print(f"Label agreement in shared strips (IoU of tree pixels): median {pairs.label_iou.median():.3f}")
print(f"Tiles with >=1 annotated 8-neighbour: {n8}/{len(pool)}")
REPORT["overlap"] = dict(pairs=len(pairs), strip_mse_adjacent_median=float(pairs.mse.median()) if len(pairs) else None,
                         strip_mse_control_median=float(np.median(ctrl)),
                         label_iou_median=float(pairs.label_iou.median()) if len(pairs) else None,
                         tiles_with_annotated_neighbour=n8)
log(f"Row/col indices present in all raw names. Neighbouring tiles overlap by {OVERLAP} px "
    f"(adjacent-strip MSE {REPORT['overlap']['strip_mse_adjacent_median']} vs control {np.median(ctrl):.1f}).")

# -- Phase 0.6: visual sanity check -------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, i in zip(axes.ravel(), rng0.choice(len(pool), 6, replace=False)):
    t = pool.iloc[i]
    rgb = read_raw(t.path).astype(np.float32) / 255
    m = MASKS[t.tile_id].astype(bool)
    over = rgb.copy(); over[m] = 0.5 * over[m] + 0.5 * np.array([1.0, 0.1, 0.9])
    ax.imshow(over)
    cnts, _ = cv2.findContours(MASKS[t.tile_id], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cn in cnts:
        ax.plot(cn[:, 0, 0], cn[:, 0, 1], lw=0.6, c="yellow")
    ax.set_title(f"{t.tile_id}  tree {t.tree_frac:.1%}  polygons {t.n_ann}"); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase0_gt_overlay.png"), dpi=120); plt.close()
print("Saved phase0_gt_overlay.png")

# %%
# -- Cell 9 - Phase 0 report (STOP POINT 1) -----------------------------------
pool.drop(columns=["md5"]).to_csv(os.path.join(OUT, "phase0_tiles.csv"), index=False)
with open(os.path.join(OUT, "phase0_report.json"), "w") as f:
    json.dump(REPORT, f, indent=2, default=str)
print("=" * 78)
print("PHASE 0 DONE - read the outputs above, phase0_report.json and the png figures.")
print("Check: (1) categories, (2) failed annotations, (3) originality, (4) S0 leakage numbers, (5) overlap.")
print("Phase 1 below only builds split files (no training). Training must wait for your review.")
print("=" * 78)

# User decision 2026-11-06: the 2 odd-size raw files are empty no-data tiles (all white / all black
# at the ortho edge, no annotations). They are removed from ALL splits. Phase 0 statistics above keep them.
EXCLUDE_TILES = sorted(pool.loc[pool.path.isin(RESIZED), "tile_id"])
assert int(pool.loc[pool.tile_id.isin(EXCLUDE_TILES), "tree_px"].sum()) == 0, "STOP: an excluded tile has tree labels"
pool = pool[~pool.tile_id.isin(EXCLUDE_TILES)].reset_index(drop=True)
log(f"Excluded from all splits (empty no-data tiles, user decision): {EXCLUDE_TILES}; tiles used for splits: {len(pool)}")
log(f"Class 'h' annotation: decision = drop (user decision); it is not in any mask or split file")

# %%
# -- Cell 10 - Phase 1: S1 tile-grouped random split 80/10/10 -----------------
# Why: secondary result. Grouping by tile_id removes copy leakage; the 64-px overlap leak
#      between neighbours is NOT removed here and is measured below (so S1 is optimistic).
tiles_sorted = sorted(pool.tile_id)
rng1 = np.random.default_rng(SEED)
perm = rng1.permutation(len(tiles_sorted))
n = len(tiles_sorted); n_tr = int(round(0.8 * n)); n_va = int(round(0.1 * n))
SPLITS = {"S1": {"train": [tiles_sorted[i] for i in perm[:n_tr]],
                 "valid": [tiles_sorted[i] for i in perm[n_tr:n_tr + n_va]],
                 "test":  [tiles_sorted[i] for i in perm[n_tr + n_va:]],
                 "buffer_unused": []}}
log(f"S1: random tile split 80/10/10 with numpy default_rng({SEED}) over sorted tile_ids")

# %%
# -- Cell 11 - Phase 1: S2 leave-one-field-out (main result) ------------------
# Why: test = a whole unseen field. Valid = a contiguous band of tile ROWS inside each training
#      field (~15% of its tiles). The rows directly above and below the band share 64-px strips
#      with valid tiles, so they are left out of training ("buffer") - valid stays spatially independent.
def row_band(df_f, frac, rng):
    counts = df_f.groupby("r").size()
    r_min, r_max = int(df_f.r.min()), int(df_f.r.max())
    per_row = np.array([counts.get(r, 0) for r in range(r_min, r_max + 1)])
    target = frac * len(df_f)
    cands = []
    for i in range(len(per_row)):
        s = 0
        for j in range(i, len(per_row)):
            s += per_row[j]
            if s >= target:
                cands.append((r_min + i, r_min + j, s)); break
    # keep bands whose size is closest to the target (ties broken by the seeded rng)
    best = min(abs(cs - target) for _, _, cs in cands)
    cands = [cd for cd in cands if abs(cd[2] - target) <= max(best, 0.02 * len(df_f))]
    r0, r1, _ = cands[int(rng.integers(len(cands)))]
    return r0, r1

for k in (1, 2, 3):
    rngk = np.random.default_rng(SEED + k)
    test = pool[pool.field == k].tile_id.tolist()
    train, valid, buffer = [], [], []
    for kk in sorted({1, 2, 3} - {k}):
        df_f = pool[pool.field == kk]
        r0, r1 = row_band(df_f, VALID_FRAC, rngk)
        in_band = df_f.r.between(r0, r1)
        in_buf = df_f.r.isin([r0 - 1, r1 + 1])
        valid += df_f[in_band].tile_id.tolist()
        buffer += df_f[in_buf].tile_id.tolist()
        train += df_f[~in_band & ~in_buf].tile_id.tolist()
        log(f"S2 fold{k}: Field{kk} valid rows {r0}-{r1} ({in_band.sum()} tiles = {in_band.mean():.1%}), "
            f"buffer rows {r0-1} & {r1+1} ({in_buf.sum()} tiles dropped from train)")
    SPLITS[f"S2_fold{k}"] = {"train": train, "valid": valid, "test": test, "buffer_unused": buffer}

# %%
# -- Cell 12 - Phase 1: leakage audit (asserts) -------------------------------
# Why: hard proof that no tile (or copy) is in two subsets, and how many eval tiles still touch train
nbrs = {}
for (k, r, c), t in key.items():
    nbrs[t.tile_id] = {key[(k, r + dr, c + dc)].tile_id for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if (dr, dc) != (0, 0) and (k, r + dr, c + dc) in key}
audit = []
for name, sp in SPLITS.items():
    sets = {s: set(v) for s, v in sp.items()}
    for a_, b_ in [("train", "valid"), ("train", "test"), ("valid", "test"),
                   ("buffer_unused", "train"), ("buffer_unused", "valid"), ("buffer_unused", "test")]:
        inter = sets[a_] & sets[b_]
        assert not inter, f"LEAK in {name}: {a_} ∩ {b_} = {sorted(inter)[:5]}"
    assert set().union(*sets.values()) == set(pool.tile_id), f"{name}: tiles missing from the split"
    if name.startswith("S2"):
        k = int(name[-1])
        assert all(t.startswith(f"F{k}_") for t in sets["test"]), "test must be exactly the held-out field"
        assert not any(t.startswith(f"F{k}_") for t in sets["train"] | sets["valid"]), "held-out field in train/valid"
    touch_v = sum(bool(nbrs[t] & sets["train"]) for t in sets["valid"])
    touch_t = sum(bool(nbrs[t] & sets["train"]) for t in sets["test"])
    print(f"{name}: OK - no tile_id in two subsets; valid tiles touching train: {touch_v}; test tiles touching train: {touch_t}")
    audit.append(dict(split=name, valid_touching_train=touch_v, test_touching_train=touch_t))
    if name.startswith("S2"):
        assert touch_v == 0 and touch_t == 0, f"{name}: spatial buffer failed"
log("Leakage audit passed for all splits (no tile_id in two subsets). Only raw un-augmented tiles are used; "
    "pre-augmented copies _0.._3 are not used at all.")
log(f"S1 neighbour exposure (64-px shared strips): {audit[0]}")

# %%
# -- Cell 13 - Phase 1: write split files + summary table (STOP POINT 2) ------
# Why: COCO-format files the training notebook can read directly; one image per tile (raw original)
cats_out = [{"id": 1, "name": "tree", "supercategory": "none"},
            {"id": 99, "name": "flagged_pending_user_decision", "supercategory": "none"}]
ann_index = {k: collections.defaultdict(list) for k in (1, 2, 3)}
for k in (1, 2, 3):
    for a in RAW[k]["annotations"]:
        ann_index[k][a["image_id"]].append(a)
by_tile = pool.set_index("tile_id")

def write_coco(name, subset, tile_list):
    images, anns = [], []
    for new_id, tid in enumerate(sorted(tile_list), start=1):
        t = by_tile.loc[tid]
        images.append(dict(id=new_id, file_name=t.path, width=int(t.width), height=int(t.height),
                           tile_id=tid, field=int(t.field), r=int(t.r), c=int(t.c),
                           merged_id=int(t.merged_id), raw_file=t.raw_file,
                           pad_w=int(t.pad_w), pad_h=int(t.pad_h),
                           file_size_on_disk=list(RESIZED.get(t.path, (TILE, TILE))),
                           note="resize file to width x height before use" if t.path in RESIZED else ""))
        for a in ann_index[int(t.field)][int(t.raw_image_id)]:
            tree = is_tree(int(t.field), a["category_id"])
            if not tree and CATEGORY_DECISION.get((int(t.field), NON_TREE[(int(t.field), a["category_id"])])) == "drop":
                continue                                   # deleted by user decision
            anns.append(dict(id=len(anns) + 1, image_id=new_id, category_id=1 if tree else 99,
                             orig_category=NON_TREE.get((int(t.field), a["category_id"]), "tree"),
                             segmentation=a["segmentation"], area=a.get("area"), bbox=a.get("bbox"),
                             iscrowd=a.get("iscrowd", 0)))
    out = dict(info=dict(split=name, subset=subset, seed=SEED, created=datetime.date.today().isoformat(),
                         source="pabna-tree-coco-raw (un-augmented)", note="file_name is an absolute Kaggle path"),
               images=images, annotations=anns, categories=cats_out)
    with open(os.path.join(SPLIT_DIR, f"{name}_{subset}.json"), "w") as f:
        json.dump(out, f)
    return len(images), len(anns)

summ = []
for name, sp in SPLITS.items():
    for subset in ("train", "valid", "test", "buffer_unused"):
        tl = sp[subset]
        if subset != "buffer_unused":
            n_img, n_ann = write_coco(name, subset, tl)
        else:
            n_img, n_ann = len(tl), int(by_tile.loc[tl].n_ann.sum()) if tl else 0
        d = by_tile.loc[tl] if tl else by_tile.iloc[:0]
        summ.append(dict(split=name, subset=subset, tiles=len(tl), images=n_img, annotations=n_ann,
                         F1_tiles=int((d.field == 1).sum()), F2_tiles=int((d.field == 2).sum()),
                         F3_tiles=int((d.field == 3).sum()),
                         tree_frac_all_px=float(d.tree_px.sum() / (len(d) * TILE * TILE)) if len(d) else np.nan,
                         tree_frac_valid_px=float(d.tree_px.sum() / d.valid_px.sum()) if len(d) else np.nan))
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(SPLIT_DIR, "split_summary.csv"), index=False)
pd.DataFrame(audit).to_csv(os.path.join(SPLIT_DIR, "leakage_audit.csv"), index=False)
with open(os.path.join(SPLIT_DIR, "split_tile_lists.json"), "w") as f:
    json.dump(SPLITS, f, indent=1)
print(summ.to_string(index=False))

log("Training plan agreed so far (not run here): train on raw tiles only, on-the-fly augmentation = Session 8 set "
    "(HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast(0.15,0.15,p=0.4), GaussianBlur(1-3,p=0.2)); "
    "valid/test un-augmented; old CoarseDropout/contour-merged masks no longer used.")
with open(os.path.join(OUT, "phase1_report.json"), "w") as f:
    json.dump(dict(summary=summ.to_dict("records"), audit=audit), f, indent=2, default=str)
with open(os.path.join(OUT, "run_log.md"), "w") as f:
    f.write("# run_log (Phase 0-1)\n\n" + "\n".join(f"- {m}" for m in RUN_LOG) + "\n")
with open(os.path.join(OUT, "environment_phase01.txt"), "w") as f:
    f.write(f"python {platform.python_version()}\nnumpy {np.__version__}\npandas {pd.__version__}\n"
            f"opencv {cv2.__version__}\npycocotools {getattr(pycocotools, '__version__', 'n/a')}\n"
            f"platform {platform.platform()}\n")
zp = os.path.join(WORK, "results_phase01.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z_:
    for p in walk_files(OUT):
        z_.write(p, os.path.relpath(p, WORK))
print("=" * 78)
print(f"PHASE 1 DONE - split files in {SPLIT_DIR}; everything zipped to {zp}")
print("STOP: send the Phase 0 + Phase 1 outputs for review before any training.")
print("=" * 78)
