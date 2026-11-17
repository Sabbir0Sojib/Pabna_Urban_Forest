"""Pabna UAV tree segmentation: Phase 2 (train), Phase 3 (tune on validation only), Phase 4 (test once).

One process = one run (one model, one split, one seed). Launched by the Kaggle driver notebook,
one process per GPU. Every setting is written to <out>/<run_id>/config.json and run_log.md.

Scientific rules enforced in code:
  * train / valid / test tiles come from the leakage-free split files (Phase 1); asserted disjoint
  * only un-augmented raw tiles; augmentation happens on the fly, training only
  * checkpoint, threshold, small-blob removal and TTA are chosen on VALIDATION only and written to
    tuning.json BEFORE the test set is read
  * the test set is predicted exactly once; a flag file blocks a second test run
  * black padding at ortho edges is ignored in the loss and excluded from every metric
"""
import os, sys, json, time, math, glob, random, hashlib, argparse, datetime, platform

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_utils

p = argparse.ArgumentParser()
p.add_argument("--run_id", required=True)
p.add_argument("--split", required=True)                 # S1 | S2_fold1 | S2_fold2 | S2_fold3
p.add_argument("--input", required=True, choices=["rgbexg", "rgb"])
p.add_argument("--seed_idx", type=int, default=0)
p.add_argument("--splits_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--pretrained", default="nvidia/mit-b2")
p.add_argument("--img", type=int, default=512)
p.add_argument("--batch", type=int, default=6)
p.add_argument("--lr", type=float, default=6e-5)
p.add_argument("--head_lr_mult", type=float, default=10.0)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--total_steps", type=int, default=69600)  # = old 4,176 images x 100 epochs / batch 6
p.add_argument("--eval_every", type=int, default=696)     # = one old epoch
p.add_argument("--warmup_frac", type=float, default=0.01)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--gsd", type=float, default=0.026)
p.add_argument("--smoke", action="store_true")
p.add_argument("--tiny_model", action="store_true")      # local CPU testing only
p.add_argument("--resume_from", default="")
args = p.parse_args()

TILE = 640
OUT = os.path.join(args.out, args.run_id)
if os.path.exists(os.path.join(OUT, "TEST_DONE.flag")) and not args.smoke:
    print(f"{args.run_id}: test already done; this run is final and will not be retrained or retested", flush=True)
    sys.exit(0)
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(msg)
    with open(os.path.join(OUT, "run_log.md"), "a") as f:
        f.write(f"- {msg}\n")

# ---------------------------------------------------------------- seed and determinism
FOLD_INDEX = {"S1": 0, "S2_fold1": 1, "S2_fold2": 2, "S2_fold3": 3}[args.split]
SEED = 20261106 + 1000 * FOLD_INDEX + args.seed_idx          # identical for RGB+ExG and RGB-only (paired)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)   # bilinear upsampling backward has no deterministic CUDA kernel
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"
IN_CH = 4 if args.input == "rgbexg" else 3

if args.smoke:
    args.total_steps, args.eval_every = 60, 30

cfg = vars(args).copy()
cfg.update(seed=SEED, in_channels=IN_CH, device=str(DEV),
           gpu=torch.cuda.get_device_name(0) if AMP else "cpu",
           torch=torch.__version__, python=platform.python_version(), numpy=np.__version__,
           opencv=cv2.__version__, started=datetime.datetime.now().isoformat(timespec="seconds"))
try:
    import transformers
    cfg["transformers"] = transformers.__version__
except Exception:
    pass
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
log(f"RUN {args.run_id}: split={args.split} input={args.input} seed={SEED} device={cfg['gpu']}")

# ---------------------------------------------------------------- data
def load_split(subset):
    path = os.path.join(args.splits_dir, f"{args.split}_{subset}.json")
    with open(path) as f:
        coco = json.load(f)
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    items = []
    for im in coco["images"]:
        fp = im["file_name"]
        if not os.path.exists(fp):                           # remap if the dataset is mounted elsewhere
            hits = glob.glob(f"/kaggle/input/**/{os.path.basename(fp)}", recursive=True)
            assert len(hits) == 1, f"cannot locate {fp}"
            fp = hits[0]
        items.append(dict(tile_id=im["tile_id"], field=im["field"], r=im["r"], c=im["c"], path=fp,
                          H=im["height"], W=im["width"], pad_w=im.get("pad_w", 0), pad_h=im.get("pad_h", 0),
                          anns=[a for a in by_img.get(im["id"], []) if a["category_id"] == 1]))
    return items

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

def load_item(it):
    im = cv2.imread(it["path"], cv2.IMREAD_COLOR)
    assert im is not None, it["path"]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (it["H"], it["W"]):                  # 2 raw files are stored at 1024 px (none left in splits)
        im = cv2.resize(im, (it["W"], it["H"]), interpolation=cv2.INTER_AREA)
    gt = np.zeros((it["H"], it["W"]), np.uint8)
    for a in it["anns"]:
        gt |= decode(a["segmentation"], it["H"], it["W"])
    valid = np.zeros_like(gt, dtype=bool)
    valid[: it["H"] - it["pad_h"], : it["W"] - it["pad_w"]] = True
    return im, gt, valid

TRAIN, VALID, TEST = load_split("train"), load_split("valid"), load_split("test")
ids = {s: {t["tile_id"] for t in L} for s, L in (("train", TRAIN), ("valid", VALID), ("test", TEST))}
assert not (ids["train"] & ids["valid"]) and not (ids["train"] & ids["test"]) and not (ids["valid"] & ids["test"]), "LEAK"
if args.smoke:                                               # smoke test never touches the real test set
    TRAIN, VALID = TRAIN[:24], VALID[:8]
    TEST = VALID
log(f"tiles: train {len(TRAIN)}, valid {len(VALID)}, test {len(TEST)}{' (SMOKE: test := valid subset)' if args.smoke else ''}")

def preload(items):
    out = [load_item(it) for it in items]
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]
t0 = time.time()
TR_IMG, TR_GT, TR_VALID = preload(TRAIN)
VA_IMG, VA_GT, VA_VALID = preload(VALID)
log(f"preloaded train+valid in {time.time() - t0:.0f}s; train tree px share {np.mean([g[v].mean() for g, v in zip(TR_GT, TR_VALID)]):.3f}")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
def to_tensor(rgb_u8):
    """RGB uint8 -> CxHxW float. ExG (Session 8 formula) is computed from the SAME (augmented) RGB."""
    f = rgb_u8.astype(np.float32) / 255.0
    chans = [(f - MEAN) / STD]
    if IN_CH == 4:
        exg = (2 * f[..., 1] - f[..., 0] - f[..., 2] + 1.0) / 3.0
        chans.append(exg[..., None])
    return torch.from_numpy(np.concatenate(chans, axis=2).transpose(2, 0, 1).copy())

def augment(img, lab, rng):
    """Session 8 on-the-fly set: HorizontalFlip .5, VerticalFlip .5, RandomRotate90 .5,
    RandomBrightnessContrast(.15,.15,p=.4), GaussianBlur(ksize 1 or 3, p=.2). Geometry on image AND label."""
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5:
        img, lab = img[::-1], lab[::-1]
    if rng.random() < 0.5:
        k = int(rng.integers(0, 4))
        img, lab = np.rot90(img, k), np.rot90(lab, k)
    img = np.ascontiguousarray(img); lab = np.ascontiguousarray(lab)
    if rng.random() < 0.4:
        alpha = 1.0 + rng.uniform(-0.15, 0.15)
        beta = rng.uniform(-0.15, 0.15) * 255.0
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        k = int(rng.choice([1, 3]))
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)
    return img, lab

def seed_worker(w):
    np.random.seed((SEED + 7919 * w + int(torch.initial_seed() % 100003)) % (2**31 - 1))

class TrainSet(torch.utils.data.Dataset):
    def __len__(self):
        return len(TRAIN)
    def __getitem__(self, i):
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))   # worker RNG seeded in worker_init_fn
        img = cv2.resize(TR_IMG[i], (args.img, args.img), interpolation=cv2.INTER_LINEAR)
        lab = TR_GT[i].copy()
        lab[~TR_VALID[i]] = 255                               # padding ignored in the loss
        lab = cv2.resize(lab, (args.img, args.img), interpolation=cv2.INTER_NEAREST)
        img, lab = augment(img, lab, rng)
        return to_tensor(img), torch.from_numpy(lab.astype(np.int64))

# ---------------------------------------------------------------- model
from transformers import SegformerForSemanticSegmentation, SegformerConfig
if args.tiny_model:
    conf = SegformerConfig(num_labels=2, hidden_sizes=[16, 32, 48, 64], depths=[1, 1, 1, 1],
                           decoder_hidden_size=32, num_attention_heads=[1, 2, 3, 4])
    model = SegformerForSemanticSegmentation(conf)
else:
    model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained, num_labels=2,
                                                             ignore_mismatched_sizes=True)
if IN_CH == 4:
    # first patch-embedding conv (the only Conv2d with 3 input channels); attribute paths differ between
    # transformers versions, so it is found by type and replaced by name
    first = [(n, m) for n, m in model.segformer.named_modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3]
    assert len(first) == 1, f"expected one 3-channel conv, found {[n for n, _ in first]}"
    conv_name, old = first[0]
    new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3] = old.weight[:, 1]                     # ExG channel initialised from green (Session 8)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    parent = model.segformer.get_submodule(conv_name.rsplit(".", 1)[0])
    setattr(parent, conv_name.rsplit(".", 1)[1], new)
    model.config.num_channels = 4
    for m in model.segformer.modules():                      # some versions also store num_channels on the embedding
        if hasattr(m, "num_channels") and getattr(m, "num_channels") == 3:
            m.num_channels = 4
    log(f"4-channel input: replaced {conv_name} (ExG weights copied from green)")
model.to(DEV)
log(f"model {('tiny-random' if args.tiny_model else args.pretrained)}, in_channels={IN_CH}, "
    f"params={sum(q.numel() for q in model.parameters()) / 1e6:.1f}M")

def logits_full(x):
    out = model(pixel_values=x).logits                        # 1/4 resolution
    return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

def loss_fn(logits, lab):
    ce = F.cross_entropy(logits.float(), lab, ignore_index=255)
    valid = (lab != 255).float()
    p1 = torch.softmax(logits.float(), dim=1)[:, 1] * valid
    t = (lab == 1).float()
    dice = 1 - (2 * (p1 * t).sum() + 1e-6) / (p1.sum() + t.sum() + 1e-6)
    return 0.5 * ce + 0.5 * dice

enc = [q for n, q in model.named_parameters() if not n.startswith("decode_head")]
head = [q for n, q in model.named_parameters() if n.startswith("decode_head")]
opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": head, "lr": args.lr * args.head_lr_mult}],
                        weight_decay=args.wd)
WARM = max(1, int(args.warmup_frac * args.total_steps))
def lr_lambda(step):
    if step < WARM:
        return (step + 1) / WARM
    return 0.5 * (1 + math.cos(math.pi * (step - WARM) / max(1, args.total_steps - WARM)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)

# ---------------------------------------------------------------- prediction helpers (native 640 px)
@torch.no_grad()
def predict_prob(img_u8, tta=False):
    model.eval()
    x = to_tensor(cv2.resize(img_u8, (args.img, args.img), interpolation=cv2.INTER_LINEAR))[None].to(DEV)
    views = [x] + ([torch.flip(x, [3]), torch.flip(x, [2])] if tta else [])
    probs = []
    for k, v in enumerate(views):
        with torch.autocast("cuda", enabled=AMP):
            lg = logits_full(v)
        pr = torch.softmax(lg.float(), 1)[:, 1:2]
        if k == 1: pr = torch.flip(pr, [3])
        if k == 2: pr = torch.flip(pr, [2])
        probs.append(pr)
    pr = torch.stack(probs).mean(0)
    pr = F.interpolate(pr, size=(TILE, TILE), mode="bilinear", align_corners=False)
    return pr[0, 0].cpu().numpy().astype(np.float32)

def remove_small(pred, amin_px):
    if amin_px <= 0:
        return pred
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= amin_px
    return keep[lab]

def confusion(pred, gt, valid):
    p, g = pred[valid], gt[valid].astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum()); tn = int((~p & ~g).sum())
    return tp, fp, fn, tn

def scores(tp, fp, fn, tn=0):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    oa = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return dict(precision=prec, recall=rec, F1=f1, IoU=iou, OA=oa)

@torch.no_grad()
def evaluate_valid():
    tp = fp = fn = 0; tot_loss = 0.0
    for img, gt, va in zip(VA_IMG, VA_GT, VA_VALID):
        pr = predict_prob(img)
        a, b, c, _ = confusion(pr >= 0.5, gt, va)
        tp += a; fp += b; fn += c
        lab = gt.copy(); lab[~va] = 255
        lab_t = torch.from_numpy(cv2.resize(lab, (args.img, args.img), interpolation=cv2.INTER_NEAREST).astype(np.int64))[None].to(DEV)
        x = to_tensor(cv2.resize(img, (args.img, args.img), interpolation=cv2.INTER_LINEAR))[None].to(DEV)
        with torch.autocast("cuda", enabled=AMP):
            tot_loss += float(loss_fn(logits_full(x), lab_t))
    s = scores(tp, fp, fn)
    return s["F1"], s["IoU"], tot_loss / len(VA_IMG)

# ---------------------------------------------------------------- Phase 2: training
step, best_f1, best_step = 0, -1.0, -1
hist_path = os.path.join(OUT, "train_log.csv")
LATEST, BEST = os.path.join(OUT, "latest.pt"), os.path.join(OUT, "best.pt")
resume = args.resume_from or (LATEST if os.path.exists(LATEST) else "")
if resume and os.path.exists(resume):
    ck = torch.load(resume, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"]); step, best_f1, best_step = ck["step"], ck["best_f1"], ck["best_step"]
    if os.path.dirname(resume) != OUT:                       # copy best checkpoint and log from previous session
        for fn_ in ("best.pt", "train_log.csv"):
            src = os.path.join(os.path.dirname(resume), fn_)
            if os.path.exists(src):
                import shutil; shutil.copy(src, os.path.join(OUT, fn_))
    log(f"RESUMED from {resume} at step {step} (best valid F1 {best_f1:.4f} at step {best_step})")
elif not os.path.exists(hist_path):
    with open(hist_path, "w") as f:
        f.write("step,old_epoch,train_loss,valid_loss,valid_F1,valid_IoU,lr,seconds\n")

if step < args.total_steps:
    gen = torch.Generator(); gen.manual_seed(SEED + step)
    sampler = torch.utils.data.RandomSampler(TrainSet(), replacement=True,
                                             num_samples=(args.total_steps - step) * args.batch, generator=gen)
    loader = torch.utils.data.DataLoader(TrainSet(), batch_size=args.batch, sampler=sampler,
                                         num_workers=args.workers, drop_last=True, pin_memory=AMP,
                                         persistent_workers=args.workers > 0,
                                         worker_init_fn=seed_worker)
    log(f"training {args.total_steps - step} steps (batch {args.batch}, {args.img}px, lr {args.lr}, head x{args.head_lr_mult}, "
        f"warmup {WARM} steps then cosine), validation every {args.eval_every} steps")
    model.train(); run_loss, n_loss, t_last, t_start = 0.0, 0, time.time(), time.time()
    for x, lab in loader:
        x, lab = x.to(DEV, non_blocking=True), lab.to(DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=AMP):
            loss = loss_fn(logits_full(x), lab)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1; run_loss += float(loss); n_loss += 1
        if step == 50 or (args.smoke and step == 10):
            rate = (time.time() - t_start) / n_loss
            log(f"speed: {rate:.3f} s/step -> full run about {rate * args.total_steps / 3600:.1f} h of training")
        if step % args.eval_every == 0 or step == args.total_steps:
            vf1, viou, vloss = evaluate_valid(); model.train()
            dt = time.time() - t_last; t_last = time.time()
            with open(hist_path, "a") as f:
                f.write(f"{step},{step / args.eval_every:.2f},{run_loss / max(n_loss, 1):.5f},{vloss:.5f},"
                        f"{vf1:.5f},{viou:.5f},{sched.get_last_lr()[0]:.3e},{dt:.0f}\n")
            flag = ""
            if vf1 > best_f1:
                best_f1, best_step, flag = vf1, step, "  *best*"
                torch.save({"model": model.state_dict(), "step": step, "valid_F1": vf1}, BEST)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "best_f1": best_f1, "best_step": best_step}, LATEST)
            log(f"step {step}/{args.total_steps}  train_loss {run_loss / max(n_loss, 1):.4f}  valid_loss {vloss:.4f}  "
                f"valid F1 {vf1:.4f} IoU {viou:.4f}{flag}")
            run_loss, n_loss = 0.0, 0
        if step >= args.total_steps:
            break
log(f"training finished; best valid F1 {best_f1:.4f} at step {best_step}")

try:                                                       # training curves
    import pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = pd.read_csv(hist_path)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h.old_epoch, h.train_loss, label="train"); ax[0].plot(h.old_epoch, h.valid_loss, label="valid")
    ax[0].set_xlabel("epoch (696 steps)"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(h.old_epoch, h.valid_F1, label="valid F1"); ax[1].plot(h.old_epoch, h.valid_IoU, label="valid IoU")
    ax[1].axvline(best_step / args.eval_every, ls="--", c="grey"); ax[1].set_xlabel("epoch (696 steps)"); ax[1].legend()
    fig.suptitle(args.run_id); plt.tight_layout(); plt.savefig(os.path.join(OUT, "train_curves.png"), dpi=120); plt.close()
except Exception as e:
    log(f"curve plot failed: {e}")

# ---------------------------------------------------------------- Phase 3: tuning on VALIDATION only
TUNE = os.path.join(OUT, "tuning.json")
ck = torch.load(BEST, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
if not os.path.exists(TUNE):
    THRS = [round(0.30 + 0.05 * i, 2) for i in range(9)]
    AMINS = [0.0, 0.5, 1.0, 2.0]
    grid = []
    for tta in (False, True):
        probs = [predict_prob(im, tta) for im in VA_IMG]
        for amin in AMINS:
            amin_px = int(round(amin / args.gsd ** 2))
            for thr in THRS:
                tp = fp = fn = 0
                for pr, gt, va in zip(probs, VA_GT, VA_VALID):
                    pred = remove_small((pr >= thr) & va, amin_px)
                    a, b, c, _ = confusion(pred, gt, va); tp += a; fp += b; fn += c
                s = scores(tp, fp, fn)
                grid.append(dict(tta=tta, A_min_m2=amin, A_min_px=amin_px, threshold=thr, valid_F1=s["F1"], valid_IoU=s["IoU"]))
    import pandas as pd
    g = pd.DataFrame(grid)
    g.to_csv(os.path.join(OUT, "tuning_grid_valid.csv"), index=False)
    # max valid F1; ties broken toward the simplest setting (no TTA, smaller A_min, threshold nearest 0.5)
    g["_k"] = list(zip(-g.valid_F1.round(6), g.tta.astype(int), g.A_min_m2, (g.threshold - 0.5).abs()))
    best = g.sort_values("_k").iloc[0]
    tuning = dict(run_id=args.run_id, checkpoint_step=int(ck["step"]), checkpoint_valid_F1_at_0p5=float(ck["valid_F1"]),
                  threshold=float(best.threshold), A_min_m2=float(best.A_min_m2), A_min_px=int(best.A_min_px),
                  tta=bool(best.tta), valid_F1_tuned=float(best.valid_F1), valid_IoU_tuned=float(best.valid_IoU),
                  gsd_m=args.gsd, chosen_on="validation subset only", created=datetime.datetime.now().isoformat(timespec="seconds"))
    tuning["sha256"] = hashlib.sha256(json.dumps(tuning, sort_keys=True).encode()).hexdigest()
    with open(TUNE, "w") as f:
        json.dump(tuning, f, indent=2)
    log(f"TUNING (valid only) saved before test: thr {tuning['threshold']}, A_min {tuning['A_min_m2']} m2, "
        f"TTA {tuning['tta']}, valid F1 {tuning['valid_F1_tuned']:.4f}")
with open(TUNE) as f:
    tuning = json.load(f)

# ---------------------------------------------------------------- Phase 4: test ONCE
FLAG = os.path.join(OUT, "TEST_DONE.flag")
if args.smoke:
    FLAG = os.path.join(OUT, "SMOKE_TEST_ON_VALID.flag")
if os.path.exists(FLAG):
    log("test already done for this run; refusing to run it again"); sys.exit(0)
MASK_DIR = os.path.join(OUT, "test_pred_masks"); os.makedirs(MASK_DIR, exist_ok=True)

def objects(pred, gt, valid):
    """Instances = 8-connected components inside the valid area; one-to-one match if IoU >= 0.5
    (with IoU >= 0.5 a match is unique, so greedy matching is exact)."""
    ng, lg = cv2.connectedComponents((gt & valid).astype(np.uint8), connectivity=8)
    npd, lp = cv2.connectedComponents((pred & valid).astype(np.uint8), connectivity=8)
    if ng <= 1 or npd <= 1:
        return ng - 1, npd - 1, 0
    inter = np.bincount((lg.astype(np.int64) * npd + lp).ravel(), minlength=ng * npd).reshape(ng, npd)
    ag, ap = inter.sum(1), inter.sum(0)
    i = inter[1:, 1:]
    iou = i / (ag[1:, None] + ap[None, 1:] - i)
    return ng - 1, npd - 1, int((iou >= 0.5).sum())

rows = []
t0 = time.time()
for it in TEST:
    img, gt, va = load_item(it)
    pr = predict_prob(img, tuning["tta"])
    pred = remove_small((pr >= tuning["threshold"]) & va, tuning["A_min_px"])
    raw = (predict_prob(img, False) >= 0.5) & va if tuning["tta"] else (pr >= 0.5) & va   # untuned reference
    tp, fp, fn, tn = confusion(pred, gt, va)
    rtp, rfp, rfn, rtn = confusion(raw, gt, va)
    ng, npd, nm = objects(pred, gt.astype(bool), va)
    cv2.imwrite(os.path.join(MASK_DIR, f"{it['tile_id']}.png"), pred.astype(np.uint8) * 255)
    rows.append(dict(run_id=args.run_id, tile_id=it["tile_id"], field=it["field"], r=it["r"], c=it["c"],
                     valid_px=int(va.sum()), TP=tp, FP=fp, FN=fn, TN=tn, gt_objects=ng, pred_objects=npd,
                     matched_objects=nm, raw_TP=rtp, raw_FP=rfp, raw_FN=rfn, raw_TN=rtn))
with open(FLAG, "w") as f:
    f.write(datetime.datetime.now().isoformat())
import pandas as pd
pt = pd.DataFrame(rows)
pt.to_csv(os.path.join(OUT, "per_tile_test.csv"), index=False)
S = pt[["TP", "FP", "FN", "TN", "gt_objects", "pred_objects", "matched_objects", "raw_TP", "raw_FP", "raw_FN", "raw_TN"]].sum()
pix = scores(S.TP, S.FP, S.FN, S.TN)
obj_p = S.matched_objects / S.pred_objects if S.pred_objects else 0.0
obj_r = S.matched_objects / S.gt_objects if S.gt_objects else 0.0
summary = dict(run_id=args.run_id, split=args.split, input=args.input, seed=SEED, n_tiles=len(pt),
               prevalence=float((S.TP + S.FN) / (S.TP + S.FP + S.FN + S.TN)),
               **{k: int(S[k]) for k in ("TP", "FP", "FN", "TN")}, **pix,
               obj_precision=float(obj_p), obj_recall=float(obj_r),
               obj_F1=float(2 * obj_p * obj_r / (obj_p + obj_r)) if obj_p + obj_r else 0.0,
               gt_objects=int(S.gt_objects), pred_objects=int(S.pred_objects), matched_objects=int(S.matched_objects),
               raw_F1_thr0p5=scores(S.raw_TP, S.raw_FP, S.raw_FN, S.raw_TN)["F1"],
               threshold=tuning["threshold"], A_min_m2=tuning["A_min_m2"], TTA=tuning["tta"],
               best_step=tuning["checkpoint_step"], best_old_epoch=tuning["checkpoint_step"] / args.eval_every,
               smoke=args.smoke, test_seconds=round(time.time() - t0))
with open(os.path.join(OUT, "test_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
log(f"TEST ({'SMOKE: valid subset' if args.smoke else 'once'}): F1 {pix['F1']:.4f} IoU {pix['IoU']:.4f} "
    f"P {pix['precision']:.4f} R {pix['recall']:.4f}; object F1 {summary['obj_F1']:.4f}; n_tiles {len(pt)}")
for fn_ in ("latest.pt",):                                  # keep outputs small: best.pt is enough after the run
    if os.path.exists(os.path.join(OUT, fn_)) and not args.smoke:
        os.remove(os.path.join(OUT, fn_))
log("RUN COMPLETE")
