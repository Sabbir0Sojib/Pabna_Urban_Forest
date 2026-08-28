"""Step 1. Train a canopy segmentation model on annotated UAV image tiles.

Input  : COCO-format tiles in paths.seg_dataset_dir (train/ valid/ test/)
Output : checkpoints/<architecture>_<bands>_best.pt  and a metrics CSV

The input can be 3-channel RGB or 4-channel RGB + ExG, controlled by
`segmentation.use_exg` in config.yaml. That switch is the RGB versus RGB+ExG
comparison reported in the manuscript.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import cv2
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO

from utils import banner, excess_green, load_config, project_path


class CocoCanopyDataset(Dataset):
    """Reads a Roboflow-style COCO folder and rasterises the polygons."""

    def __init__(self, folder, img_size, use_exg, augment=False):
        self.folder = folder
        self.img_size = img_size
        self.use_exg = use_exg
        self.augment = augment
        ann_file = os.path.join(folder, "_annotations.coco.json")
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.imgs[img_id]
        img = cv2.imread(os.path.join(self.folder, info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            mask = np.maximum(mask, self.coco.annToMask(ann))

        size = (self.img_size, self.img_size)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:
                img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k:
                img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()

        x = img.astype(np.float32) / 255.0            # (H, W, 3)
        if self.use_exg:
            exg = excess_green(x)[..., None]          # (H, W, 1)
            x = np.concatenate([x, exg], axis=-1)     # (H, W, 4)

        x = torch.from_numpy(x.transpose(2, 0, 1))
        y = torch.from_numpy(mask.astype(np.float32))[None]
        return x, y


def build_model(architecture, encoder, in_channels):
    return smp.create_model(
        arch=architecture,
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
    )


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """Accumulate confusion counts over the loader and return F1 and IoU."""
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x)) > threshold).float()
        tp += torch.sum((pred == 1) & (y == 1)).item()
        fp += torch.sum((pred == 1) & (y == 0)).item()
        fn += torch.sum((pred == 0) & (y == 1)).item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def main():
    cfg = load_config()
    seg = cfg["segmentation"]
    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = 4 if seg["use_exg"] else 3
    tag = f"{seg['architecture']}_{'rgbexg' if seg['use_exg'] else 'rgb'}"

    banner(f"Step 1  training {tag} on {device}")

    root = project_path(cfg["paths"]["seg_dataset_dir"])
    loaders = {}
    for split, augment in (("train", True), ("valid", False), ("test", False)):
        ds = CocoCanopyDataset(os.path.join(root, split), seg["img_size"],
                               seg["use_exg"], augment=augment)
        loaders[split] = DataLoader(
            ds,
            batch_size=seg["batch_size"],
            shuffle=(split == "train"),
            num_workers=seg["num_workers"],
            pin_memory=(device == "cuda"),
            drop_last=(split == "train"),
        )
        print(f"  {split:>5}: {len(ds)} tiles")

    model = build_model(seg["architecture"], seg["encoder"], in_channels).to(device)
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=seg["learning_rate"],
                                  weight_decay=seg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=seg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    ckpt_dir = project_path(cfg["paths"]["checkpoint_dir"])
    best_path = os.path.join(ckpt_dir, f"{tag}_best.pt")
    history, best_f1 = [], -1.0

    for epoch in range(1, seg["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = 0.5 * dice(logits, y) + 0.5 * bce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        val = evaluate(model, loaders["valid"], device)
        history.append({"epoch": epoch,
                        "train_loss": running / max(len(loaders["train"]), 1),
                        **val})
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"state_dict": model.state_dict(),
                        "architecture": seg["architecture"],
                        "encoder": seg["encoder"],
                        "in_channels": in_channels,
                        "use_exg": seg["use_exg"],
                        "val_f1": best_f1}, best_path)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  loss {history[-1]['train_loss']:.4f}"
                  f"  val F1 {val['f1']:.4f}  val IoU {val['iou']:.4f}")

    out_dir = project_path(cfg["paths"]["output_dir"])
    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"{tag}_history.csv"), index=False)

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test = evaluate(model, loaders["test"], device)
    with open(os.path.join(out_dir, f"{tag}_test_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(test, fh, indent=2)

    print(f"\n  best validation F1 : {best_f1:.4f}")
    print(f"  test  F1 / IoU     : {test['f1']:.4f} / {test['iou']:.4f}")
    print(f"  checkpoint         : {best_path}")


if __name__ == "__main__":
    main()
