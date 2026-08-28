"""Step 2. Apply the trained model to the full UAV orthomosaics.

Input  : checkpoints/<tag>_best.pt and the field orthomosaics
Output : outputs/<Field>_canopy_mask.tif, a georeferenced 0/1 mask
         outputs/<Field>_canopy_prob.tif, the probability surface

Inference is tiled with overlap and overlapping predictions are averaged, so
the result has no visible tile seams.
"""

import os

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

import segmentation_models_pytorch as smp

from utils import (banner, excess_green, load_config, pixel_area_m2,
                   project_path, write_raster)


def load_model(checkpoint_path, device, seg):
    """Load a checkpoint from step 1, or a bare state dict from elsewhere.

    Weights published on a model hub are not always saved with the metadata that
    step 1 records, so fall back to the geometry declared in config.yaml.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("architecture", seg["architecture"])
        encoder = ckpt.get("encoder", seg["encoder"])
        use_exg = ckpt.get("use_exg", seg["use_exg"])
        in_channels = ckpt.get("in_channels", 4 if use_exg else 3)
        state = ckpt["state_dict"]
    else:
        arch, encoder = seg["architecture"], seg["encoder"]
        use_exg = seg["use_exg"]
        in_channels = 4 if use_exg else 3
        state = ckpt

    model = smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, use_exg


@torch.no_grad()
def predict_raster(model, use_exg, ortho_path, tile, overlap, device):
    """Return (probability map, rasterio profile) for one orthomosaic."""
    with rasterio.open(ortho_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        prob = np.zeros((height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        step = max(tile - overlap, 1)
        for row in range(0, height, step):
            for col in range(0, width, step):
                h = min(tile, height - row)
                w = min(tile, width - col)
                if h < 32 or w < 32:
                    continue

                patch = src.read(indexes=[1, 2, 3],
                                 window=Window(col, row, w, h)).astype(np.float32)
                if patch.max() > 1.5:
                    patch /= 255.0

                if use_exg:
                    patch = np.concatenate([patch, excess_green(patch)[None]], axis=0)

                # pad to a multiple of 32 for the encoder
                pad_h = (-h) % 32
                pad_w = (-w) % 32
                if pad_h or pad_w:
                    patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

                x = torch.from_numpy(patch)[None].to(device)
                out = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                prob[row:row + h, col:col + w] += out[:h, :w]
                count[row:row + h, col:col + w] += 1.0

    prob = np.divide(prob, count, out=np.zeros_like(prob), where=count > 0)
    return prob, profile


def main():
    cfg = load_config()
    seg, inf = cfg["segmentation"], cfg["inference"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{seg['architecture']}_{'rgbexg' if seg['use_exg'] else 'rgb'}"

    banner(f"Step 2  tiled inference with {tag} on {device}")

    configured = seg.get("checkpoint") or ""
    checkpoint = (project_path(configured) if configured else
                  project_path(cfg["paths"]["checkpoint_dir"], f"{tag}_best.pt"))
    if not os.path.exists(checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            "Either run step 1 to train one, or download the published weights and\n"
            "set segmentation.checkpoint in config.yaml to the downloaded file:\n"
            "  https://huggingface.co/Sabbir12345/P6_S4_SegFormer_RGBExG_100ep")

    print(f"  checkpoint {checkpoint}")
    model, use_exg = load_model(checkpoint, device, seg)

    out_dir = project_path(cfg["paths"]["output_dir"])
    for field, rel_path in cfg["paths"]["field_orthomosaics"].items():
        ortho = project_path(rel_path)
        if not os.path.exists(ortho):
            print(f"  {field}: orthomosaic missing, skipped ({rel_path})")
            continue

        prob, profile = predict_raster(model, use_exg, ortho,
                                       inf["tile_size"], inf["tile_overlap"], device)
        mask = (prob >= inf["probability_threshold"]).astype(np.uint8)

        mask_path = os.path.join(out_dir, f"{field}_canopy_mask.tif")
        write_raster(mask_path, mask, profile, dtype="uint8", nodata=255)
        write_raster(os.path.join(out_dir, f"{field}_canopy_prob.tif"),
                     prob.astype(np.float32), profile, dtype="float32")

        canopy_ha = int(mask.sum()) * pixel_area_m2(profile) / 10000.0
        print(f"  {field}: canopy {canopy_ha:8.3f} ha  ->  {os.path.basename(mask_path)}")


if __name__ == "__main__":
    main()
