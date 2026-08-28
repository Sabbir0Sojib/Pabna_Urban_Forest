"""Shared helpers: configuration, raster IO, the excess-green band, metrics."""

import os

import numpy as np
import rasterio
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    """Read config.yaml from the repository root."""
    path = path or os.path.join(HERE, "config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("output_dir", "checkpoint_dir"):
        os.makedirs(os.path.join(HERE, cfg["paths"][key]), exist_ok=True)
    return cfg


def project_path(*parts):
    """Build a path relative to the repository root."""
    return os.path.join(HERE, *parts)


# ---------------------------------------------------------------- raster IO

def read_raster(path, band=None):
    """Return (array, profile). With band=None all bands are read."""
    with rasterio.open(path) as src:
        arr = src.read(band) if band else src.read()
        profile = src.profile.copy()
    return arr, profile


def pixel_area_m2(profile):
    """Ground area of one pixel, in square metres."""
    return abs(profile["transform"].a * profile["transform"].e)


def write_raster(path, array, profile, dtype=None, nodata=None):
    """Write a single-band raster, keeping the grid of `profile`."""
    array = np.asarray(array)
    out = profile.copy()
    out.update(count=1, dtype=dtype or str(array.dtype), compress="lzw")
    if nodata is not None:
        out.update(nodata=nodata)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **out) as dst:
        dst.write(array.astype(out["dtype"]), 1)
    return path


# ------------------------------------------------------------- spectral band

def excess_green(rgb):
    """Excess green index, ExG = 2G - R - B, rescaled to [0, 1].

    Parameters
    ----------
    rgb : array, either (3, H, W) or (H, W, 3), any numeric range.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.shape[0] == 3 and rgb.ndim == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    else:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if r.max() > 1.5:                     # assume 8 bit input
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    exg = 2.0 * g - r - b                 # range [-2, 2]
    return ((exg + 2.0) / 4.0).astype(np.float32)


def normalized_difference(a, b):
    """(a - b) / (a + b), guarded against division by zero."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = a + b
    return np.where(np.abs(denom) > 1e-6, (a - b) / denom, 0.0).astype(np.float32)


# -------------------------------------------------------------------- metrics

def binary_metrics(y_true, y_pred):
    """Precision, recall, F1, IoU, overall accuracy and Cohen's kappa."""
    y_true = np.asarray(y_true).ravel().astype(bool)
    y_pred = np.asarray(y_pred).ravel().astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    total = tp + fp + fn + tn

    def div(num, den):
        return float(num) / float(den) if den else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    accuracy = div(tp + tn, total)
    expected = div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total * total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "iou": div(tp, tp + fp + fn),
        "overall_accuracy": accuracy,
        "kappa": div(accuracy - expected, 1 - expected),
    }


def gini(values):
    """Gini coefficient of a non-negative distribution."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n)


def banner(text):
    print("\n" + "=" * 68)
    print(f"  {text}")
    print("=" * 68)
