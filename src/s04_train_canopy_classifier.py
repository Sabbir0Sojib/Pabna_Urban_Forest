"""Step 4. Train a Random Forest canopy classifier on the satellite imagery.

Input  : satellite image (8 band, 3 m) and the UAV field canopy masks
Output : outputs/citywide_tree_mask.tif, cross-validation metrics, threshold
         sweep and feature importances

The labels come from the UAV canopy masks, aggregated to the satellite grid:
this is the cross-resolution transfer step. A satellite pixel is labelled
canopy when at least half of its area is canopy in the UAV mask.
"""

import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from utils import (banner, binary_metrics, load_config, normalized_difference,
                   pixel_area_m2, project_path, write_raster)


def build_features(image, window_sizes):
    """Stack spectral bands, vegetation indices and local means.

    SuperDove band order: coastal blue, blue, green I, green, yellow, red,
    red edge, NIR (1-indexed 1..8).
    """
    image = image.astype(np.float32)
    if image.max() > 2.0:                       # scale reflectance to 0..1
        image = image / 10000.0

    bands = [image[i] for i in range(image.shape[0])]
    names = [f"band_{i + 1}" for i in range(image.shape[0])]

    red = image[5] if image.shape[0] >= 8 else image[2]
    nir = image[7] if image.shape[0] >= 8 else image[-1]
    red_edge = image[6] if image.shape[0] >= 8 else image[-1]
    green = image[3] if image.shape[0] >= 8 else image[1]

    ndvi = normalized_difference(nir, red)
    ndre = normalized_difference(nir, red_edge)
    gndvi = normalized_difference(nir, green)
    bands += [ndvi, ndre, gndvi]
    names += ["ndvi", "ndre", "gndvi"]

    for size in window_sizes:                    # local texture context
        bands.append(ndimage.uniform_filter(ndvi, size=size))
        names.append(f"ndvi_mean_{size}")
        bands.append(ndimage.uniform_filter(ndvi ** 2, size=size)
                     - ndimage.uniform_filter(ndvi, size=size) ** 2)
        names.append(f"ndvi_var_{size}")

    return np.stack(bands, axis=0), names


def labels_from_uav(mask_path, satellite_path):
    """Resample a UAV canopy mask onto the satellite grid as a canopy fraction."""
    with rasterio.open(satellite_path) as ref:
        with rasterio.open(mask_path) as src:
            with WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                           width=ref.width, height=ref.height,
                           resampling=Resampling.average) as vrt:
                fraction = vrt.read(1).astype(np.float32)
                covered = vrt.read_masks(1) > 0
    return fraction, covered


def main():
    cfg = load_config()
    clf_cfg = cfg["classifier"]
    out_dir = project_path(cfg["paths"]["output_dir"])
    satellite = project_path(cfg["paths"]["satellite_image"])

    banner("Step 4  city-wide Random Forest canopy classifier")

    if not os.path.exists(satellite):
        raise SystemExit(f"Satellite image not found: {satellite}\n"
                         "PlanetScope imagery is not redistributable; see data/README.md.")

    with rasterio.open(satellite) as src:
        image = src.read()
        profile = src.profile.copy()

    features, names = build_features(image, clf_cfg["local_window_sizes"])
    n_features, height, width = features.shape
    flat = features.reshape(n_features, -1).T

    # ---- assemble the training set from the UAV field footprints
    X, y, groups = [], [], []
    for group_id, (field, rel) in enumerate(cfg["paths"]["field_masks"].items()):
        mask_path = project_path(rel)
        if not os.path.exists(mask_path):
            print(f"  {field}: reference mask missing, skipped")
            continue
        fraction, covered = labels_from_uav(mask_path, satellite)
        valid = covered & np.isfinite(fraction)
        idx = np.flatnonzero(valid.ravel())
        X.append(flat[idx])
        y.append((fraction.ravel()[idx] >= 0.5).astype(np.uint8))
        groups.append(np.full(idx.size, group_id))
        print(f"  {field}: {idx.size:>7} labelled satellite pixels"
              f"  ({y[-1].mean() * 100:.1f}% canopy)")

    if not X:
        raise SystemExit("No reference masks available, cannot train.")

    X = np.vstack(X)
    y = np.concatenate(y)
    groups = np.concatenate(groups)

    def make_model():
        return RandomForestClassifier(
            n_estimators=clf_cfg["n_estimators"],
            max_depth=clf_cfg["max_depth"],
            min_samples_leaf=clf_cfg["min_samples_leaf"],
            n_jobs=-1,
            class_weight="balanced",
            random_state=cfg["project"]["seed"],
        )

    # ---- cross-validation, grouped by field so folds are spatially disjoint
    n_splits = min(clf_cfg["cv_folds"], len(np.unique(groups)))
    fold_metrics = []
    if n_splits >= 2:
        for fold, (train_idx, test_idx) in enumerate(
                GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
            model = make_model().fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[test_idx])[:, 1]
            metrics = binary_metrics(y[test_idx], prob >= clf_cfg["decision_threshold"])
            fold_metrics.append({"fold": fold, **metrics})
            print(f"  fold {fold}: F1 {metrics['f1']:.4f}  IoU {metrics['iou']:.4f}")
        pd.DataFrame(fold_metrics).to_csv(
            os.path.join(out_dir, "classifier_cv_metrics.csv"), index=False)
        print(f"  grouped CV mean F1: "
              f"{np.mean([m['f1'] for m in fold_metrics]):.4f}")

    # ---- final model on all labelled pixels
    model = make_model().fit(X, y)

    sweep = []
    prob_train = model.predict_proba(X)[:, 1]
    for threshold in np.round(np.arange(0.05, 1.00, 0.05), 2):
        sweep.append({"threshold": float(threshold),
                      **binary_metrics(y, prob_train >= threshold)})
    pd.DataFrame(sweep).to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)
    best = max(sweep, key=lambda r: r["f1"])
    print(f"  best threshold on reference data: {best['threshold']:.2f} "
          f"(F1 {best['f1']:.4f})")

    pd.DataFrame({"feature": names, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)

    # ---- predict the whole study area
    prob_city = model.predict_proba(flat)[:, 1].reshape(height, width)
    mask_city = (prob_city >= clf_cfg["decision_threshold"]).astype(np.uint8)

    # drop segments below the minimum crown area
    px_area = pixel_area_m2(profile)
    min_px = max(int(round(cfg["crowns"]["min_crown_area_m2"] / px_area)), 1)
    labelled, n = ndimage.label(mask_city)
    if n:
        sizes = ndimage.sum(mask_city, labelled, range(1, n + 1))
        too_small = np.flatnonzero(sizes < min_px) + 1
        mask_city[np.isin(labelled, too_small)] = 0

    mask_path = os.path.join(out_dir, "citywide_tree_mask.tif")
    write_raster(mask_path, mask_city, profile, dtype="uint8", nodata=255)
    write_raster(os.path.join(out_dir, "citywide_tree_prob.tif"),
                 prob_city.astype(np.float32), profile, dtype="float32")

    canopy_ha = int(mask_city.sum()) * px_area / 10000.0
    total_ha = cfg["study_area"]["total_area_ha"]
    with open(os.path.join(out_dir, "citywide_canopy.json"), "w", encoding="utf-8") as fh:
        json.dump({"canopy_ha": canopy_ha,
                   "study_area_ha": total_ha,
                   "canopy_percent": 100.0 * canopy_ha / total_ha,
                   "threshold": clf_cfg["decision_threshold"]}, fh, indent=2)

    print(f"\n  city-wide canopy: {canopy_ha:.2f} ha "
          f"({100.0 * canopy_ha / total_ha:.2f}% of {total_ha:.2f} ha)")
    print(f"  mask -> {mask_path}")


if __name__ == "__main__":
    main()
