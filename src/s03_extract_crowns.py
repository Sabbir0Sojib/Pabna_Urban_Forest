"""Step 3. Delineate individual crowns from the field canopy masks.

Input  : field canopy masks (the reference masks when configured, otherwise the
         step 2 predictions), and optionally a canopy height model per field
Output : outputs/crowns_<Field>.csv     one row per crown
         outputs/crowns_<Field>.gpkg    the same crowns as polygons
         outputs/crown_summary.csv      per-field summary

Marker-controlled watershed separates touching crowns: the distance transform
of the canopy mask peaks near each crown centre, and those peaks seed the
watershed basins.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import xy as transform_xy
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy import ndimage
from shapely.geometry import shape
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils import banner, load_config, pixel_area_m2, project_path, read_raster


def delineate(mask, min_distance_px):
    """Label individual crowns in a binary canopy mask."""
    mask = mask.astype(bool)
    distance = ndimage.distance_transform_edt(mask)
    peaks = peak_local_max(distance, min_distance=min_distance_px, labels=mask)

    markers = np.zeros(distance.shape, dtype=np.int32)
    for i, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = i
    if markers.max() == 0:                       # no peaks found: use blobs
        markers, _ = ndimage.label(mask)
    return watershed(-distance, markers, mask=mask)


def crown_polygon(sub_mask, row_off, col_off, transform):
    """Return the crown outline as a shapely polygon in map coordinates."""
    local = window_transform(
        Window(col_off, row_off, sub_mask.shape[1], sub_mask.shape[0]), transform)
    for geom, value in features.shapes(sub_mask.astype(np.uint8),
                                       mask=sub_mask, transform=local):
        if value == 1:
            return shape(geom)
    return None


def crown_table(labels, profile, chm=None, height_percentile=95, min_area_m2=1.5):
    """Build one row per crown, with geometry in map coordinates."""
    px_area = pixel_area_m2(profile)
    transform = profile["transform"]
    rows, geometries = [], []

    for label_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        sub = labels[slices] == label_id
        area_m2 = int(sub.sum()) * px_area
        if area_m2 < min_area_m2:
            continue

        polygon = crown_polygon(sub, slices[0].start, slices[1].start, transform)
        if polygon is None:
            continue

        # equivalent circular diameter of the crown projection
        diameter_m = 2.0 * np.sqrt(area_m2 / np.pi)
        centre = ndimage.center_of_mass(sub)
        x, y = transform_xy(transform,
                            slices[0].start + centre[0],
                            slices[1].start + centre[1])

        record = {
            "crown_id": label_id,
            "area_m2": round(area_m2, 3),
            "crown_diameter_m": round(diameter_m, 3),
            "centroid_x": round(float(x), 3),
            "centroid_y": round(float(y), 3),
        }
        if chm is not None:
            heights = chm[slices][sub]
            heights = heights[np.isfinite(heights) & (heights > 0)]
            record["height_m"] = (round(float(np.percentile(heights, height_percentile)), 3)
                                  if heights.size else np.nan)

        rows.append(record)
        geometries.append(polygon)

    return pd.DataFrame(rows), geometries


def main():
    cfg = load_config()
    crowns_cfg = cfg["crowns"]
    out_dir = project_path(cfg["paths"]["output_dir"])

    banner("Step 3  crown delineation")

    summary = []
    for field in cfg["paths"]["field_orthomosaics"]:
        reference = project_path(cfg["paths"]["field_masks"].get(field, "") or "missing")
        predicted = os.path.join(out_dir, f"{field}_canopy_mask.tif")
        mask_path = reference if os.path.exists(reference) else predicted
        if not os.path.exists(mask_path):
            print(f"  {field}: no canopy mask found, skipped")
            continue

        mask, profile = read_raster(mask_path, band=1)
        mask = (mask == 1)

        chm = None
        chm_rel = cfg["paths"]["field_chm"].get(field) or ""
        if chm_rel and os.path.exists(project_path(chm_rel)):
            chm, _ = read_raster(project_path(chm_rel), band=1)
            chm = chm.astype(np.float32)

        labels = delineate(mask, crowns_cfg["peak_min_distance_px"])
        table, geometries = crown_table(labels, profile, chm,
                                        crowns_cfg["height_percentile"],
                                        crowns_cfg["min_crown_area_m2"])
        if table.empty:
            print(f"  {field}: no crowns above the minimum area, skipped")
            continue

        table.to_csv(os.path.join(out_dir, f"crowns_{field}.csv"), index=False)
        gpd.GeoDataFrame(table, geometry=geometries, crs=profile["crs"]).to_file(
            os.path.join(out_dir, f"crowns_{field}.gpkg"), driver="GPKG")

        canopy_ha = float(table["area_m2"].sum()) / 10000.0
        record = {
            "field": field,
            "source_mask": os.path.basename(mask_path),
            "n_crowns": int(len(table)),
            "canopy_ha": round(canopy_ha, 4),
            "mean_crown_area_m2": round(float(table["area_m2"].mean()), 3),
            "mean_crown_diameter_m": round(float(table["crown_diameter_m"].mean()), 3),
        }
        if "height_m" in table.columns:
            record["mean_height_m"] = round(float(table["height_m"].mean()), 3)
            record["max_height_m"] = round(float(table["height_m"].max()), 3)
        summary.append(record)

        line = (f"  {field}: {len(table):>5} crowns  canopy {canopy_ha:7.3f} ha"
                f"  mean CD {record['mean_crown_diameter_m']:.2f} m")
        if "mean_height_m" in record:
            line += f"  mean H {record['mean_height_m']:.2f} m"
        print(line)

    if summary:
        path = os.path.join(out_dir, "crown_summary.csv")
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"\n  summary -> {path}")


if __name__ == "__main__":
    main()
