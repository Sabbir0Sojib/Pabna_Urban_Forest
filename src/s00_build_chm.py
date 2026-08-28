"""Step 0. Build a canopy height model for each field.

Equation 1 of the manuscript:  CHM = DSM - DEM

Input  : paths.field_dsm    UAV digital surface model per field
         paths.terrain_dem  ALOS PALSAR 12.5 m terrain reference
Output : outputs/<Field>_chm.tif, on the DSM grid

The terrain DEM is far coarser than the UAV surface, so it is resampled onto
the DSM grid bilinearly before subtraction. Negative differences, which occur
where the coarse terrain sits above the fine surface, are clipped to zero.

This step is optional. If the DSM or the terrain DEM is missing the step exits
quietly, and the pipeline still produces every canopy result. Crown heights,
and therefore biomass in step 5, do require it.
"""

import os

import numpy as np
from rasterio.warp import Resampling, reproject

from utils import (banner, load_config, project_path, read_raster,
                   write_raster)


def resample_onto(source, source_profile, target_profile):
    """Resample `source` onto the grid described by `target_profile`."""
    destination = np.zeros(
        (target_profile["height"], target_profile["width"]), dtype=np.float32)
    reproject(
        source=np.asarray(source, dtype=np.float32),
        destination=destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        src_nodata=source_profile.get("nodata"),
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return destination


def build_chm(dsm, dsm_profile, dem, dem_profile, min_height, max_height):
    """Subtract the terrain surface from the UAV surface."""
    dsm = np.asarray(dsm, dtype=np.float32)
    nodata = dsm_profile.get("nodata")
    if nodata is not None:
        dsm = np.where(dsm == nodata, np.nan, dsm)

    terrain = resample_onto(dem, dem_profile, dsm_profile)
    chm = dsm - terrain

    chm = np.where(np.isfinite(chm), chm, 0.0)
    chm = np.clip(chm, min_height, max_height)
    return chm.astype(np.float32)


def main():
    cfg = load_config()
    paths = cfg["paths"]
    chm_cfg = cfg["chm"]
    out_dir = project_path(paths["output_dir"])

    banner("Step 0  canopy height model (Equation 1)")

    dem_path = project_path(paths["terrain_dem"])
    if not os.path.exists(dem_path):
        print(f"  terrain DEM not found at {paths['terrain_dem']}")
        print("  skipping. Steps 1-4 and 6 do not need it; step 5 does.")
        return

    dem, dem_profile = read_raster(dem_path, band=1)

    built = 0
    for field in paths["field_orthomosaics"]:
        dsm_rel = paths["field_dsm"].get(field) or ""
        dsm_path = project_path(dsm_rel) if dsm_rel else ""
        if not dsm_path or not os.path.exists(dsm_path):
            print(f"  {field}: no DSM at {dsm_rel or '(unset)'}, skipped")
            continue

        dsm, dsm_profile = read_raster(dsm_path, band=1)
        chm = build_chm(dsm, dsm_profile, dem, dem_profile,
                        chm_cfg["min_height_m"], chm_cfg["max_height_m"])

        destination = os.path.join(out_dir, f"{field}_chm.tif")
        write_raster(destination, chm, dsm_profile, dtype="float32")
        built += 1

        canopy = chm[chm > 1.0]
        print(f"  {field}: CHM written  mean {canopy.mean():5.2f} m"
              f"  max {chm.max():6.2f} m  -> {os.path.basename(destination)}"
              if canopy.size else
              f"  {field}: CHM written, no pixels above 1 m")

    if built == 0:
        print("\n  No CHM produced. Export the UAV DSM from your photogrammetry")
        print("  project and set paths.field_dsm in config.yaml.")
    else:
        print(f"\n  {built} canopy height model(s) in {out_dir}")


if __name__ == "__main__":
    main()
