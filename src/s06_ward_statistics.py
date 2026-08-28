"""Step 6. Break the canopy and biomass totals down by ward.

Input  : the city-wide tree mask, the ward boundaries, and the biomass density
         written by step 5
Output : outputs/ward_statistics.csv and outputs/ward_summary.json

Also reports the Gini coefficient of canopy area across wards, and the canopy
deficit: the additional canopy each below-average ward would need to reach the
city-wide canopy percentage.
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask

from utils import banner, gini, load_config, project_path


def ward_name(row, index):
    for key in ("ward", "Ward", "ward_no", "WARD_NO", "name", "NAME", "ward_name"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return f"Ward {index + 1}"


def main():
    cfg = load_config()
    out_dir = project_path(cfg["paths"]["output_dir"])

    banner("Step 6  ward-level statistics")

    produced = os.path.join(out_dir, "citywide_tree_mask.tif")
    supplied = project_path(cfg["paths"]["citywide_tree_mask"])
    mask_path = produced if os.path.exists(produced) else supplied
    if not os.path.exists(mask_path):
        raise SystemExit("No city-wide tree mask found. Run step 4 first.")

    wards_path = project_path(cfg["paths"]["ward_boundaries"])
    if not os.path.exists(wards_path):
        raise SystemExit(f"Ward boundaries not found: {wards_path}")

    density = None
    biomass_json = os.path.join(out_dir, "biomass_citywide.json")
    if os.path.exists(biomass_json):
        with open(biomass_json, "r", encoding="utf-8") as fh:
            density = json.load(fh)["biomass_density_Mg_per_canopy_ha"]

    rows = []
    with rasterio.open(mask_path) as src:
        wards = gpd.read_file(wards_path).to_crs(src.crs)
        px_area = abs(src.transform.a * src.transform.e)

        for i, ward in wards.iterrows():
            clipped, _ = rio_mask(src, [ward.geometry], crop=True, filled=True, nodata=0)
            canopy_px = int((clipped[0] == 1).sum())
            canopy_ha = canopy_px * px_area / 10000.0
            ward_ha = ward.geometry.area / 10000.0

            record = {
                "ward": ward_name(ward, i),
                "ward_area_ha": round(ward_ha, 3),
                "canopy_ha": round(canopy_ha, 3),
                "canopy_percent": round(100.0 * canopy_ha / ward_ha, 3) if ward_ha else 0.0,
            }
            if density is not None:
                record["agb_Mg"] = round(canopy_ha * density, 2)
                record["carbon_Mg"] = round(record["agb_Mg"] * cfg["carbon"]["carbon_fraction"], 2)
            rows.append(record)

    table = pd.DataFrame(rows).sort_values("canopy_ha", ascending=False)
    table.to_csv(os.path.join(out_dir, "ward_statistics.csv"), index=False)

    canopy_total = float(table["canopy_ha"].sum())
    area_total = float(table["ward_area_ha"].sum())
    city_percent = 100.0 * canopy_total / area_total if area_total else 0.0

    # canopy needed to bring every below-average ward up to the city percentage
    deficit = float(sum(
        max(0.0, ward["ward_area_ha"] * city_percent / 100.0 - ward["canopy_ha"])
        for _, ward in table.iterrows()))

    top5 = table.head(5)
    summary = {
        "n_wards": int(len(table)),
        "canopy_ha_total": round(canopy_total, 3),
        "canopy_percent_city": round(city_percent, 3),
        "canopy_gini": round(gini(table["canopy_ha"].to_numpy()), 4),
        "canopy_percent_min": round(float(table["canopy_percent"].min()), 3),
        "canopy_percent_max": round(float(table["canopy_percent"].max()), 3),
        "share_in_top5_wards_percent": round(100.0 * float(top5["canopy_ha"].sum()) / canopy_total, 2)
        if canopy_total else 0.0,
        "canopy_deficit_ha": round(deficit, 3),
    }
    if density is not None:
        summary["agb_Mg_total"] = round(float(table["agb_Mg"].sum()), 1)

    with open(os.path.join(out_dir, "ward_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(table.to_string(index=False))
    print(f"\n  city canopy      : {canopy_total:.2f} ha ({city_percent:.2f}%)")
    print(f"  Gini coefficient : {summary['canopy_gini']:.3f}")
    print(f"  top five wards   : {summary['share_in_top5_wards_percent']:.1f}% of all canopy")
    print(f"  canopy deficit   : {deficit:.1f} ha to bring every ward to the city average")


if __name__ == "__main__":
    main()
