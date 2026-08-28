"""Step 5. Convert crowns into biomass, carbon and CO2 equivalent.

Input  : outputs/crowns_<Field>.csv and the city-wide tree mask
Output : outputs/biomass_per_field.csv and outputs/biomass_citywide.json

Equations, matching the manuscript:

  (2)  D_cm  = 0.557 * (H_m * CD_m) ** 0.809 * exp(0.0562 / 2)
                                              Jucker et al. (2017)
  (3)  AGB_kg = 0.0673 * (rho * D_cm ** 2 * H_m) ** 0.976
                                              Chave et al. (2014)
  (4)  C      = AGB * 0.47                    IPCC (2006)
  (5)  CO2e   = C * 3.67                      IPCC (2006)

with rho = 0.55 g cm-3 from Chave et al. (2009). All coefficients live in
config.yaml so nothing is hard-coded here.

Scaling, matching Section 2.5:
  1. apply Equations 2 and 3 to every delineated crown
  2. sum crown biomass per field and divide by that field's canopy area, which
     gives a biomass density per hectare of canopy
  3. average those three field densities, and keep their spread
  4. multiply the mean density by the city-wide canopy area from step 4
  5. convert biomass to carbon and to CO2 equivalent

Step 2 normalises by canopy area rather than by ground area, so the density
transfers to the rest of the city independently of how dense the canopy is.
"""

import json
import os

import numpy as np
import pandas as pd

from utils import banner, load_config, pixel_area_m2, project_path, read_raster

NO_HEIGHT_MESSAGE = """\
The crown table has no usable height column, so Equations 2 and 3 cannot be
applied: both need crown height H.

Run step 0 to build the canopy height model. It needs
  - paths.field_dsm    the UAV digital surface model, exported from the
                       photogrammetry project
  - paths.terrain_dem  the ALOS PALSAR 12.5 m DEM, free from the Alaska
                       Satellite Facility

Steps 1 to 4 and step 6 do not need height, so the canopy area results are
unaffected."""


def stem_diameter_cm(height_m, crown_diameter_m, jucker):
    """Equation 2. Predicted stem diameter in centimetres."""
    correction = np.exp(jucker["sigma_squared"] / 2.0)
    return jucker["a"] * np.power(height_m * crown_diameter_m,
                                  jucker["b"]) * correction


def agb_kg(diameter_cm, height_m, rho, chave):
    """Equation 3. Above-ground biomass in kilograms."""
    return chave["a"] * np.power(rho * np.square(diameter_cm) * height_m,
                                 chave["b"])


def crown_biomass(table, allometry):
    """Apply Equations 2 and 3 to a crown table. Returns (diameter, agb)."""
    if "height_m" not in table.columns:
        raise SystemExit(NO_HEIGHT_MESSAGE)

    crown_diameter = table["crown_diameter_m"].to_numpy(dtype=float)
    height = table["height_m"].to_numpy(dtype=float)

    usable = np.isfinite(height) & (height > 0) & (crown_diameter > 0)
    if not usable.any():
        raise SystemExit(NO_HEIGHT_MESSAGE)

    diameter = np.full(height.shape, np.nan)
    biomass = np.full(height.shape, np.nan)

    diameter[usable] = stem_diameter_cm(
        height[usable], crown_diameter[usable], allometry["jucker"])
    biomass[usable] = agb_kg(
        diameter[usable], height[usable],
        allometry["wood_density_g_cm3"], allometry["chave"])

    return diameter, biomass, int((~usable).sum())


def main():
    cfg = load_config()
    allometry = cfg["allometry"]
    carbon_cfg = cfg["carbon"]
    out_dir = project_path(cfg["paths"]["output_dir"])

    banner("Step 5  biomass and carbon")

    rho = allometry["wood_density_g_cm3"]
    chave_b = allometry["chave"]["b"]
    print(f"  Equation 2  D = {allometry['jucker']['a']} (H CD)^"
          f"{allometry['jucker']['b']} exp({allometry['jucker']['sigma_squared']}/2)")
    print(f"  Equation 3  AGB = {allometry['chave']['a']} (rho D^2 H)^{chave_b}")
    print(f"  wood density rho = {rho} g cm-3")

    # ---- per-field biomass density
    rows = []
    for field in cfg["paths"]["field_orthomosaics"]:
        csv_path = os.path.join(out_dir, f"crowns_{field}.csv")
        if not os.path.exists(csv_path):
            print(f"  {field}: crown table missing, skipped")
            continue

        table = pd.read_csv(csv_path)
        if table.empty:
            print(f"  {field}: crown table is empty, skipped")
            continue

        diameter, biomass, dropped = crown_biomass(table, allometry)
        table["stem_diameter_cm"] = np.round(diameter, 3)
        table["agb_kg"] = np.round(biomass, 3)
        table.to_csv(csv_path, index=False)          # keep per-crown biomass

        canopy_ha = float(table["area_m2"].sum()) / 10000.0
        total_Mg = float(np.nansum(biomass)) / 1000.0
        rows.append({
            "field": field,
            "n_crowns": int(len(table)),
            "n_without_height": dropped,
            "canopy_ha": round(canopy_ha, 4),
            "mean_height_m": round(float(np.nanmean(table["height_m"])), 3),
            "mean_stem_diameter_cm": round(float(np.nanmean(diameter)), 3),
            "agb_Mg": round(total_Mg, 3),
            "agb_Mg_per_canopy_ha": round(total_Mg / canopy_ha, 4) if canopy_ha else 0.0,
            "mean_crown_agb_kg": round(float(np.nanmean(biomass)), 2),
        })
        note = f"  ({dropped} without height)" if dropped else ""
        print(f"  {field}: {len(table):>5} crowns  {total_Mg:8.2f} Mg"
              f"  {rows[-1]['agb_Mg_per_canopy_ha']:7.2f} Mg per canopy ha{note}")

    if not rows:
        raise SystemExit("No crown tables found. Run step 3 first.")

    per_field = pd.DataFrame(rows)
    per_field.to_csv(os.path.join(out_dir, "biomass_per_field.csv"), index=False)

    densities = per_field["agb_Mg_per_canopy_ha"].to_numpy(dtype=float)
    density_mean = float(densities.mean())
    density_sd = float(densities.std(ddof=1)) if densities.size > 1 else 0.0

    # ---- city-wide canopy area
    produced = os.path.join(out_dir, "citywide_tree_mask.tif")
    supplied = project_path(cfg["paths"]["citywide_tree_mask"])
    mask_path = produced if os.path.exists(produced) else supplied
    if not os.path.exists(mask_path):
        raise SystemExit("No city-wide tree mask found. Run step 4, or place the "
                         "mask at paths.citywide_tree_mask.")

    mask, profile = read_raster(mask_path, band=1)
    canopy_ha_city = int((mask == 1).sum()) * pixel_area_m2(profile) / 10000.0

    # ---- scale up, then convert to carbon (Equations 4 and 5)
    total_agb = canopy_ha_city * density_mean
    total_carbon = total_agb * carbon_cfg["carbon_fraction"]
    total_co2e = total_carbon * carbon_cfg["co2_per_carbon"]

    # ---- uncertainty, the three components described in Section 2.5
    # Equation 3 is not linear in rho: AGB scales as rho ** chave.b
    rho_low, rho_high = allometry["wood_density_range"]
    agb_low_rho = total_agb * (rho_low / rho) ** chave_b
    agb_high_rho = total_agb * (rho_high / rho) ** chave_b
    rho_relative = max(abs(agb_high_rho - total_agb),
                       abs(total_agb - agb_low_rho)) / total_agb
    field_relative = density_sd / density_mean if density_mean else 0.0
    combined = float(np.hypot(rho_relative, field_relative))

    total_area_ha = cfg["study_area"]["total_area_ha"]
    result = {
        "mask_used": os.path.basename(mask_path),
        "canopy_ha_citywide": round(canopy_ha_city, 3),
        "canopy_percent": round(100.0 * canopy_ha_city / total_area_ha, 3),
        "biomass_density_Mg_per_canopy_ha": round(density_mean, 4),
        "biomass_density_sd": round(density_sd, 4),
        "total_agb_Mg": round(total_agb, 1),
        "total_carbon_Mg": round(total_carbon, 1),
        "total_co2e_Mg": round(total_co2e, 1),
        "agb_Mg_per_study_area_ha": round(total_agb / total_area_ha, 3),
        "agb_Mg_per_canopy_ha": round(density_mean, 3),
        "uncertainty": {
            "wood_density_relative": round(rho_relative, 4),
            "between_field_relative": round(field_relative, 4),
            "combined_relative": round(combined, 4),
            "agb_range_Mg": [round(total_agb * (1 - combined), 1),
                             round(total_agb * (1 + combined), 1)],
            "agb_range_wood_density_only_Mg": [round(agb_low_rho, 1),
                                               round(agb_high_rho, 1)],
        },
    }
    with open(os.path.join(out_dir, "biomass_citywide.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    low, high = result["uncertainty"]["agb_range_Mg"]
    print(f"\n  biomass density   : {density_mean:.2f} Mg per canopy ha "
          f"(SD {density_sd:.2f}, n={densities.size})")
    print(f"  city canopy area  : {canopy_ha_city:.2f} ha "
          f"({result['canopy_percent']:.2f}%)")
    print(f"  total AGB         : {total_agb:,.1f} Mg")
    print(f"  total carbon      : {total_carbon:,.1f} Mg")
    print(f"  total CO2e        : {total_co2e:,.1f} Mg")
    print(f"  per study-area ha : {result['agb_Mg_per_study_area_ha']:.2f} Mg")
    print(f"  uncertainty       : +/-{100 * combined:.1f}% "
          f"({low:,.0f} to {high:,.0f} Mg)")
    print(f"                      wood density {100 * rho_relative:.1f}%, "
          f"between-field {100 * field_relative:.1f}%")


if __name__ == "__main__":
    main()
