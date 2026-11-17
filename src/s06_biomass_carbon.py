"""Step 6. Above-ground biomass and carbon with Monte Carlo uncertainty (Equations 2 to 5).

Per crown (crowns from step 3):
  Eq. 2  D   = 0.557 (H x CD)^0.809 exp(e1),          e1 ~ N(0, 0.056)      Jucker et al. (2017)
  Eq. 3  AGB = 0.0673 (rho D^2 H)^0.976 exp(e2 - 0.357^2/2), e2 ~ N(0, 0.357)  Chave et al. (2014)
  Eq. 4  C   = 0.47 AGB;  Eq. 5  CO2e = 3.67 C                                 IPCC (2006)
Each of 1,000 runs adds random error to crown height (0.5 m per field + 1.0 m per crown), crown
diameter (15 %, log scale), wood density (field mean 0.58 +/- 0.05, crown SD 0.10, clipped 0.25-1.0)
and both allometric residuals. Field AGB = median and 95 % interval of the runs.

Indicative study-area estimate: pooled AGB per hectare of canopy (field bounds summed, so the
bounds are conservative) x the bias-corrected city canopy area from step 5. The three fields are not
a random sample, and systematic errors (crown segmentation, rooftop trees, photogrammetric crown-top
underestimation, unmeasured stem diameter and wood density) are NOT in these intervals.

Outputs: outputs/biomass_results.json, outputs/biomass_MC_draws_Mg.csv
"""
import json
import os

import numpy as np

from s03_extract_crowns import crown_table, load_chm
from utils import load_config, out, save_json


def main():
    cfg = load_config()
    al, mc, wd = cfg["allometry"], cfg["monte_carlo"], cfg["allometry"]["wood_density"]
    cc = cfg["crowns"]
    rng = np.random.default_rng(mc["seed"])
    res, draws, canopy_ha = {}, {}, {}
    for k in (1, 2, 3):
        chm, frac, X0, Y0, RES = load_chm(cfg, k)
        _, t = crown_table(cfg, k, chm, frac, X0, Y0, RES)
        h95, cd = t["h95"], t["cd"]
        can_ha = float(((frac >= cc["canopy_fraction_min"]) & (chm >= cc["min_height_m"])).sum() * RES * RES / 1e4)
        tot = np.zeros(mc["runs"])
        for i in range(mc["runs"]):
            H = np.clip(h95 + rng.normal(0, mc["height_sd_field_m"]) + rng.normal(0, mc["height_sd_crown_m"], len(h95)), 1.3, None)
            CD = cd * np.exp(rng.normal(0, mc["crown_diameter_log_sd"], len(cd)))
            D = al["jucker"]["a"] * (H * CD) ** al["jucker"]["b"] * np.exp(rng.normal(0, al["jucker"]["sigma"], len(H)))
            rho = np.clip(rng.normal(wd["mean"], wd["field_sd"]) + rng.normal(0, wd["crown_sd"], len(H)), wd["min"], wd["max"])
            agb = (al["chave"]["a"] * (rho * D ** 2 * H) ** al["chave"]["b"]
                   * np.exp(rng.normal(0, al["chave"]["sigma"], len(H))) * np.exp(-al["chave"]["sigma"] ** 2 / 2))
            tot[i] = agb.sum() / 1000
        q = np.percentile(tot, [2.5, 50, 97.5])
        draws[k] = tot
        canopy_ha[k] = can_ha
        res[k] = dict(crowns=int(len(h95)), canopy_ha=round(can_ha, 2),
                      AGB_Mg=[round(float(v), 1) for v in q],
                      AGB_Mg_per_ha_canopy=[round(float(v / can_ha), 1) for v in q],
                      carbon_Mg_C=[round(float(cfg["carbon"]["carbon_fraction"] * v), 1) for v in q])
        print(f"Field {k}:", res[k], flush=True)

    # three fields pooled (bounds = sums of field bounds: conservative)
    lo = sum(res[k]["AGB_Mg"][0] for k in (1, 2, 3))
    md = sum(res[k]["AGB_Mg"][1] for k in (1, 2, 3))
    hi = sum(res[k]["AGB_Mg"][2] for k in (1, 2, 3))
    ha = sum(round(v, 2) for v in canopy_ha.values())      # canopy areas as reported (2 decimals)
    dens = [lo / ha, md / ha, hi / ha]
    res["three_fields"] = dict(canopy_ha=round(ha, 2), AGB_Mg=[round(v, 1) for v in (lo, md, hi)],
                               AGB_Mg_per_ha_canopy=[round(v, 1) for v in dens],
                               carbon_Mg_C=[round(cfg["carbon"]["carbon_fraction"] * v, 1) for v in (lo, md, hi)])

    city = out(cfg, "city_canopy_corrected.json")
    if os.path.exists(city):
        cz = json.load(open(city))["city"]
        ha_lo, ha_md, ha_hi = cz["low_ha"], cz["middle_ha"], cz["high_ha"]
        agb = [ha_lo * dens[0], ha_md * dens[1], ha_hi * dens[2]]
        cf, co2 = cfg["carbon"]["carbon_fraction"], cfg["carbon"]["co2_per_carbon"]
        res["study_area_indicative"] = dict(canopy_ha=[ha_lo, ha_md, ha_hi], AGB_Mg=[round(v, -1) for v in agb],
                                            carbon_Mg_C=[round(cf * v, -1) for v in agb],
                                            CO2e_Mg=[round(cf * co2 * v, -1) for v in agb])
    else:
        print("Run step 5 first for the study-area estimate.")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(k, int)}, indent=1))
    save_json({str(k): v for k, v in res.items()}, out(cfg, "biomass_results.json"))
    np.savetxt(out(cfg, "biomass_MC_draws_Mg.csv"), np.c_[draws[1], draws[2], draws[3]], delimiter=",",
               header="field1_Mg,field2_Mg,field3_Mg", comments="", fmt="%.2f")


if __name__ == "__main__":
    main()
