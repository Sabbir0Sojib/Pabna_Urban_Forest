# Urban tree canopy, biomass and carbon from UAV and PlanetScope imagery: Pabna, Bangladesh

Code for the article *"Characterization of Urban Trees with Deep Learning Techniques in Pabna
Municipality of Bangladesh"* (Md Sabbir Islam and Md Rahedul Islam), revised version
(International Journal of Applied Earth Observation and Geoinformation, JAG-D-26-04430).

A small UAV survey (three fields, 33.4 ha) replaces ground-based tree inventory:

```
UAV RGB (2.6 cm) ──► SegFormer canopy (leave-one-field-out) ──► 1 m / 3 m canopy fraction ──► satellite models (3 m PlanetScope)
UAV DSM − UAV DTM ──► canopy height ──► watershed crowns ──► allometry + Monte Carlo ──► biomass and carbon
```

Every model is evaluated in an **independent field test**: it is trained on two fields and tested
once on the third (fields at least 1.1 km apart). All thresholds, shifts and settings are chosen on
the training fields only.

## Layout

```
config.yaml                    paths and every setting used in the paper
run_all.sh                     steps 0-7 in order
data/README.md                 input files and where to put them
src/
  segmentation/                SegFormer MiT-B2 training, tuning, testing and full-field canopy maps
                               (run on Kaggle GPUs; see src/segmentation/README.md)
  utils.py                     raster IO, grids, polygons, shared helpers
  _tiff_fallback.py            GeoTIFF reader used only when rasterio is not installed
  s00_build_chm.py             CHM = UAV DSM − UAV ground-only DTM (Equation 1), 0.25 m grid
  s01_reference_fractions.py   UAV canopy as 3 m and 1 m fractions on the PlanetScope grid
  s02_satellite_tests.py       NDVI/NDRE thresholds and five classifiers, 1 and 3 dates, 50 % and 20 % rules,
                               threshold sweep (Figure 4)
  s03_extract_crowns.py        marker-controlled watershed crowns, parameter scan and validation against
                               1,320 reference crowns
  s04_subpixel_lofo.py         coarse-to-fine sub-pixel model (1 m canopy fraction from 5 x 5 PlanetScope pixels)
  s05_city_map.py              indicative city-wide map, bias correction, ward table (Table 4, Figure 5)
  s06_biomass_carbon.py        Monte Carlo biomass and carbon (Equations 2-5, Table 5)
  s07_global_chm.py            Meta/WRI v1 and CHMv2 canopy height maps in the same test
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The analysis steps (0-7) run on a CPU. Segmentation training needs a GPU (we used Kaggle T4 x 2).

## Run

Put the input files in `data/` as described in `data/README.md`, check `config.yaml`, then:

```bash
bash run_all.sh
```

or run the steps one by one (`python src/s00_build_chm.py`, ...). Results go to `outputs/`.
Steps 2 and 4 take the longest (several hours on a laptop for step 2).

## Equations

| Eq. | Step | Formula | Source |
| --- | --- | --- | --- |
| 1 | s00 | CHM = DSM − DTM (both from the UAV survey) | |
| 2 | s06 | D = 0.557 (H × CD)^0.809 exp(0.056² / 2) | Jucker et al. (2017) |
| 3 | s06 | AGB = 0.0673 (ρ D² H)^0.976 | Chave et al. (2014) |
| 4 | s06 | C = 0.47 × AGB | IPCC (2006) |
| 5 | s06 | CO₂e = 3.67 × C | IPCC (2006) |

H = crown height (95th percentile of the CHM in the crown, m), CD = equal-area crown diameter (m),
D = stem diameter (cm), ρ = wood density (g cm⁻³; mean 0.58, Reyes et al., 1992), AGB in kg.

## Main results reproduced by the code

| Result | Value | Step |
| --- | --- | --- |
| SegFormer canopy F1, independent field test (RGB) | 0.878 ± 0.006 | segmentation |
| Same model, original random split / whole-tile split | 0.944 / 0.910 | segmentation |
| Excess-green band, change in F1 | +0.0011 (95 % CI −0.0007 to +0.0030) | segmentation |
| Reference crowns matched at IoU ≥ 0.5 | 57.3 % (64.4 / 48.8 / 52.8 %) | s03 |
| Crowns delineated | 2,043 (925 / 491 / 627) | s03 |
| NDVI / NDRE thresholds, F1 | 0.31 / 0.29 | s02 |
| Best 3 m classifier (3-date random forest), F1 | 0.39 | s02 |
| Sub-pixel model, F1 at 1 m | 0.488 (0.481 / 0.453 / 0.531) | s04 |
| City canopy cover, predicted / bias-corrected | 22.3 % / 17.3–24.3 % (187–263 ha) | s05 |
| Field AGB (Mg), median (95 % interval) | 46.2 (33.8–62.5), 59.4 (44.6–78.3), 104.0 (78.3–133.5) | s06 |
| Study-area AGB, indicative | 5,160–12,720 Mg | s06 |

City-wide and study-area values are indicative, first-order estimates: no independent reference
exists outside the three UAV fields, the fields are not a random sample, and systematic errors
(crown segmentation, rooftop trees, photogrammetric crown-top underestimation, unmeasured stem
diameter and wood density) are not in the Monte Carlo intervals.

Small differences in the last digit can occur with other library versions.

## What changed from version 1 of this repository

Version 1 described the first submission. It is replaced because the revision changed the method:

- The 12.5 m ALOS PALSAR DEM is removed; canopy height uses the UAV ground-only terrain model.
- Random splits are replaced by the independent field test everywhere; the earlier accuracies
  (0.8513, 0.8205) are withdrawn.
- The city map now comes from a sub-pixel canopy-fraction model, reported as a bias-corrected range.
- Wood density is a distribution and uncertainty is propagated by Monte Carlo simulation.
- The excess-green claim, the Gini index and the ward canopy-deficit analysis are removed.
- The Jucker et al. (2017) σ is 0.056 (version 1 used 0.0562 by mistake).

## Data availability

UAV orthomosaics: Hugging Face, https://doi.org/10.57967/hf/10147. Other UAV products, masks and
reference crowns are available from the corresponding author on reasonable request. PlanetScope
imagery was obtained through the Planet Education and Research Program and cannot be redistributed.

## Citation

See `CITATION.cff`. License: MIT.
