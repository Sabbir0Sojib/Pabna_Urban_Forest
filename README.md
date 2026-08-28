# Urban Tree Characterization with Deep Learning - Pabna Municipality, Bangladesh

Code for the study *"Characterization of Urban Trees with Deep Learning Techniques in Pabna
Municipality of Bangladesh"*.

The pipeline maps urban tree canopy across a whole municipality by using high-resolution UAV
imagery as the label generator for a coarser satellite classifier, then converts the mapped
canopy into above-ground biomass and carbon estimates.

```
UAV orthomosaic (5 cm)  ->  deep learning canopy segmentation  ->  crown polygons
         |                                                              |
         |                                                    allometric biomass
         v                                                              |
  training labels for                                                   v
  PlanetScope (3 m)  ->  Random Forest canopy map  ->  city and ward statistics
```

## Repository layout

```
pabna-urban-forest/
  config.yaml                     all paths, parameters and coefficients
  requirements.txt
  run_all.sh                      runs steps 0-6 in order
  data/README.md                  where to place the input data
  src/
    utils.py                      raster IO, ExG, metrics, Gini
    s00_build_chm.py              canopy height model, CHM = DSM - DEM
    s01_train_segmentation.py     train canopy segmentation on UAV tiles
    s02_predict_field_masks.py    tiled inference -> georeferenced canopy mask
    s03_extract_crowns.py         watershed crown delineation + crown metrics
    s04_train_canopy_classifier.py  Random Forest city-wide canopy map
    s05_biomass_carbon.py         crown biomass -> city totals, carbon, CO2e
    s06_ward_statistics.py        ward canopy/biomass, Gini, canopy deficit
```

## Equations

All five equations of the manuscript are implemented, with every coefficient held in
`config.yaml` rather than hard-coded.

| Eq. | Script | Formula | Source |
| --- | --- | --- | --- |
| 1 | `s00_build_chm.py` | `CHM = DSM - DEM` | - |
| 2 | `s05_biomass_carbon.py` | `D = 0.557 (H x CD)^0.809 exp(0.0562/2)` | Jucker et al. (2017) |
| 3 | `s05_biomass_carbon.py` | `AGB = 0.0673 (rho D^2 H)^0.976` | Chave et al. (2014) |
| 4 | `s05_biomass_carbon.py` | `C = AGB x 0.47` | IPCC (2006) |
| 5 | `s05_biomass_carbon.py` | `CO2e = C x 3.67` | IPCC (2006) |

Units: crown height `H` and crown diameter `CD` in metres, predicted stem diameter `D` in
centimetres, wood density `rho` in g cm-3 (0.55, Chave et al. 2009), `AGB` in kilograms. The
exponential term in Equation 2 is the back-transformation correction for a model fitted in log
space.

Per-crown biomass is aggregated to a density per hectare of canopy within each of the three
UAV fields. The mean of those three densities is applied to the city-wide canopy area, and
their spread is carried into the reported uncertainty.

## Install

```bash
git clone <this-repository>
cd pabna-urban-forest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is needed for step 1 only. Steps 0 and 2-6 run on CPU.

## Data

Raw UAV photographs for the three survey fields are on Zenodo:

```
UAV Imagery - https://doi.org/10.5281/zenodo.22141709
```

Build the orthomosaic and DSM for each field yourself with photogrammetry software such as
Agisoft Metashape, Pix4Dmapper, or OpenDroneMap, then place the exports in `data/`. Training
annotations are not released; use the published segmentation weights instead:

```
https://huggingface.co/Sabbir12345/P6_S4_SegFormer_RGBExG_100ep
```

Full details on what is and isn't included are in `data/README.md`.

## Running

Edit the paths in `config.yaml` first, then either run everything:

```bash
bash run_all.sh
```

or run the steps one at a time:

```bash
python src/s00_build_chm.py
python src/s01_train_segmentation.py
python src/s02_predict_field_masks.py
python src/s03_extract_crowns.py
python src/s04_train_canopy_classifier.py
python src/s05_biomass_carbon.py
python src/s06_ward_statistics.py
```

Each step writes to `outputs/` and prints the numbers it produced.

## Canopy height, and what runs without it

Equations 2 and 3 both need crown height, so steps 0 and 5 need two extra inputs:

- the **UAV digital surface model**, exported from the photogrammetry project that produced
  the orthomosaics, and
- the **ALOS PALSAR 12.5 m DEM**, downloaded free from the Alaska Satellite Facility.

Step 0 subtracts the second from the first to build the canopy height model, and step 3 reads
crown heights from it. If either file is missing, step 0 exits with a message instead of
failing, and steps 2, 3, 4 and 6 still produce the full canopy result: segmentation accuracy,
crown polygons, the city-wide canopy map, canopy cover per ward, the Gini coefficient and the
canopy deficit. Only the biomass and carbon numbers require height.

Because the deposit holds the raw photographs rather than a finished DSM, the biomass and
carbon rows below can still be rebuilt end to end, but only after re-running photogrammetry
yourself. There is no reference mask left to check segmentation accuracy against, so treat any
rerun's F1/IoU as a new result to interpret on its own, not a check against Table 2.

## Reported results

These are the values the pipeline produced for the study area, for reference when checking a
re-run.

| Quantity | Value |
| --- | --- |
| Study area | 1,081.33 ha (15 wards) |
| Best segmentation model | SegFormer, RGB + ExG input |
| Segmentation F1 / IoU | 0.944 / 0.895 |
| Satellite canopy F1 (threshold 0.50) | 0.8513 |
| Mapped tree canopy | 285.62 ha (26.41% of the study area) |
| Delineated crowns, three UAV fields | 2,046 |
| Mean / maximum crown height | 9.46 m / 35.51 m |
| Above-ground biomass | 4,416 Mg (3,166-5,665 Mg, +/-28.3%) |
| Carbon stock | 2,076 Mg |
| CO2 equivalent | 7,617 Mg |
| Biomass density | 4.08 Mg ha-1 of land, 15.46 Mg ha-1 of canopy |
| Canopy Gini coefficient | 0.182 |
| Canopy deficit, five wards below average | 43.1 ha |

## Citation

See `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
