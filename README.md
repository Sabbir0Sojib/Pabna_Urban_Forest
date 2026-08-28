# Urban Tree Characterization with Deep Learning: Pabna Municipality, Bangladesh

Code for the study "Characterization of Urban Trees with Deep Learning Techniques in Pabna
Municipality of Bangladesh."

UAV imagery trains a canopy segmentation model. That model's output trains a satellite
classifier for city-wide canopy mapping. Canopy is then converted to biomass and carbon.

```
UAV orthomosaic -> segmentation -> crown polygons -> biomass/carbon
                                          |
                          training labels for satellite classifier -> ward statistics
```

## Layout

```
config.yaml                     paths, parameters, coefficients
run_all.sh                      runs steps 0-6 in order
data/README.md                  input data and where to place it
src/
  utils.py                      raster IO, ExG, metrics, Gini
  s00_build_chm.py              CHM = DSM - DEM
  s01_train_segmentation.py     train canopy segmentation on UAV tiles
  s02_predict_field_masks.py    tiled inference -> canopy mask
  s03_extract_crowns.py         watershed crown delineation + metrics
  s04_train_canopy_classifier.py  Random Forest city-wide canopy map
  s05_biomass_carbon.py         crown biomass -> totals, carbon, CO2e
  s06_ward_statistics.py        ward canopy/biomass, Gini, canopy deficit
```

## Equations

| Eq. | Script | Formula | Source |
| --- | --- | --- | --- |
| 1 | s00_build_chm.py | CHM = DSM - DEM | - |
| 2 | s05_biomass_carbon.py | D = 0.557 (H x CD)^0.809 exp(0.0562/2) | Jucker et al. (2017) |
| 3 | s05_biomass_carbon.py | AGB = 0.0673 (rho D^2 H)^0.976 | Chave et al. (2014) |
| 4 | s05_biomass_carbon.py | C = AGB x 0.47 | IPCC (2006) |
| 5 | s05_biomass_carbon.py | CO2e = C x 3.67 | IPCC (2006) |

H and CD in metres, D in centimetres, rho in g/cm3 (0.55, Chave et al. 2009), AGB in kg.

## Install

```bash
git clone <this-repository>
cd pabna-urban-forest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU needed for step 1 only.

## Data

Orthomosaics for the three UAV survey fields are on Zenodo:

```
https://doi.org/10.5281/zenodo.22141709
```

Place them in `data/` as described in `data/README.md`. Everything else (DSM, satellite
imagery, code, derived results) is available from the corresponding author on request.

Published segmentation weights, in place of training annotations:

```
https://huggingface.co/Sabbir12345/P6_S4_SegFormer_RGBExG_100ep
```

## Running

Edit `config.yaml`, then:

```bash
bash run_all.sh
```

or step by step:

```bash
python src/s00_build_chm.py
python src/s01_train_segmentation.py
python src/s02_predict_field_masks.py
python src/s03_extract_crowns.py
python src/s04_train_canopy_classifier.py
python src/s05_biomass_carbon.py
python src/s06_ward_statistics.py
```

Each step writes to `outputs/`.

## Reported results

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
| Biomass density | 4.08 Mg/ha of land, 15.46 Mg/ha of canopy |
| Canopy Gini coefficient | 0.182 |
| Canopy deficit, five wards below average | 43.1 ha |

## Citation

See `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
