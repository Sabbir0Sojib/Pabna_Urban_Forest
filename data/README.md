# Input data

All rasters are GeoTIFFs in EPSG:32645 (UTM zone 45N). Put them under `data/` with the names
below, or edit the paths in `config.yaml`. None of these files are in the repository.

| Path | Content | Used by |
| --- | --- | --- |
| `uav/Field{1,2,3}_DSM.tif` | UAV digital surface model, 5.3 cm (Agisoft Metashape 2.3.1) | s00 |
| `uav/Field{1,2,3}_DTM.tif` | UAV ground-only digital terrain model, 10 cm, same project | s00 |
| `uav/Field{1,2,3}_canopy_mask.tif` | Full-field SegFormer canopy mask from the leave-one-field-out models (1 tree, 0 non-tree, 255 nodata), output of `src/segmentation/pabna_canopy_maps.py` | s00, s01 |
| `uav/Field{1,2,3}_mask.tif` | UAV reference canopy mask on the orthomosaic grid (1 tree, 0 non-tree), used for the 3 m classifier tests | s01 |
| `reference/gt_crowns_1320.geojson` | 1,320 manually delineated reference crowns, property `field` = `Field1`/`Field2`/`Field3` | s03 |
| `planetscope/PlanetScope_SR_8b_2026-01-12.tif` | SuperDove 8-band surface reflectance, 3 m; defines the 3 m grid | s01-s05, s07 |
| `planetscope/PlanetScope_SR_8b_2026-01-12_composite.tif` | Full-coverage composite of the same date (fills 3.12 ha of Ward 6) | s05 |
| `planetscope/PlanetScope_SR_8b_2026-01-15_clip.tif`, `..._2026-01-22_clip.tif` | Scenes clipped around the UAV fields, for the 3-date tests | s02 |
| `planetscope/PlanetScope_SR_8b_2026-01-22.tif` | Full scene for the robustness check (`s05_city_map.py --check`) | s05 |
| `boundaries/ward_boundary_15wards.geojson` | 15 ward polygons digitized from the BBS Small Area Atlas, property `id` = 1-15 | s05 |
| `global_chm/Meta_WRI_v1_1m.tif`, `global_chm/CHMv2_1m.tif` | Global canopy height maps resampled to 1 m on a grid aligned with the 3 m PlanetScope grid (integer-metre offset) | s07 |

## Grids

- **3 m grid**: the grid of the 12 January 2026 scene. Other scenes are pasted into it at their
  pixel offset.
- **1 m grid**: the 3 m grid subdivided 3 x 3 (same origin).
- **0.25 m grid**: per field, over the extent of the SegFormer canopy mask (CHM and crowns).

## Availability

- UAV orthomosaics: Hugging Face, https://doi.org/10.57967/hf/10147
- DSM, DTM, canopy masks, reference masks, reference crowns and ward boundaries: from the
  corresponding author on reasonable request
- PlanetScope: Planet Labs PBC (Education and Research Program); not redistributable
- Meta/WRI v1 canopy height (Tolan et al., 2024) and CHMv2 (Brandt et al., 2026): public; resample
  them to the 1 m grid with GDAL, e.g.
  `gdalwarp -t_srs EPSG:32645 -tr 1 1 -tap -r bilinear input.tif data/global_chm/Meta_WRI_v1_1m.tif`
