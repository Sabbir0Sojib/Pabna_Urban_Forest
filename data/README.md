# Input data

Zenodo deposit: https://doi.org/10.5281/zenodo.22141709

Contains the RGB orthomosaic for each UAV survey field:

| File | Content |
| --- | --- |
| Field1_ortho.tif | Field 1 orthomosaic, EPSG:32645 |
| Field2_ortho.tif | Field 2 orthomosaic, EPSG:32645 |
| Field3_ortho.tif | Field 3 orthomosaic, EPSG:32645 |

Place them in `data/` under these names.

## Not in the deposit

- **DSMs** (`Field1_dsm.tif`, etc.), needed for steps 0, 3 and 5. Available from the
  corresponding author on request.
- **ALOS PALSAR DEM** (`alos_palsar_dem.tif`), needed for step 0. Free download from the
  Alaska Satellite Facility.
- **PlanetScope satellite imagery** (`planetscope_superdove.tif`), needed for step 4. Not
  redistributable; request equivalent imagery from Planet Labs PBC.
- **Training annotations** for step 1. Not released; use the published weights instead
  (see main README).

## What runs with the orthomosaics alone

Steps 1 and 2 (segmentation and inference) run directly on the orthomosaics using the
published weights. Steps 0, 3, 5 and 6 need the DSM. Step 4 needs the satellite image.

## CRS

All rasters: EPSG:32645 (UTM zone 45N). If your files differ, set `project.crs` in
`config.yaml` and reproject first.
