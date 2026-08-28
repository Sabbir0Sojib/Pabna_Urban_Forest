#!/usr/bin/env bash
# Run the full pipeline in order. Stops at the first failing step.
#
# Steps 0 and 5 need canopy height, which comes from the UAV DSM and the ALOS
# PALSAR DEM. Without those two files, comment out both steps: steps 1-4 and 6
# still produce every canopy result.
set -euo pipefail

echo "=== Step 0/6  canopy height model (Equation 1) ==="
python src/s00_build_chm.py

echo "=== Step 1/6  train canopy segmentation ==="
python src/s01_train_segmentation.py

echo "=== Step 2/6  predict field canopy masks ==="
python src/s02_predict_field_masks.py

echo "=== Step 3/6  delineate crowns ==="
python src/s03_extract_crowns.py

echo "=== Step 4/6  train city-wide canopy classifier ==="
python src/s04_train_canopy_classifier.py

echo "=== Step 5/6  biomass and carbon (Equations 2-5) ==="
python src/s05_biomass_carbon.py

echo "=== Step 6/6  ward statistics ==="
python src/s06_ward_statistics.py

echo "=== Done. Results are in outputs/ ==="
