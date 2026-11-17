#!/usr/bin/env bash
# Analysis steps 0-7 in order. Stops at the first failing step.
# The SegFormer canopy masks (data/uav/Field*_canopy_mask.tif) come from src/segmentation/ first.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 0/7  canopy height model (UAV DSM - UAV DTM) ==="
python src/s00_build_chm.py

echo "=== Step 1/7  reference canopy fractions on the satellite grids ==="
python src/s01_reference_fractions.py

echo "=== Step 2/7  satellite methods, independent field test ==="
python src/s02_satellite_tests.py

echo "=== Step 3/7  crown delineation and validation ==="
python src/s03_extract_crowns.py

echo "=== Step 4/7  sub-pixel model, independent field test ==="
python src/s04_subpixel_lofo.py

echo "=== Step 5/7  city-wide canopy map and ward table ==="
python src/s05_city_map.py

echo "=== Step 6/7  biomass and carbon (Monte Carlo) ==="
python src/s06_biomass_carbon.py

echo "=== Step 7/7  global canopy height maps ==="
python src/s07_global_chm.py

echo "=== Done. Results are in outputs/ ==="
