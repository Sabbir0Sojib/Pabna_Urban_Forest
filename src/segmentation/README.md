# SegFormer canopy segmentation (UAV, 2.6 cm)

Code used for the UAV canopy results of the paper. It was run on Kaggle (2 x T4 GPUs), one process
per run, launched by `pabna_train_driver.ipynb`. Inputs are read from `/kaggle/input` (set
`PABNA_INPUT` to change this).

| File | Phase | What it does |
| --- | --- | --- |
| `pabna_phase0_phase1.ipynb`, `pabna_phase01.py` | 0-1 | Data audit; leakage-free splits from the raw Roboflow tiles (field read from the tile file name). S1 = random split of whole tiles; S2 = leave-one-field-out (train two fields, test the third), with buffer tiles between validation and training |
| `pabna_train_driver.ipynb`, `pabna_train.py` | 2-4 | One run = one input (RGB or RGB+ExG), one split, one seed. SegFormer MiT-B2 (`nvidia/mit-b2`), 512 px, batch 6, AdamW lr 6e-5 (head x10), weight decay 0.01, 69,600 steps. Checkpoint, threshold, minimum segment area and TTA are chosen on validation tiles only and frozen in `tuning.json` before the test tiles are read; the test set is predicted once |
| `expected_runs.json` | | Frozen tuning (threshold, minimum area, TTA, checkpoint hash) and test F1 of the 9 RGB leave-one-field-out runs |
| `pabna_verify_runs.py` | | Checks finished runs (disjoint splits, tuning written before testing) |
| `pabna_aggregate.py` | 5-6 | Pooled metrics, 95 % block-bootstrap intervals (3 x 3 tile blocks, about 50 m), paired RGB vs RGB+ExG comparison, final tables |
| `pabna_error_analysis.py` | | Hit rate by crown size, per-field precision/recall, examples |
| `pabna_canopy_maps.py` | | Full-field canopy masks: field k is predicted only by the 3 fold-k models that never saw it (seed-mean probability, mean tuned threshold and minimum area, TTA). Writes `Field{k}_canopy_prob.tif` and `Field{k}_canopy_mask.tif` (1 tree, 0 non-tree, 255 nodata) used by the analysis steps |
| `check_a_decomposition.py` | | Cross-check of the stitched masks against the frozen test results |

Results (independent field test, RGB, mean of 3 seeds per field): F1 0.878 +/- 0.006
(0.883 / 0.871 / 0.880), IoU 0.782 +/- 0.010. Whole-tile random split (S1): 0.910. Original
random split with tile copies in training and test: 0.944. Adding the excess-green band:
dF1 = +0.0011 (95 % CI -0.0007 to +0.0030), i.e. no measurable gain.
