# Landmark Baseline Fix Summary

Generated on 2026-04-27 after replacing the incomplete 300W folder and fixing
the landmark coordinate convention.

## What Changed

- Replaced the incomplete `datasets/datasets/300w` folder with the full 300W
  layout: AFW, LFPW, HELEN, and IBUG.
- Updated `W300Dataset` to use the standard 300W protocol:
  - train: AFW + LFPW train + HELEN train = 3148 images
  - test/full: LFPW test + HELEN test + IBUG = 689 images
- Fixed the landmark coordinate convention:
  - checkpoint predictions are centered coordinates in `[-1, 1]`
  - dataset labels are now returned in the same `[-1, 1]` space
  - NME maps centered coordinates back to `[0, 1]` before computing pixel-space error

## Result

| Run | Dataset | Samples | NME (%) | Gap vs Paper 4.67 |
| --- | --- | ---: | ---: | ---: |
| Before coordinate fix | W300 full | 689 | 153.3609 | +148.6909 |
| After coordinate fix | W300 full | 689 | 6.7528 | +2.0828 |

This changes landmark from a broken baseline to a plausible but still below-paper
result. The next check is subset reporting for common/challenging/full.

## Subset Breakdown

Saved in `results/baseline_landmark_300w_subsets/`.

| Subset | Samples | NME (%) |
| --- | ---: | ---: |
| full | 689 | 6.7528 |
| common | 554 | 5.5955 |
| challenging | 135 | 11.5021 |

The remaining gap is concentrated in the IBUG challenging subset.
