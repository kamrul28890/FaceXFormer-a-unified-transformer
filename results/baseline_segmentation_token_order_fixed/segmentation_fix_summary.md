# Segmentation Mapping Fix Summary

Generated on 2026-04-27 after fixing CelebAMask-HQ label construction.

## What Changed

- Removed the previous `min(i, max_class)` clamp that collapsed multiple
  CelebAMask-HQ parts into class 10.
- Inferred the checkpoint's segmentation token order from model predictions
  against raw CelebAMask-HQ part masks.
- Updated the label map to:
  - 0 background
  - 1 skin
  - 2 right eyebrow
  - 3 left eyebrow
  - 4 right eye
  - 5 left eye
  - 6 nose
  - 7 upper lip
  - 8 mouth
  - 9 lower lip
  - 10 hair
- Updated baseline segmentation reporting to compute one global macro-F1 over
  the full split instead of averaging per-batch macro-F1.

## Result

| Run | Dataset | Samples | F1 (%) | Gap vs Paper 92.01 |
| --- | --- | ---: | ---: | ---: |
| Before fix | CelebAMask-HQ | 2000 | 23.7049 | -68.3051 |
| First explicit map, wrong order | CelebAMask-HQ | 2000 | 18.1395 | -73.8705 |
| Token-order fixed | CelebAMask-HQ | 2000 | 91.7653 | -0.2447 |
