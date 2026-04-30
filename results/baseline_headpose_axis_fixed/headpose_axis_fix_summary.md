# Head-Pose Axis Fix Summary

Generated on 2026-04-27 after standardizing the head-pose label convention.

## What Changed

- Standardized internal head-pose labels to `[yaw, pitch, roll]` in radians.
- Updated `W300LPDataset` to convert 300W-LP `Pose_Para` from
  `[pitch, yaw, roll]` into `[yaw, pitch, roll]`.
- Updated `BIWIDataset` to return `[yaw, pitch, roll]`, matching the loss code.

## Result

| Run | Dataset | Samples | MAE (deg) | Gap vs Paper 3.52 |
| --- | --- | ---: | ---: | ---: |
| Before axis fix | BIWI | 15678 | 25.1768 | +21.6568 |
| After axis fix | BIWI | 15678 | 20.6497 | +17.1297 |

This fixes a real axis-order inconsistency, but the head-pose baseline is still
far from the paper. Remaining causes likely include BIWI preprocessing/cropping,
Euler-angle conversion protocol, and checkpoint/training-domain mismatch.
