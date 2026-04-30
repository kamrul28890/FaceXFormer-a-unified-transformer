# Fresh Fixed 8-Task Baseline Summary

Generated from `results/baseline_current_8task_fresh_fixed/` on 2026-04-27.

Checkpoint: `checkpoints/ckpts/model.pt`
Checkpoint SHA256: `327A755849BA64D336FB96589FF87B27E84A12BE1ECF8BCFAA503D66F803286D`
Dataset root: `D:\Projects\facexformer-main\datasets\datasets`
Run scope: current repo 8-task implementation, not the full 10-task paper system.

## Paper-Target Rows

| Task | Dataset | Ours | Paper | Gap |
| --- | --- | ---: | ---: | ---: |
| segmentation | CelebAMask-HQ | 91.7652% F1 | 92.01% | -0.2448 |
| landmark | 300W full | 6.7528 NME | 4.67 | +2.0828 |
| headpose | BIWI | 20.6497 deg MAE | 3.52 | +17.1297 |
| attribute | CelebA | 91.7211% acc | 91.83% | -0.1089 |
| age | UTKFace | 1.1714 years MAE | 4.17 | -2.9986 |
| visibility | COFW | 87.1264% Recall@P80 | 72.56% | +14.5664 |

## Additional Rows

| Task | Dataset | Result |
| --- | --- | ---: |
| landmark | 300W common | 5.5955 NME |
| landmark | 300W challenging | 11.5021 NME |
| attribute | LFWA | 61.0595% acc |
| age | FairFace | 6.2990 years MAE |
| gender | UTKFace | 99.1845% acc |
| gender | FairFace | 93.1121% acc |
| race | UTKFace | 98.0315% acc |
| race | FairFace | 64.5883% acc |

## Interpretation

- Segmentation and CelebA attributes are now very close to the paper.
- Landmark is plausible but still behind, especially on the 300W challenging subset.
- Head pose remains the largest unresolved mismatch.
- Age and visibility should be treated carefully because their exact paper protocol still needs verification.
- FairFace race is mapped from 7 race groups into the current 5-class model head, so it is useful diagnostically but not directly paper-comparable.
