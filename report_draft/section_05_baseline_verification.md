# 5. Baseline Verification

This section evaluates the released FaceXFormer checkpoint in inference-only mode. The purpose is not to train a model, but to verify whether the public checkpoint is loadable and whether its predictions can reproduce the paper's reported metrics under the eight-task evaluation pipeline reconstructed in this project.

## 5.1 Experimental Setup

We loaded the released checkpoint at `checkpoints/ckpts/model.pt` and ran task-specific evaluation on the corresponding dataset splits. The checkpoint is 1,104,869,851 bytes and has SHA256 hash `327A755849BA64D336FB96589FF87B27E84A12BE1ECF8BCFAA503D66F803286D`. The fixed baseline manifest reports `checkpoint_loaded: true`, confirming that the checkpoint is functional in the reproduced code path. The run was generated under `results/baseline_current_8task_fresh_fixed/` with dataset root `D:\Projects\facexformer-main\datasets\datasets`, batch size 64, seed 0, and no sample cap. The run scope is the current eight-task implementation, not the full ten-task paper system.

For each task, the evaluator recorded raw metrics, normalized metrics, normalized units, paper targets where available, and paper-relative gaps. The fixed manifest contains 14 result rows and 129,060 evaluated samples because it includes diagnostic 300W common/challenging subset rows in addition to 300W full.

## 5.2 Results

**Table 5.1: Baseline Inference Results - Released Checkpoint vs. Paper Targets**

| Task | Dataset | Samples | Metric | Paper Target | Raw Value | Normalized Value | Gap | Status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Segmentation | CelebAMask-HQ | 2,000 | F1 (%) | 92.01 | 0.9177 | 91.77 | -0.24 pp | MATCH |
| Landmark | 300W full | 689 | NME (%) | 4.67 | 6.7528 | 6.75 | +2.08 | CLOSE GAP |
| Landmark | 300W common | 554 | NME (%) | - | 5.5955 | 5.60 | - | Diagnostic |
| Landmark | 300W challenging | 135 | NME (%) | - | 11.5021 | 11.50 | - | Diagnostic |
| Head Pose | BIWI | 15,678 | MAE (deg) | 3.52 | 20.6497 | 20.65 | +17.13 deg | LARGE GAP |
| Attribute | CelebA | 19,962 | Acc (%) | 91.83 | 0.9172 | 91.72 | -0.11 pp | MATCH |
| Attribute | LFWA | 13,143 | Acc (%) | - | 0.6106 | 61.06 | - | Cross-dataset |
| Age | UTKFace | 3,556 | MAE (years) | 4.17 | 1.1714 | 1.17 | -3.00 years | SUSPECT |
| Age | FairFace | 21,908 | MAE (years) | - | 6.2990 | 6.30 | - | No target |
| Gender | UTKFace | 3,556 | Acc (%) | - | 0.9918 | 99.18 | - | No target |
| Gender | FairFace | 21,908 | Acc (%) | - | 0.9311 | 93.11 | - | Diagnostic |
| Race | UTKFace | 3,556 | Acc (%) | - | 0.9803 | 98.03 | - | No target |
| Race | FairFace | 21,908 | Acc (%) | - | 0.6459 | 64.59 | - | Diagnostic |
| Visibility | COFW | 507 | Recall@P80 (%) | 72.56 | 0.8713 | 87.13 | +14.57 pp | ABOVE PAPER |

*Caption: Baseline inference results for the released checkpoint. Source: `results/baseline_current_8task_fresh_fixed/gap_analysis_baseline_current_8task.json`. MATCH indicates within 1 percentage point of the paper target when the metric is a percentage. Diagnostic rows do not have direct paper targets under this reproduction protocol.*

**Figure 5.1.** Baseline inference bar chart comparing paper targets and normalized released-checkpoint values for paper-target rows.
Source asset: `report_assets/fig4_baseline_inference_bars.pdf`.

## 5.3 Analysis

The strongest validation comes from segmentation and CelebA attributes. Segmentation reaches 91.77% F1 against the paper target of 92.01%, a gap of only -0.24 percentage points. Attribute prediction reaches 91.72% accuracy against the paper target of 91.83%, a gap of -0.11 percentage points. These two near-matches show that the released checkpoint is meaningful, the core inference path is functional, and the reconstructed normalization framework is sufficient for at least two major paper-target tasks.

Landmark detection is plausible but behind the paper. On 300W full, the released checkpoint obtains 6.75 NME compared with the paper's 4.67 target. The common/challenging subset split is also informative: 5.60 NME on common and 11.50 NME on challenging. This pattern is qualitatively reasonable because the challenging subset contains harder poses, occlusions, and expression variation. The remaining gap may reflect the landmark architecture discrepancy documented in Section 3, protocol differences, or residual preprocessing mismatch.

Head pose remains the largest unresolved baseline mismatch. The fixed manifest reports 20.65 degrees MAE on BIWI compared with the paper target of 3.52 degrees. Earlier intermediate runs mixed radians and degrees, but the fresh fixed manifest records head pose as degrees already; therefore this is not simply a display-unit error in the final baseline table. It should be treated as a real unresolved protocol or implementation gap until the pose preprocessing, axis convention, crop generation, and rotation conversion are verified against the original evaluation protocol.

Age and visibility should be interpreted carefully. UTKFace age MAE is 1.17 years, which is substantially better than the paper target of 4.17 years and is therefore suspicious rather than a clean success. Possible explanations include a split mismatch, label-bin conversion issue, leakage, or a metric computation difference. Visibility reaches 87.13% Recall@P80 on COFW compared with the paper target of 72.56%, also above paper. This may reflect a corrected recall computation or a protocol difference in how precision thresholds and visibility labels are defined.

The remaining rows are useful diagnostics rather than direct paper comparisons. LFWA attribute accuracy drops to 61.06%, showing a large cross-dataset generalization gap relative to CelebA. FairFace gender reaches 93.11%, while FairFace race reaches 64.59% under a repo-compatible mapping; the latter is especially sensitive to class-taxonomy mismatch because FairFace has seven race categories while the current model head uses five. Overall, the baseline run largely validates the released checkpoint for segmentation and attributes, but it also exposes unresolved evaluation-protocol risks for head pose, age, visibility, and cross-dataset demographic tasks.
