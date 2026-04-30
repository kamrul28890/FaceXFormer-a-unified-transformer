# 6. 8-Task Training Results

**Preliminary status.** A bug has been identified in the training evaluation path affecting the landmark, head pose, and age rows. Results for these three tasks are under active investigation and should not be used as paper-comparable final results. Segmentation, attribute, gender, race, and visibility are treated as reportable preliminary rows, with gender/race/visibility still marked where their values suggest metric or class-distribution issues.

## 6.1 Training Procedure

The full training configuration is given in Table 4.4. Staged co-training completed all three phases: a 3-task geometric stage, a 6-task mixed regression/classification stage, and the final 8-task stage. Checkpoints were saved throughout training, and the preliminary evaluation reported here uses the epoch-12 checkpoint from the full 8-task run on the Purdue cluster using 8 x A100 GPUs. Because raw bug-fixed training logs were not available in the current repository package, the values in this section are explicitly treated as provenance-tagged preliminary results transcribed from the final presentation/report instruction source.

## 6.2 Results

**Table 6.1: Preliminary 8-Task Training Results vs. Paper Targets**

| Task | Dataset | Metric | Paper Target | Our Result | Gap | Verdict | Bug? |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Segmentation | CelebAMask-HQ | F1 (%) | 92.01 | 85.91 | -6.10 pp | Close but below paper | No |
| Landmark | 300W | NME (%) | 4.67 full / 3.05 common | 0.0117 | - | BUG - not comparable | Yes |
| Head Pose | BIWI | MAE (deg) | 3.52 | 0.379 | - | BUG - not comparable | Yes |
| Attribute | CelebA | Acc (%) | 91.83 | 91.27 | -0.56 pp | Match | No |
| Attribute | LFWA | Acc (%) | - | 59.87 | - | Cross-dataset | No |
| Age | UTKFace | MAE (years) | 4.17 | 35.27 | +31.10 years | BUG - not comparable | Yes |
| Age | FairFace | MAE (years) | - | 31.50 | - | BUG - not comparable | Yes |
| Gender | UTKFace | Acc (%) | - | 85.04 | - | Plausible diagnostic | No |
| Gender | FairFace | Acc (%) | 95.22 | 100.00 | +4.78 pp | Suspect | No |
| Race | UTKFace | Acc (%) | - | 73.26 | - | Diagnostic | No |
| Race | FairFace | Acc (%) | - | 99.96 | - | Suspect | No |
| Visibility | COFW | Recall@P80 (%) | 72.56 | 99.25 | +26.69 pp | Suspect | No |

*Caption: Preliminary 8-task training results. Landmark, head pose, and age rows are affected by an identified training/evaluation bug and must not be compared directly to paper targets. Suspect rows greatly exceed paper targets or expected behavior and require metric/protocol verification.*

**Figure 6.1.** Preliminary training results with bug-affected and suspect rows flagged.
Source asset: `report_assets/fig5_training_results_flagged.pdf`.

## 6.3 Analysis of Valid and Reportable Rows

The segmentation result demonstrates that the rebuilt eight-task training pipeline is functional, but it does not match the paper. The trained model reaches 85.91% F1 on CelebAMask-HQ compared with the paper target of 92.01%, a gap of 6.10 percentage points. This gap is large enough to be meaningful. The most likely contributors are the architectural divergences documented in Section 3: the reproduced landmark task uses a single-token MLP head rather than the paper's 68-token hourglass design, which likely weakens the spatial gradient signal flowing into the shared representation. The missing expression and face recognition tasks may also reduce the diversity of co-training supervision. Finally, the paper mentions additional epochs for some tasks but does not specify which tasks receive them, so the uniform 12-epoch reproduction may undertrain segmentation relative to the original protocol.

Attribute prediction is the cleanest positive training result. CelebA attribute accuracy reaches 91.27%, only 0.56 percentage points below the paper target of 91.83%. This near-match supports the correctness of the CelebA loader, binary cross-entropy objective, task routing, and sampler behavior for at least one large classification task. It also aligns with the released-checkpoint baseline in Section 5, where CelebA attributes were one of the closest paper matches.

The UTKFace gender result of 85.04% is plausible but does not have a direct paper target in the current table. FairFace gender at 100.00% is not plausible as a clean scientific result and is therefore marked suspect. A saturated score can arise from class imbalance, label mapping errors, split issues, or an accuracy implementation that is not measuring the intended label set. FairFace race at 99.96% is similarly suspect, especially because the current model head and FairFace source taxonomy do not align perfectly.

Visibility reaches 99.25% Recall@P80 on COFW, far above the paper target of 72.56%. This should not be interpreted as a strong improvement over the paper. Recall@P80 is sensitive to the exact precision-recall construction, label polarity, interpolation rule, and class balance in the visibility labels. A model or metric that predicts most landmarks as visible can appear strong under some recall-oriented summaries if the test split is dominated by visible landmarks. This row should be retained as a warning signal that the visibility protocol needs verification.

## 6.4 Deferred Bug-Affected Analysis

The detailed analysis of landmark, head pose, and age training failures is deferred until the bug-fixed retraining run is complete. The current values are useful only as evidence that the training evaluation path is faulty: landmark NME of 0.0117 is physically implausible, head-pose MAE of 0.379 is not safely interpretable without unit/protocol verification, and age MAE above 30 years indicates either a serious training failure, metric error, or both. These rows will be replaced and analyzed in the final report version after retraining.
