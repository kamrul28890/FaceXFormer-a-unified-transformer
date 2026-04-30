# 8. Discussion

## 8.1 What Worked: Validated Components

Several components of the reproduction are validated by converging evidence across the released-checkpoint baseline and the trained eight-task run. The strongest signal is CelebA attribute prediction: the released checkpoint reaches 91.72% against the paper target of 91.83%, and the reproduced trained model reaches 91.27%. These near-matches support the CelebA data loader, task routing, binary cross-entropy loss, metric normalization, and multi-task sampler for at least one large-scale classification task.

Segmentation provides a second, more qualified validation. The released checkpoint reaches 91.77% F1, nearly matching the paper's 92.01% target. The trained reproduction reaches 85.91%, below the paper but still high enough to show that the rebuilt eight-task training pipeline is learning meaningful dense face representations rather than collapsing. The staged training strategy also worked operationally: it allowed the pipeline to be debugged first on geometric tasks, then expanded to classification tasks, and finally extended to the full eight-task reproduction scope.

## 8.2 Failure Mode A: Metric Documentation Gaps

The baseline evaluation exposed how easily a weights-only release can be misread when metric units are not documented with the same care as model architecture. Raw script outputs may be fractions, percentages, years, normalized coordinate errors, radians, or degrees. Without a normalization manifest, a valid result can look like a catastrophic failure, or a suspect result can look like a breakthrough. For example, a raw segmentation F1 fraction near 0.9177 corresponds to 91.77%, which nearly matches the paper. Interpreting the raw fraction as a percentage would lead to an obviously wrong conclusion.

This is not merely a formatting problem. Metric normalization determines whether independent reproducers can compare results at all. The same checkpoint may appear to fail or succeed depending on whether the evaluator knows the unit convention, label mapping, and postprocessing rule. The fixed baseline manifest improves this by recording `raw_metric`, `normalized_metric`, `normalized_metric_unit`, and `normalization_rules`. A table like Table 4.5 should be considered part of the reproducibility artifact, not an optional reporting convenience.

## 8.3 Failure Mode B: Multi-Task Loss Scale and Age

The age rows are not final because the training evaluation bug affects them, but they still highlight a broader reproducibility risk: the paper's loss coefficients are underspecified. Multi-task training is not defined solely by the list of losses. It is defined by the relative scale at which those losses enter the optimizer. A Dice loss is bounded near the [0, 1] range, binary cross-entropy for a balanced binary decision is on the order of log(2), and a continuous age error can naturally live on a scale of tens of years if it is not normalized. Setting all task weights to 1.0 may therefore produce very different gradient magnitudes across tasks.

The reported preliminary age MAE above 30 years cannot yet be treated as a final result, but it is consistent with the kind of instability that loss-scale imbalance can create. The important reproducibility point does not depend on the final rerun value: lambda values are first-order hyperparameters in a multi-task system. Omitting them from the paper, appendix, and public code means that an independent reproducer must either guess, tune, or contact the authors. Future versions of this reproduction should test gradient-norm balancing, uncertainty weighting, or explicit loss normalization as controlled variants rather than treating `lambda_i = 1.0` as a harmless default.

**Figure 8.1.** Loss scale comparison across tasks.
Source asset: `report_assets/fig7_loss_scale_comparison.pdf`.

## 8.4 Failure Mode C: Architectural Divergence in Landmark and Head Pose

The most consequential architectural divergence is the landmark pathway. The paper describes 68 landmark tokens and an hourglass landmark head. The reproduced implementation uses one landmark token followed by an MLP that emits all 136 coordinate values. This changes the role of landmark supervision in the shared representation. With 68 tokens, each keypoint can form its own query into the face representation and send localized gradients back through the FaceX decoder. With one token, that geometric signal is compressed before prediction, reducing the opportunity for per-keypoint attention and task-specific spatial feedback.

This may partly explain the segmentation gap in the trained model. Segmentation depends on high-quality spatial face tokens. If the landmark task provides weaker spatial supervision than the paper's architecture intended, the shared representation may be less precise, even if the segmentation head itself is implemented correctly. Head pose contains a related representation mismatch: the paper describes 9 rotation-matrix tokens, while the implementation predicts 3 Euler-angle values from one token. These divergences do not make the reproduction invalid, but they mean the reproduced model is not an exact implementation of the paper's architecture.

## 8.5 What Remains To Be Done

- Fix the training/evaluation bug affecting landmark, head pose, and age, then rerun and report final values.
- Verify BIWI head-pose preprocessing, axis convention, crop generation, and rotation conversion against the original protocol.
- Verify COFW Recall@P80 against the paper definition, including label polarity, thresholding, and interpolation.
- Investigate FairFace gender/race saturated or near-saturated values by checking split composition, label mapping, and class distributions.
- Run a controlled loss-weighting study, including `lambda_i = 1.0`, tuned weights, and gradient-norm balancing.
- Extend the reproduction to the full ten-task setting if resources permit: expression recognition with RAF-DB/AffectNet and face recognition with MS1MV3 plus PartialFC/ArcFace.

## 8.6 Implications for Reproducibility Standards

The main lesson is that the unit of reproducibility is the complete training pipeline, not the pretrained checkpoint. A checkpoint can verify that some inference behavior is real, as the segmentation and attribute baselines show. It cannot explain how the model was trained, how datasets were sampled, how losses were weighted, how augmentations were applied, or how ambiguous metrics were normalized. For a multi-task model, those choices are not secondary. They define the experiment.

Metric documentation should be treated as infrastructure. Papers and repositories should publish the raw metric convention, normalized reporting unit, label mapping, and evaluation protocol for every task. This is especially important when a single model spans segmentation, regression, classification, verification, and precision-recall metrics. Without this documentation, independent evaluation becomes a guessing exercise even when the checkpoint loads successfully.

Loss coefficients also need to be reported explicitly. In a single-task model, a missing scalar may be harmless; in a multi-task model, the lambda vector determines how tasks compete for shared representation capacity. Requiring private email confirmation for these values is not sufficient for scientific reproducibility. A reproducible release should include training code, dataset adapters or exact preprocessing instructions, sampler logic, loss definitions, loss coefficients, evaluation scripts, metric normalization rules, and checkpoint metadata.
