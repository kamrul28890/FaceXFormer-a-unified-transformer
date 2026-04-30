# 4. Reproduction Methodology

## 4.1 Paper-vs-Code Gap Analysis

The reproduction began with a structured audit of the FaceXFormer paper against the released repository. Each gap was categorized using four fields: **Paper Claim**, **Repo Reality**, **Our Resolution**, and **Impact**. Impact is used as a practical reproducibility label: **CRITICAL** means the reproduction cannot proceed without rebuilding the missing component; **HIGH** means the gap can materially change results; **MED** means the gap is measurable but bounded; **NONE** means no substantive divergence was found.

**Table 4.1: Structured Paper-vs-Code Gap Analysis**

| # | Component | Paper Claim | Repo Reality | Our Resolution | Impact |
| ---: | --- | --- | --- | --- | --- |
| 1 | Task scope | 10 tasks | 8 tasks; no expression or face recognition path | Reproduced 8 tasks; dropped expression and recognition | HIGH |
| 2 | Task token count | Segmentation: one per class; landmarks: 68; head pose: 9; other tasks: 1 each | 18 total tokens: 11 mask + 1 token each for landmark, pose, attribute, visibility, age, gender, race | Followed repo tokenization for reproduced model; documented mismatch | HIGH |
| 3 | Segmentation class mapping | One token per parsing class; paper appendix mentions 19 CelebAMask-HQ classes | Current implementation uses 11 mask tokens/classes | Used repo mapping; treated as class-mapping ambiguity | MED |
| 4 | Landmark head | Hourglass network on 68 landmark tokens | Simple MLP on one landmark token | Used repo MLP and documented as architectural divergence | HIGH |
| 5 | Head pose representation | 9 tokens representing a 3x3 rotation matrix | One pose token producing 3 Euler-angle values | Used repo representation; converted to rotation matrices for loss/eval where needed | HIGH |
| 6 | Loss coefficients | Joint weighted objective, but lambda values not published | No public loss implementation in released inference code | Reported reproduction used lambda_i = 1.0 following author email confirmation | HIGH |
| 7 | Training loop | Implied by paper results | Not released | Rebuilt PyTorch DDP training loop with checkpointing and evaluation | CRITICAL |
| 8 | Dataset loaders | Implied for all paper datasets | Not released | Built task-specific dataset adapters for the 8 reproduced tasks | CRITICAL |
| 9 | Multi-task sampler | Upsampling-based balancing described only at high level | Not released | Implemented upsampling and balanced per-batch task sampling | HIGH |
| 10 | Epoch schedule | 12 epochs plus unspecified extra epochs for some tasks | Extra-task schedule absent | Trained 12 epochs for all reproduced tasks | MED |
| 11 | Augmentation | Appendix F describes task-specific augmentations | No released training augmentations | Implemented task-specific augmentation pipelines from appendix/code reconstruction | MED |
| 12 | Expression recognition | Included in paper results | Absent from codebase | Dropped; RAF-DB/AffectNet not included in final scope | HIGH |
| 13 | Face recognition | PartialFC + ArcFace on MS1MV3 | Absent from codebase | Dropped; MS1MV3 and PartialFC training were compute/data prohibitive | HIGH |

*Caption: Complete paper-vs-code gap analysis. Impact ratings describe how strongly each gap affects independent reproduction.*

This audit shows that FaceXFormer is a weights-only release for training reproducibility purposes. The checkpoint and inference code are valuable for baseline verification, but they do not define the training process that produced the reported multi-task model. The most critical omissions are the missing training loop, missing dataset loaders, missing sampler, and unpublished loss coefficients. The architecture also contains consequential mismatches between the paper description and the released/reproduced implementation, especially for landmark detection and head pose estimation.

**Figure 4.1.** Gap analysis summary heatmap.
Source asset: `report_assets/fig8_gap_analysis_heatmap.pdf`.

## 4.2 Dataset Preparation

The reproduced pipeline covers eight tasks and uses task-specific train/evaluation adapters. Evaluation includes the primary paper-comparable datasets plus diagnostic cross-dataset rows used to understand generalization and protocol mismatch.

**Table 4.2: Datasets, Splits, and Sample Counts**

| Task | Dataset | Train Split | Evaluation Split | Eval Samples | Preprocessing |
| --- | --- | ---: | --- | ---: | --- |
| Segmentation | CelebAMask-HQ | 24,000 to 28,000, depending on split convention | Test | 2,000 | Resize to 224x224; repo maps masks to 11 classes |
| Landmark | 300W | 3,148 | Full test set | 689 | 5-point affine/ArcFace-style alignment; 68 keypoints |
| Head Pose | 300W-LP train, BIWI eval | 122,450 for 300W-LP | BIWI | 15,678 | Loose crop / resized crop; preserve pose labels |
| Attribute | CelebA | 162,770 | CelebA test | 19,962 | 40 binary labels; aligned face crops |
| Attribute | LFWA | Not used for training | LFWA | 13,143 | Cross-dataset diagnostic only |
| Age | UTKFace | Train split | UTKFace eval split | 3,556 | Age parsed from filename; mapped to age bins as needed |
| Age | FairFace | Train split | FairFace eval split | 21,908 | Age-group labels; multi-ethnic distribution |
| Gender | UTKFace | Shared with age/race | UTKFace eval split | 3,556 | Binary label |
| Gender | FairFace | Shared with age/race | FairFace eval split | 21,908 | Binary label |
| Race | UTKFace | Shared with age/gender | UTKFace eval split | 3,556 | 5-class label |
| Race | FairFace | Shared with age/gender | FairFace eval split | 21,908 | 7-class source labels mapped into repo-compatible taxonomy where needed |
| Visibility | COFW | 1,345 | Test | 507 | 29 per-landmark visibility/occlusion labels |

*Caption: Datasets used for the eight-task reproduction and evaluation. Counts reflect the current fixed evaluation manifest where available.*

**Sample count reconciliation.** The report outline and presentation use 127,735 total evaluation samples across 12 dataset-task combinations. The fresh fixed manifest in `results/baseline_current_8task_fresh_fixed/` reports 129,060 samples across 14 result rows because it includes 300W common/challenging subset rows in addition to the 300W full row. It also evaluates 300W full as 689 images, matching the paper appendix, whereas the outline table lists a 53-sample landmark row. In this report, we use dataset-specific counts from the current manifest when discussing concrete evaluation runs, and we reserve the 127,735 figure for the original scoped presentation total.

Landmark preprocessing required an engineering choice not fully specified in the paper. The reproduction derives five anchor points from the 68 landmark annotations and applies an ArcFace-style affine alignment before resizing to the target resolution. During training, geometric transforms are applied jointly to the image and landmarks so the labels remain consistent. LFWA is treated as a diagnostic cross-dataset attribute evaluation; it is not a paper target row and should not be interpreted as a reproduction failure.

## 4.3 Augmentation Pipeline

Task-specific augmentations were implemented in `datasets.py` and applied only during training. Evaluation uses clean resized/aligned inputs with ImageNet normalization. The paper appendix describes augmentation categories, but not every probability and implementation detail is fully specified. Where the paper was ambiguous, the reproduction used the closest task-appropriate implementation and documented the choice.

**Table 4.3: Augmentation Operations and Per-Task Application**

| Augmentation | Parameters | Segmentation | Landmark | Head Pose | Attribute | Age/Gender/Race | Visibility |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Resize / normalization | 224x224; ImageNet mean/std | Yes | Yes | Yes | Yes | Yes | Yes |
| ArcFace-style alignment | 5-point template | No | Yes | No | No | No | No |
| Random rotation | +/-18 deg | No | Yes | No | Yes | Yes | No |
| Random scaling | 0.9 to 1.1 | No | Yes | No | Yes | Yes | No |
| Random translation / crop | 1 to 5 percent or resized crop | No | Yes | Crop proxy | Yes | Yes | No |
| Horizontal flip | p = 0.5 with label remapping when needed | No | Yes | No | Yes | Yes | Yes |
| Grayscale | p = 0.1 to 0.2 | No | Yes | Yes | Yes | Yes | Yes |
| Gaussian blur | radius 0.5 to 2.0 px | No | Yes | Yes | Yes | Yes | Yes |
| Random occlusion | random patch | No | Yes | No | No | No | No |
| Gamma adjustment | gamma in [0.7, 1.3] | No | Yes | Yes | Yes | Yes | Yes |

*Caption: Task-specific augmentation operations in the reproduced training pipeline. Head pose and visibility avoid geometric transforms that would invalidate labels unless labels can be transformed consistently.*

This differs slightly from the high-level outline because the actual implementation is more conservative for segmentation, head pose, and visibility. Segmentation uses clean resizing and normalization in the current dataset adapter. Head pose uses a random resized crop proxy plus photometric perturbations rather than arbitrary in-plane rotation, because rotating the image without updating the pose label would corrupt the supervision. Visibility supports horizontal flip by reordering the 29 visibility labels using a symmetric landmark mapping.

## 4.4 Loss Functions

The paper defines a weighted joint objective over tasks:

```text
L = lambda_seg L_seg
  + lambda_lnd L_lnd
  + lambda_hpe L_hpe
  + lambda_attr L_attr
  + lambda_age L_age
  + lambda_g/r L_g/r
  + lambda_vis L_vis
```

The full paper objective also includes expression and face recognition terms, but those tasks are outside the eight-task reproduction scope. For the reported reproduction run, all task weights were set to `lambda_i = 1.0` following author email confirmation. The paper does not publish these values. The current repository contains later exploratory non-uniform `LOSS_WEIGHTS`; those tuned values are not treated as the paper-faithful reproduction setting.

**Segmentation loss.** `L_seg` is the mean of Dice loss and pixelwise cross-entropy over the reproduced segmentation classes. Dice loss addresses class imbalance in small facial regions, while cross-entropy provides dense per-pixel gradients.

**Landmark loss.** `L_lnd` is a STARLoss-style landmark loss applied to the 68-keypoint output of the reproduced MLP head. This is already downstream of an architectural divergence: the paper specifies an hourglass head operating from a 68-token landmark representation, while the reproduced model uses a single landmark token and emits 136 coordinate values.

**Head pose loss.** `L_hpe` is geodesic loss on SO(3). The implementation predicts Euler angles, converts prediction and target into rotation matrices, and computes the angular distance between rotations. This is geometrically more appropriate than direct Euclidean distance, but it differs from the paper's stated 9-token rotation-matrix output.

**Attribute loss.** `L_attr` is binary cross-entropy with logits over 40 CelebA attributes. Each attribute is treated as an independent binary label and the loss is averaged over labels and samples.

**Age loss.** `L_age` combines cross-entropy over age bins with an L1-style age expectation term. The reproduced implementation uses eight age groups with representative bin centers. This differs from a fully continuous age regression objective and should be treated as an implementation decision where the paper is underspecified.

**Gender and race loss.** `L_g/r` is standard cross-entropy. Gender uses two classes. Race uses the repo-compatible class taxonomy, with UTKFace and FairFace requiring careful handling because their race labels do not use identical class sets.

**Visibility loss.** `L_vis` is binary cross-entropy with logits over 29 per-landmark visibility values. Invalid or missing visibility entries are filtered before averaging when necessary.

## 4.5 Task-Balanced Multi-Task Sampler

The reproduction implements an upsampling-based multi-task sampler. For each task, smaller datasets are repeated and shuffled until they match the effective size of the largest task pool. A custom balanced batch sampler then draws a fixed or near-fixed number of samples from each active task in every batch. When the batch size is not divisible by the number of active tasks, the extra sample slots are randomly assigned across tasks per batch to prevent systematic bias. This implements the paper's high-level upsampling description, but the paper does not specify whether the original authors used pure repetition, weighted sampling, shuffled repetition, or another balancing rule.

## 4.6 Training Configuration

**Table 4.4: Full Training Configuration**

| Parameter | Value |
| --- | --- |
| Hardware | 8 x NVIDIA A100 GPUs on Purdue institutional cluster |
| Framework | PyTorch with DistributedDataParallel |
| Distributed backend | NCCL |
| Precision | FP16 mixed precision for cluster-scale runs |
| Batch size | 48 per GPU, 384 effective |
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| LR decay | multiply by 0.1 at epochs 6 and 10 |
| Weight decay | 1e-5 |
| Total epochs | 12 |
| Additional epochs | Not applied; paper does not identify which tasks receive extra epochs |
| Backbone initialization | ImageNet-pretrained Swin-B |
| Input resolution | 224x224 |
| Loss coefficients | lambda_i = 1.0 for reported paper-faithful reproduction |
| Checkpointing | Per-epoch checkpointing and resume support |
| Gradient stabilization | Gradient clipping in training loop |

*Caption: Training hyperparameters used for the reported reproduction. Later exploratory config edits are not treated as the paper-faithful setting.*

## 4.7 Staged Training Strategy

The eight-task training run was staged to reduce the risk of debugging all tasks simultaneously. Stage 1 trained the geometric core: segmentation, landmark detection, and head pose estimation. This validated the sampler, dense prediction path, coordinate regression path, and pose loss. Stage 2 added attribute prediction, age estimation, and gender classification, expanding the system to multi-label and categorical classification tasks with different loss scales. Stage 3 added race and visibility, producing the full eight-task reproduction. Each stage initialized from the checkpoint produced by the previous stage.

**Figure 4.2.** Staged 3-to-6-to-8 task training strategy.
Source asset: `report_assets/fig3_staged_training_timeline.pdf`.

## 4.8 Metric Normalization Framework

A major reproducibility problem was that raw script outputs were not always in the same unit as paper-reported metrics. The reproduction therefore records both raw and normalized values and uses explicit normalization rules from the baseline manifest.

**Table 4.5: Metric Normalization Rules**

| Task | Raw Script Output | Paper Unit | Normalization | Example |
| --- | --- | --- | --- | --- |
| Segmentation | F1 fraction | F1 percent | multiply by 100 | 0.9177 -> 91.77% |
| Landmark | NME percent | NME percent | no conversion | 6.75 -> 6.75% |
| Head Pose | MAE in degrees in current fixed manifest | MAE degrees | no conversion in fixed manifest | 20.65 -> 20.65 deg |
| Attribute | Accuracy fraction | Accuracy percent | multiply by 100 | 0.9172 -> 91.72% |
| Age | MAE in years | MAE years | no conversion | 1.17 -> 1.17 years |
| Gender | Accuracy fraction | Accuracy percent | multiply by 100 | 0.9918 -> 99.18% |
| Race | Accuracy fraction | Accuracy percent | multiply by 100 | 0.9803 -> 98.03% |
| Visibility | Recall fraction | Recall percent | multiply by 100 | 0.8713 -> 87.13% |

*Caption: Normalization rules used to convert raw evaluation outputs to report-ready units. Source: `results/baseline_current_8task_fresh_fixed/baseline_current_8task_manifest.json`.*

The scientific point is simple: without a normalization layer, evaluation numbers can look catastrophically wrong for reasons unrelated to model quality. A raw fraction such as 0.9177 is not a 0.92% segmentation result; it is 91.77% after conversion. Similarly, older smoke and metric-check runs mixed raw radians, degrees, fractions, and percentages in ways that made direct paper comparison unsafe. The fixed manifest makes the unit convention explicit through its `normalization_rules` field. That kind of metric documentation is infrastructure, not cosmetic reporting.
