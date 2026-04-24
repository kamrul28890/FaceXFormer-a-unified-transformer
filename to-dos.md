# FaceXFormer Reproduction Todo List

Generated from:
- `Reproducing_FaceXFormer__A_Unified_Transformerfor_Multi_Task_Facial_Analysis.pdf`
- Current repo audit on 2026-04-24
- Existing `FaceXFormer_Implementation_Plan.md`

## Repo Snapshot

- [x] Repo contains an 8-task FaceXFormer training scaffold: segmentation, landmarks, head pose, attributes, age, gender, race, visibility.
- [x] Datasets have been copied into this repo under `datasets/`.
- [x] Current model is not paper-accurate yet: it uses 18 task tokens and 11 segmentation classes.
- [x] Expression recognition is intentionally excluded in current code comments and wiring.
- [ ] Face recognition training path is missing. Current dataset copy appears to include `vggface2_cache`; proposal expects MS1MV3 plus verification sets.
- [ ] Dataset roots still need cleanup: many dataset classes default to `../facexformer-my/datasets` instead of this repo's `./datasets`.

## Phase 0 - Baseline Verification

- [ ] Install and verify the current environment with `python test_setup.py`.
- [ ] Run a short dry run with `python train.py --dry-run` or `python train_simple.py`.
- [ ] Run released FaceXFormer inference/checkpoint on sample images.
- [ ] Evaluate released checkpoint where possible and create `gap_analysis_baseline.csv`.
- [ ] Record paper target vs reproduced metric vs gap for:
  - [ ] CelebAMask-HQ parsing F1 target about 92.01
  - [ ] 300W landmark NME target about 4.67
  - [ ] BIWI head-pose MAE target about 3.52
  - [ ] CelebA attribute accuracy target about 91.83
  - [ ] UTKFace age MAE target about 4.17
  - [ ] COFW visibility recall@80%P target about 72.56
  - [ ] RAF-DB expression accuracy target about 88.24
  - [ ] LFW/CFP-FP/AgeDB/CALFW/CPLFW face-recognition mean accuracy target about 95.94

## Phase 1 - Paper-Accurate Architecture

- [ ] Update `config.py` task-token config from 18 tokens to paper-style 142 tokens:
  - [ ] 68 landmark tokens
  - [ ] 9 head-pose tokens
  - [ ] 19 segmentation tokens
  - [ ] 40 attribute tokens
  - [ ] 1 each for age, gender, race, expression, face recognition, visibility
- [ ] Update `network/models/facexformer.py` `FaceDecoder` embeddings and token slicing.
- [ ] Change segmentation from 11 classes to 19 classes after verifying the exact CelebAMask-HQ label mapping.
- [ ] Update segmentation hypernetwork output to use 19 semantic-class tokens.
- [ ] Replace single-token landmark MLP with a paper-aligned landmark head:
  - [ ] Start with a minimum viable 68-token MLP baseline.
  - [ ] Add/document lightweight hourglass-style head.
  - [ ] Keep the direct MLP as an ablation.
- [ ] Change head-pose output from Euler-angle `[B, 3]` to 9-token rotation-matrix `[B, 9]`.
- [ ] Change attribute head from one token producing 40 logits to 40 tokens producing one logit each.
- [ ] Add expression and face-recognition output heads to the model return path.
- [ ] Update all call sites in `train.py`, `evaluate.py`, and `losses.py` to match the expanded model outputs.

## Phase 2 - Dataset Root and Inventory Cleanup

- [ ] Standardize dataset root handling around `./datasets` or a single config/CLI value.
- [ ] Decide whether to point loaders at top-level `datasets/` or nested `datasets/datasets/`.
- [ ] Verify expected folders for existing tasks:
  - [ ] CelebAMask-HQ
  - [ ] 300W
  - [ ] 300W-LP
  - [ ] CelebA
  - [ ] UTKFace
  - [ ] FairFace
  - [ ] COFW
  - [ ] BIWI
  - [ ] LFWA
  - [ ] 300VW if needed for test-only landmark evaluation
- [ ] Clear or regenerate stale cache files after path changes.
- [ ] Add a dataset inventory script that reports found/missing folders, sample counts, and label files.

## Phase 3 - Expression Recognition

- [ ] Add `RAFDBDataset` from `datasets/RAFDB/basic`.
- [ ] Add `AffectNetDataset` from the copied AffectNet location if labels are available.
- [ ] Confirm expression class convention:
  - [ ] RAF-DB uses 7 classes.
  - [ ] AffectNet may use 7 or 8 classes depending on the filtered split.
- [ ] Add task id `8` for expression.
- [ ] Add expression CE loss in `losses.py`.
- [ ] Add expression metrics to `evaluate.py`.
- [ ] Add expression to train/test dataset dictionaries and sampler.
- [ ] Add expression augmentations:
  - [ ] rotation +-18 degrees
  - [ ] scaling +-10%
  - [ ] translation 1% of 224
  - [ ] horizontal flip 50%
  - [ ] grayscale 10%
  - [ ] Gaussian blur 10%
  - [ ] color jitter 10%
  - [ ] gamma adjustment 10%

## Phase 4 - Face Recognition

- [ ] Decide face-recognition dataset:
  - [ ] Preferred per proposal: MS1MV3, about 5.1M images and 93K identities.
  - [ ] Available local candidate: `datasets/vggface2_cache`.
  - [ ] Document the final decision and expected impact.
- [ ] Add face-recognition task id `9`.
- [ ] Implement face-recognition dataset class with identity labels.
- [ ] Add five-point alignment preprocessing before model input.
- [ ] Add face-recognition embedding head.
- [ ] Implement ArcFace loss.
- [ ] Implement or defer PartialFC; document if starting with standard ArcFace.
- [ ] Add verification evaluation for LFW, CFP-FP, AgeDB, CALFW, and CPLFW if datasets are available.
- [ ] Add a one-time conversion/preprocessing tool if using MXNet `.rec` MS1MV3 data.

## Phase 5 - Loss and Metric Corrections

- [ ] Replace landmark smooth-L1 approximation with STARLoss or a clearly documented STARLoss approximation.
- [ ] Update geodesic head-pose loss to accept `[B, 9]` rotation-matrix predictions.
- [ ] Update segmentation Dice + CE loss for 19 classes and correct `ignore_index` behavior.
- [ ] Add expression CE loss.
- [ ] Add face-recognition ArcFace or PartialFC+ArcFace loss.
- [ ] Add loss weights for expression and face recognition.
- [ ] Start with all omitted proposal lambda values at 1.0, then add gradient-norm balancing experiment.
- [ ] Run `python losses.py` after each loss change.

## Phase 6 - Augmentation and Preprocessing Audit

- [x] Landmark five-point alignment noted as done in previous todo.
- [x] Landmark augmentation noted as done in previous todo.
- [x] Head-pose loose crop/augmentation noted as done in previous todo.
- [x] Attribute augmentation noted as done in previous todo.
- [x] Age/gender/race augmentation noted as done in previous todo.
- [x] Visibility augmentation noted as done in previous todo.
- [ ] Re-verify the actual code matches those "done" notes after dataset-root cleanup.
- [ ] Confirm landmark horizontal flip swaps the standard 68-point symmetric indices.
- [ ] Confirm segmentation masks receive identical spatial transforms as images.
- [ ] Confirm head-pose augmentation does not corrupt pose labels.
- [ ] Add expression augmentation pipeline.
- [ ] Add face-recognition alignment pipeline.

## Phase 7 - Balanced Sampling and Staged Training

- [ ] Update `TASK_ID_MAP` everywhere:
  - [ ] segmentation = 0
  - [ ] landmark = 1
  - [ ] headpose = 2
  - [ ] attribute = 3
  - [ ] age = 4
  - [ ] gender = 5
  - [ ] race = 6
  - [ ] visibility = 7
  - [ ] expression = 8
  - [ ] face_recognition = 9
- [ ] Update `multi_task_collate_fn` dummy labels for expression and face recognition.
- [ ] Add a `--stage` flag to train progressively:
  - [ ] Stage 1: segmentation + landmarks + head pose
  - [ ] Stage 2: add attributes + age/gender/race
  - [ ] Stage 3: add visibility + expression
  - [ ] Stage 4: add face recognition
- [ ] Verify each batch logs balanced per-task counts.
- [ ] Add or update SLURM launch script for 8x A100 DDP.
- [ ] Save checkpoint after every epoch and resume cleanly between stages.

## Phase 8 - Evaluation and Ablations

- [ ] Extend `evaluate.py` for expression recognition.
- [ ] Extend `evaluate.py` for face-recognition verification benchmarks.
- [ ] Report 3-task, 6-task, 8-task, and 10-task staged results.
- [ ] Add ablation flag for bidirectional vs unidirectional cross-attention.
- [ ] Add ablation for balanced vs unbalanced sampling.
- [ ] Add ablation for direct landmark MLP vs hourglass-style landmark head.
- [ ] Add ablation for fixed lambda weights vs gradient-norm balancing.
- [ ] Produce final reproducibility table:
  - [ ] paper claim
  - [ ] released code support
  - [ ] repo implementation
  - [ ] assumption made
  - [ ] sensitivity/risk
  - [ ] measured gap

## Open Decisions to Document

- [ ] Exact 19-class CelebAMask-HQ mapping.
- [ ] Final landmark hourglass implementation details.
- [ ] Expression label convention for AffectNet and RAF-DB.
- [ ] Whether face recognition uses MS1MV3 or available VGGFace2 cache.
- [ ] Whether PartialFC is implemented now or deferred after a standard ArcFace baseline.
- [ ] Initial and final task-loss weights.
- [ ] Age-bin centers for age L1 loss.
- [ ] Learning-rate scaling for staged training vs full 10-task training.

## Immediate Next Steps

- [ ] Fix dataset roots to use the copied local datasets.
- [ ] Run dataset inventory and `python test_setup.py`.
- [ ] Implement Phase 1 token/head changes before adding new tasks.
- [ ] Add expression task after Phase 1 is stable.
- [ ] Resolve face-recognition dataset choice before implementing Phase 4.
