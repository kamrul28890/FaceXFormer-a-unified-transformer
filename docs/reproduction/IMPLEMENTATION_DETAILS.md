# FaceXFormer Reproduction Implementation Details

## Model

The reproduced model keeps the FaceXFormer idea of a shared image encoder, fused multi-scale face representation, task tokens, and a bidirectional FaceX decoder.

Core files:

- `network/models/facexformer.py`
- `network/models/transformer.py`

The implementation uses:

- A Swin-B backbone initialized from ImageNet-pretrained weights.
- An MLP-Fusion module that projects four encoder feature levels into a shared decoder dimension.
- Learnable task tokens for the reproduced tasks.
- A FaceX decoder with task self-attention, task-to-face cross-attention, and face-to-task feedback.
- Lightweight task heads for segmentation, landmarks, pose, attributes, age, gender, race, and visibility.

Known paper-vs-code divergences are documented in the final report:

- The paper describes ten tasks; this repo completes eight.
- The implementation uses a practical task-token/head design rather than every paper-described token count.
- Landmark and head-pose heads differ from the exact paper description.
- Segmentation class mapping uses the reproduced implementation mapping rather than all paper-described classes.

## Data and Preprocessing

Core file:

- `datasets.py`

The data layer implements adapters and preprocessing for:

- CelebAMask-HQ segmentation
- 300W / 300VW landmark evaluation paths
- 300W-LP / BIWI head pose paths
- CelebA and LFWA attributes
- UTKFace and FairFace age/gender/race diagnostics
- COFW visibility

Important behavior:

- Paths are routed through `config.DATASET_ROOT`.
- Multi-task training uses task-aware samples and collate logic.
- Evaluation uses clean resized/aligned inputs, while training applies task-specific augmentation.
- Some datasets are used diagnostically rather than as direct paper-target rows when label taxonomies differ.

## Losses

Core file:

- `losses.py`

Implemented objectives:

- Segmentation: Dice + cross entropy
- Landmark: STARLoss-style coordinate loss
- Head pose: geodesic/Euler-aware pose loss path
- Attributes: binary cross entropy with logits
- Visibility: binary cross entropy with logits
- Age: cross entropy over age bins plus age-value regression term
- Gender/race: cross entropy

The final report records the paper-confirmed `lambda_i = 1.0` task weighting used for the paper-faithful reproduction setting.

## Training

Core files:

- `train.py`
- `config.py`
- `submit_job.slurm`
- `run_torch.sh`

Training support includes:

- Single-GPU and distributed data parallel training.
- Balanced multi-task sampling.
- Upsampled multi-task data balancing.
- Automatic batch-size/GPU configuration.
- Mixed precision support.
- Checkpointing and resume logic.
- Staged 3-to-6-to-8 task co-training strategy.

The final eight-task results in the report are explicitly marked preliminary. The report separates valid/reportable rows from bug-affected or suspect rows.

## Evaluation and Baseline Verification

Core files:

- `evaluate.py`
- `scripts/baseline_current_8task.py`
- `scripts/baseline_verification.py`

The evaluation pipeline produces raw metrics, normalized metrics, paper-target comparisons, and manifests. The key fixed baseline output is:

- `results/baseline_current_8task_fresh_fixed/`

The manifest records metric units and normalization rules so that raw outputs can be compared to paper-reported values without silent unit mismatches.

## Report Generation

Core files:

- `scripts/generate_report_assets.py`
- `report_final/scripts/build_latex_report.py`

Report generation includes:

- Generated architecture and result figures.
- Markdown section drafts in `report_draft/` and `report_final/draft_sources/`.
- A compiled LaTeX PDF at `report_final/main.pdf`.
- Rendered page PNGs at `report_final/rendered_pages/` for visual checking.

The report build was validated by:

- Regenerating LaTeX from Markdown.
- Running `pdflatex` twice.
- Rendering all PDF pages with `pdftoppm`.
- Scanning for placeholder text and inconsistent labels.

## Rebuild Commands

Regenerate report assets:

```powershell
python scripts\generate_report_assets.py
```

Rebuild the final report:

```powershell
cd report_final
python scripts\build_latex_report.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -png -r 150 main.pdf rendered_pages\page
```

Run baseline verification:

```powershell
python scripts\baseline_current_8task.py
```
