# FaceXFormer Reproduction Artifact Manifest

## Final Report Package

| Path | Description |
| --- | --- |
| `report_final/main.pdf` | Final compiled reproduction report. |
| `report_final/main.tex` | Generated LaTeX source for the report. |
| `report_final/assets/` | PDF/PNG figures used by the LaTeX report. |
| `report_final/draft_sources/` | Section-level Markdown sources. |
| `report_final/source_materials/` | Paper, proposal, final presentation, and writing instructions. |
| `report_final/scripts/` | Report-specific build and asset scripts. |
| `report_final/rendered_pages/` | Rendered PDF pages used for visual inspection. |

## Draft and Standalone Report Assets

| Path | Description |
| --- | --- |
| `report_draft/` | Working section drafts created before LaTeX composition. |
| `report_assets/` | Standalone generated figures and extracted presentation media. |
| `report_writing_instructions.md` | Original detailed report-writing instructions. |
| `FaceXFormer_Reproduction_v3 2.pptx` | Final presentation source used for report alignment. |
| `FaceXFormer_Reproduction_backup_20260427_211942.pptx` | Backup presentation artifact. |
| `FaceXFormer2403.12960v3.pdf` | Original FaceXFormer paper. |
| `Reproducing_FaceXFormer__A_Unified_Transformerfor_Multi_Task_Facial_Analysis.pdf` | Project proposal. |

## Code

| Path | Description |
| --- | --- |
| `network/models/facexformer.py` | Reproduced FaceXFormer model. |
| `network/models/transformer.py` | FaceX decoder / transformer logic. |
| `datasets.py` | Dataset adapters and preprocessing. |
| `losses.py` | Task losses. |
| `train.py` | Main training script. |
| `evaluate.py` | Evaluation script. |
| `config.py` | Configuration and paths. |
| `scripts/` | Baseline, report asset, and diagnostic scripts. |

## Results

| Path | Description |
| --- | --- |
| `results/baseline_current_8task_fresh_fixed/` | Primary fixed baseline verification output used by the report. |
| `results/baseline_current_8task_*` | Intermediate metric, synchronization, and diagnostic baseline outputs. |
| `results/baseline_landmark_300w_*` | Landmark-specific diagnostic outputs. |
| `results/baseline_fairface_*` | FairFace taxonomy/label diagnostic outputs. |
| `results/baseline_segmentation_*` | Segmentation mapping/token-order diagnostic outputs. |
| `results/ablation_tiny/` | Tiny diagnostic training artifact retained for traceability; not used as a final ablation study in the report. |

## Large Files

The following binary artifacts are tracked through Git LFS:

| Path | Approx. size | Description |
| --- | ---: | --- |
| `checkpoints/ckpts/model.pt` | 1.05 GB | Main FaceXFormer checkpoint artifact. |
| `results/ablation_tiny/full/tiny_checkpoint_last.pth` | 352 MB | Tiny diagnostic checkpoint artifact. |

## Excluded Local State

The repository does not include:

- `.venv/`, `env/`, `venv/`, or other local runtime environments.
- `datasets/` and raw dataset files.
- Python cache folders such as `__pycache__/`.
- IDE-local state and machine-local environment variables.
