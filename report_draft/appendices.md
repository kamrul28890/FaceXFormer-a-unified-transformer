# Appendix A: Repository and Checkpoint Details

**Table A.1: Reproducibility Metadata**

| Field | Value |
| --- | --- |
| Baseline scope | Current eight-task implementation |
| Tasks | Segmentation, landmark, headpose, attribute, age, gender, race, visibility |
| Checkpoint file | `checkpoints/ckpts/model.pt` |
| Checkpoint SHA256 | `327A755849BA64D336FB96589FF87B27E84A12BE1ECF8BCFAA503D66F803286D` |
| Checkpoint size | 1,104,869,851 bytes |
| Checkpoint loadable | Yes (`checkpoint_loaded: true`) |
| Baseline evaluation generated | 2026-04-28T11:38:36 |
| Dataset root | `D:\Projects\facexformer-main\datasets\datasets` |
| Batch size | 64 |
| Seed | 0 |
| Number of result rows | 14 |
| Total evaluated samples | 129,060 |
| Result CSV | `results/baseline_current_8task_fresh_fixed/gap_analysis_baseline_current_8task.csv` |
| Result JSON | `results/baseline_current_8task_fresh_fixed/gap_analysis_baseline_current_8task.json` |
| Manifest | `results/baseline_current_8task_fresh_fixed/baseline_current_8task_manifest.json` |
| Repository | `https://github.com/kamrul28890/FaceXFormer-a-unified-transformer` |

**Normalization rules from manifest.**

| Task group | Rule |
| --- | --- |
| Segmentation | Raw F1 fraction multiplied by 100 gives percent |
| Attribute / Gender / Race | Raw accuracy fraction multiplied by 100 gives percent |
| Visibility | Raw recall fraction multiplied by 100 gives percent |
| Head pose | Raw MAE is already degrees in the fixed manifest |
| Landmark | Raw NME is already percent; normalized-coordinate datasets use sqrt(2) as image diagonal |
| Age | Raw MAE is already years; bucket targets are converted to representative age-bin centers |

# Appendix B: Raw Baseline Metrics

**Table B.1: Full Raw vs. Normalized Baseline Metrics**

| Task | Dataset | Samples | Raw Metric | Normalized Metric | Unit | Paper Target | Normalized Gap |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Segmentation | CelebAMaskHQ | 2,000 | 0.9177 | 91.77 | percent | 92.01 | -0.24 |
| Landmark | 300W full | 689 | 6.7528 | 6.75 | NME percent | 4.67 | +2.08 |
| Landmark | 300W common | 554 | 5.5955 | 5.60 | NME percent | - | - |
| Landmark | 300W challenging | 135 | 11.5021 | 11.50 | NME percent | - | - |
| Head Pose | BIWI | 15,678 | 20.6497 | 20.65 | degrees | 3.52 | +17.13 |
| Attribute | CelebA | 19,962 | 0.9172 | 91.72 | percent | 91.83 | -0.11 |
| Attribute | LFWA | 13,143 | 0.6106 | 61.06 | percent | - | - |
| Age | UTKFace | 3,556 | 1.1714 | 1.17 | years | 4.17 | -3.00 |
| Age | FairFace | 21,908 | 6.2990 | 6.30 | years | - | - |
| Gender | UTKFace | 3,556 | 0.9918 | 99.18 | percent | - | - |
| Gender | FairFace | 21,908 | 0.9311 | 93.11 | percent | - | - |
| Race | UTKFace | 3,556 | 0.9803 | 98.03 | percent | - | - |
| Race | FairFace | 21,908 | 0.6459 | 64.59 | percent | - | - |
| Visibility | COFW | 507 | 0.8713 | 87.13 | percent | 72.56 | +14.57 |

*Source: `results/baseline_current_8task_fresh_fixed/gap_analysis_baseline_current_8task.json`.*

# Appendix C: Figure List

| Figure | Title | Source / Generation |
| --- | --- | --- |
| Fig. 3.1 | FaceXFormer end-to-end pipeline | `report_assets/fig1_facexformer_pipeline.pdf`; redrawn from paper architecture description |
| Fig. 3.2 | FaceX decoder block detail | `report_assets/fig2_facex_decoder_block.pdf`; drawn from TSA, TFCA, FTCA decoder description |
| Fig. 4.1 | Gap analysis summary heatmap | `report_assets/fig8_gap_analysis_heatmap.pdf`; generated from Table 4.1 impact labels |
| Fig. 4.2 | Staged training strategy timeline | `report_assets/fig3_staged_training_timeline.pdf`; generated from staged 3-to-6-to-8 task schedule |
| Fig. 5.1 | Baseline inference results bar chart | `report_assets/fig4_baseline_inference_bars.pdf`; generated from fixed baseline JSON |
| Fig. 6.1 | Preliminary training results with flagged rows | `report_assets/fig5_training_results_flagged.pdf`; generated from report instruction/final presentation values |
| Fig. 8.1 | Loss scale comparison across tasks | `report_assets/fig7_loss_scale_comparison.pdf`; illustrative loss-scale comparison |

The final presentation's embedded architecture image was also extracted for reference at `report_assets/pptx_media/image1.png`.

# Appendix D: Supplementary Implementation Notes

1. **STARLoss implementation.** The reproduced landmark loss is STARLoss-style and is applied to the MLP output coordinates. This differs from the paper's described 68-token hourglass landmark head.

2. **Head pose representation.** The paper describes 9 tokens representing a 3x3 rotation matrix. The reproduced implementation predicts three Euler-angle values from one pose token and converts them to rotation matrices for geodesic-loss computation where needed.

3. **Segmentation class mapping.** The paper states that segmentation uses one token per class, and the appendix mentions 19 CelebAMask-HQ classes. The reproduced implementation uses 11 segmentation/mask tokens. This is treated as a class-mapping ambiguity and architectural divergence.

4. **Age bins.** The reproduced implementation uses eight age bins with representative age-bin centers. This differs from the outline's earlier 11-bin assumption and should be treated as an implementation decision where the paper is underspecified.

5. **FairFace race mapping.** FairFace provides seven race categories, while the reproduced model head uses a five-class race output. FairFace race evaluation is therefore diagnostic rather than directly paper-comparable.

6. **COFW Recall@P80.** Visibility evaluation is sensitive to label polarity, class balance, precision-recall interpolation, and threshold selection. The COFW row is marked suspect until the exact paper protocol is verified.

7. **Loss coefficients.** The reported paper-faithful reproduction uses `lambda_i = 1.0` for all reproduced tasks based on author email confirmation. The paper and public release do not document these coefficients.

8. **Training results provenance.** The preliminary training metrics in Section 6 are taken from the report instruction/final presentation source. The raw training screenshot or log file was not present in the current repository, so those rows should be replaced once bug-fixed training logs are available.
