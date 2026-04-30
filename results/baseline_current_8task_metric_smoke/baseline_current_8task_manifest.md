# Current 8-Task Baseline Run Manifest

Generated: 2026-04-27T13:38:33

Scope: current 8-task repo implementation. This is not the full 10-task paper baseline.

Checkpoint: `checkpoints\ckpts\model.pt`

SHA256: `327A755849BA64D336FB96589FF87B27E84A12BE1ECF8BCFAA503D66F803286D`

Dataset root: `D:\Projects\facexformer-main\datasets\datasets`

Result files:

- `results\baseline_current_8task_metric_smoke\gap_analysis_baseline_current_8task.csv`
- `results\baseline_current_8task_metric_smoke\gap_analysis_baseline_current_8task.json`
- `results\baseline_current_8task_metric_smoke\baseline_current_8task_manifest.json`

Metric columns:

- `metric`: original raw script metric.
- `raw_metric`: same as `metric`, explicit name.
- `raw_gap_metric_minus_target`: deprecated mixed-unit gap column, intentionally blank.
- `normalized_metric`: report-ready value.
- `normalized_metric_unit`: unit for normalized metric.
- `normalized_gap_metric_minus_target`: normalized metric minus paper target when the row dataset matches the paper target dataset.

Normalization rules: segmentation/attribute/gender/race/visibility fractions are multiplied by 100; headpose radians are converted to degrees; landmark NME remains percent; age MAE remains years.
