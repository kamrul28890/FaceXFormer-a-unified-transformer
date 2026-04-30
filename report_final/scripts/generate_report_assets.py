from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets"
BASELINE_JSON = ROOT / "results" / "baseline_current_8task_fresh_fixed" / "gap_analysis_baseline_current_8task.json"


COLORS = {
    "paper": "#3f6ea8",
    "ours": "#d95f4d",
    "accent": "#2c7a7b",
    "soft": "#e8edf3",
    "line": "#263238",
    "critical": "#b91c1c",
    "high": "#f97316",
    "medium": "#facc15",
    "none": "#22c55e",
    "pending": "#9ca3af",
}


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, fc="#ffffff", ec=COLORS["line"], fs=9, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def arrow(ax, p1, p2, color=COLORS["line"], rad=0.0, lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box(ax, (0.25, 2.0), (1.25, 1.0), "224x224\nRGB image", fc="#f8fafc", weight="bold")
    box(ax, (2.0, 1.55), (1.65, 1.9), "Swin-B\nencoder\nImageNet init", fc="#eef6ff", weight="bold")
    for i, s in enumerate(["s4", "s8", "s16", "s32"]):
        y = 0.35 + i * 0.48
        box(ax, (3.95, y), (1.15, 0.33), f"feature {s}", fc="#f8fafc", fs=7)
        arrow(ax, (3.65, 2.0 + (i - 1.5) * 0.28), (3.95, y + 0.17), lw=0.8)

    box(ax, (5.55, 1.55), (1.45, 1.9), "MLP-Fusion\n983K params\nunified F", fc="#ecfdf5", weight="bold")
    box(ax, (7.55, 1.55), (1.65, 1.9), "FaceX Decoder\nN = 2 blocks\nTSA + TFCA + FTCA", fc="#fff7ed", weight="bold")
    box(ax, (9.75, 1.55), (1.35, 1.9), "Unified\nhead\nfinal TFCA", fc="#fefce8", weight="bold")
    task_head_text = "task heads\n\nsegmentation\nlandmark\nhead pose\nattribute\nage/gender/race\nvisibility\nexpression*\nrecognition*"
    box(ax, (11.35, 0.45), (1.35, 4.1), task_head_text, fc="#f8fafc", fs=6, weight="normal")

    arrow(ax, (1.5, 2.5), (2.0, 2.5))
    arrow(ax, (5.1, 1.85), (5.55, 2.1))
    arrow(ax, (5.1, 2.75), (5.55, 2.9))
    arrow(ax, (7.0, 2.5), (7.55, 2.5))
    arrow(ax, (9.2, 2.5), (9.75, 2.5))
    arrow(ax, (11.1, 2.5), (11.35, 2.5))

    ax.text(6.05, 4.55, "FaceXFormer end-to-end pipeline", ha="center", fontsize=13, weight="bold")
    ax.text(6.05, 0.12, "* expression and recognition are paper tasks, but outside the 8-task reproduction scope", ha="center", fontsize=8)
    save(fig, "fig1_facexformer_pipeline")


def fig2_decoder():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    box(ax, (0.45, 3.55), (1.45, 0.8), "task tokens\nT", fc="#fefce8", weight="bold")
    box(ax, (0.45, 0.85), (1.45, 0.8), "face tokens\nF", fc="#ecfdf5", weight="bold")
    box(ax, (2.65, 3.35), (1.55, 1.2), "TSA\nT attends to T", fc="#fff7ed", weight="bold")
    box(ax, (4.85, 2.35), (1.75, 1.2), "TFCA\nT queries F", fc="#eef6ff", weight="bold")
    box(ax, (7.2, 1.35), (1.85, 1.2), "FTCA\nF queries T'", fc="#fdecec", weight="bold")
    box(ax, (8.25, 3.6), (1.2, 0.65), "updated\nT'", fc="#fefce8", weight="bold")
    box(ax, (8.25, 0.55), (1.2, 0.65), "refined\nF hat", fc="#ecfdf5", weight="bold")

    arrow(ax, (1.9, 3.95), (2.65, 3.95))
    arrow(ax, (4.2, 3.95), (4.85, 3.05))
    arrow(ax, (1.9, 1.25), (4.85, 2.75), rad=0.18)
    arrow(ax, (6.6, 2.95), (7.2, 2.05))
    arrow(ax, (6.6, 2.95), (8.25, 3.92), rad=-0.15)
    arrow(ax, (1.9, 1.25), (7.2, 1.95), rad=-0.1)
    arrow(ax, (9.05, 1.35), (8.86, 1.2))

    ax.text(5.0, 4.85, "FaceX Decoder block: bidirectional task-face exchange", ha="center", fontsize=13, weight="bold")
    ax.text(5.0, 0.15, "Standard task-to-face cross-attention updates only task tokens; FaceXFormer adds face-to-task feedback.", ha="center", fontsize=8)
    save(fig, "fig2_facex_decoder_block")


def fig3_timeline():
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    stages = [
        ("Stage 1", "3 tasks", "segmentation\nlandmark\nhead pose"),
        ("Stage 2", "6 tasks", "+ attribute\n+ age\n+ gender"),
        ("Stage 3", "8 tasks", "+ race\n+ visibility\nfull reproduction scope"),
    ]
    xs = [0.6, 3.95, 7.3]
    for i, (name, count, lines) in enumerate(stages):
        box(ax, (xs[i], 1.0), (2.45, 1.65), f"{name}\n{count}\n{lines}", fc=["#eef6ff", "#ecfdf5", "#fff7ed"][i], weight="bold")
        if i < 2:
            arrow(ax, (xs[i] + 2.45, 1.83), (xs[i + 1], 1.83), lw=1.6)
    ax.text(5.25, 3.35, "Staged 3-to-6-to-8 task training strategy", ha="center", fontsize=13, weight="bold")
    ax.text(5.25, 0.35, "Report instruction source: staged co-training on Purdue cluster, 8xA100, 12 epochs.", ha="center", fontsize=8)
    save(fig, "fig3_staged_training_timeline")


def load_baseline_rows():
    data = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r.get("paper_target") is not None]
    order = ["segmentation", "landmark", "headpose", "attribute", "age", "visibility"]
    rows.sort(key=lambda r: order.index(r["task"]) if r["task"] in order else 99)
    return rows


def fig4_baseline():
    rows = load_baseline_rows()
    labels = [
        "Seg.\nF1 %",
        "Landmark\nNME %",
        "Head pose\nMAE deg",
        "Attr.\nAcc %",
        "Age\nMAE yr",
        "Vis.\nR@P80 %",
    ]
    paper = [r["paper_target"] for r in rows]
    ours = [r["normalized_metric"] for r in rows]
    x = np.arange(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, paper, width, label="paper target", color=COLORS["paper"])
    ax.bar(x + width / 2, ours, width, label="released checkpoint", color=COLORS["ours"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("reported metric value")
    ax.set_title("Baseline inference: paper target vs. normalized released-checkpoint value", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.text(0.5, -0.22, "Lower is better for landmark NME, head-pose MAE, and age MAE; higher is better for F1, accuracy, and Recall@P80.", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, "fig4_baseline_inference_bars")


def fig5_training_results():
    labels = ["Seg.\nF1 %", "Landmark\nNME %", "Head pose\nMAE deg", "Attr.\nAcc %", "Age\nMAE yr", "Gender\nAcc %", "Vis.\nR@P80 %"]
    paper = [92.01, 4.67, 3.52, 91.83, 4.17, 95.22, 72.56]
    ours = [85.91, 0.0117, 0.379, 91.27, 35.27, 100.0, 99.25]
    status = ["valid", "bug", "bug", "valid", "bug", "suspect", "suspect"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 4.9))
    ax.bar(x - width / 2, paper, width, label="paper target", color=COLORS["paper"])
    bars = ax.bar(x + width / 2, ours, width, label="trained reproduction", color=COLORS["ours"])
    for bar, st in zip(bars, status):
        if st == "bug":
            bar.set_hatch("///")
            bar.set_edgecolor("#111827")
        elif st == "suspect":
            bar.set_hatch("...")
            bar.set_edgecolor("#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("reported metric value")
    ax.set_title("8-task training results: valid rows plus flagged bug/suspect rows", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    for barset in ax.containers:
        ax.bar_label(barset, fmt="%.2f", fontsize=7, padding=2)
    ax.text(0.5, -0.24, "Hatched bars are not final comparisons: diagonal = known training metric bug; dotted = suspect metric/protocol issue.", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, "fig5_training_results_flagged")


def fig7_loss_scale():
    labels = ["Dice loss\nsegmentation", "BCE\nattribute/visibility", "CE\nclassification", "L1 age\n(years)"]
    values = [1.0, math.log(2), 2.0, 50.0]
    colors = ["#3f6ea8", "#2c7a7b", "#8b5cf6", "#d95f4d"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("approximate loss scale (log axis)")
    ax.set_title("Why lambda_i = 1 can imbalance multi-task training", weight="bold")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.bar_label(bars, labels=["~1", "~0.69", "~2", "~50"], fontsize=8, padding=3)
    ax.text(0.5, -0.22, "Illustrative scale comparison: age L1 can be tens of units while segmentation and BCE terms are bounded near 1.", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, "fig7_loss_scale_comparison")


def fig8_gap_heatmap():
    components = [
        "scope",
        "tokens",
        "landmark\nhead",
        "lambda_i",
        "training\nloop",
        "loaders",
        "sampler",
        "epochs",
        "augment.",
        "seg tokens",
        "expression",
        "recognition",
    ]
    impact = ["HIGH", "HIGH", "MED", "HIGH", "CRITICAL", "CRITICAL", "HIGH", "MED", "MED", "NONE", "HIGH", "HIGH"]
    impact_to_num = {"NONE": 0, "MED": 1, "HIGH": 2, "CRITICAL": 3}
    nums = np.array([[impact_to_num[i] for i in impact]])
    cmap = plt.matplotlib.colors.ListedColormap([COLORS["none"], COLORS["medium"], COLORS["high"], COLORS["critical"]])

    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.imshow(nums, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(components)))
    ax.set_xticklabels(components, fontsize=8)
    ax.set_title("Gap analysis summary heatmap", weight="bold", pad=14)
    for j, label in enumerate(impact):
        ax.text(j, 0, label, ha="center", va="center", color="white" if label in {"HIGH", "CRITICAL"} else "#111827", fontsize=8, weight="bold")
    ax.set_frame_on(False)
    save(fig, "fig8_gap_analysis_heatmap")


def write_manifest():
    manifest = """# Report Asset Manifest

Generated by `scripts/generate_report_assets.py`.

## Figures

| Figure | File stem | Source |
| --- | --- | --- |
| Fig. 1 | `fig1_facexformer_pipeline` | Redrawn from paper architecture description and original Figure 1 concept. |
| Fig. 2 | `fig2_facex_decoder_block` | Drawn from FaceX decoder description: TSA, TFCA, FTCA. |
| Fig. 3 | `fig3_staged_training_timeline` | Report instructions: staged 3-to-6-to-8 task training. |
| Fig. 4 | `fig4_baseline_inference_bars` | `results/baseline_current_8task_fresh_fixed/gap_analysis_baseline_current_8task.json`. |
| Fig. 5 | `fig5_training_results_flagged` | Report instructions and final presentation slide 8. |
| Fig. 6 | `fig7_loss_scale_comparison` | Illustrative loss-scale analysis from report instructions. |
| Fig. 7 | `fig8_gap_analysis_heatmap` | Report instructions Table 4.1. |

Each figure is exported as `.png` and `.pdf`.

The final PPTX embedded source image was also extracted to
`report_assets/pptx_media/image1.png` for reference.

## Cross-check notes

- Use 12 epochs in the report unless newer run metadata says otherwise. The final deck contains an inconsistent 15-epoch statement on the implementation slide, while the paper, report instructions, and training-results slide state 12 epochs.
- For 300W landmark reporting, distinguish 300W full NME 4.67 from 300W common NME 3.05. The baseline slide appears to mix these targets.
- The fresh fixed baseline manifest reports 129,060 total samples and 14 rows because it includes 300W common/challenging rows. The report instructions use 127,735 samples across 12 dataset-task combinations. Use the report value for the scoped report unless the appendix explicitly includes the extra 300W subset rows.
- The training screenshot/source file for final 8-task training metrics is not present in the current repository. Values used in Fig. 5 come from `report_writing_instructions.md` and the final PPTX text extraction.
- Some typography in `report_writing_instructions.md` is mojibake-corrupted. Normalize punctuation and symbols during drafting.
"""
    (OUT / "asset_manifest.md").write_text(manifest, encoding="utf-8")


def main():
    ensure_out()
    fig1_pipeline()
    fig2_decoder()
    fig3_timeline()
    fig4_baseline()
    fig5_training_results()
    fig7_loss_scale()
    fig8_gap_heatmap()
    write_manifest()


if __name__ == "__main__":
    main()
