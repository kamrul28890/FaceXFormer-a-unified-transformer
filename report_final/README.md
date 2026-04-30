# FaceXFormer Reproduction Report Package

This folder contains the polished LaTeX report package and the rendered final PDF.

## Key Files

- `main.pdf` - compiled final report.
- `main.tex` - generated LaTeX source for the full report.
- `assets/` - report figures in PDF/PNG form plus extracted presentation media.
- `draft_sources/` - section-level Markdown sources used to compose the LaTeX file.
- `source_materials/` - original paper, proposal, final presentation, and report-writing instructions.
- `scripts/` - code used to generate figures and compose the LaTeX document.
- `rendered_pages/` - PNG renders of the compiled PDF used for visual inspection.

## Rebuild

From this folder:

```powershell
python scripts\build_latex_report.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Optional visual render:

```powershell
pdftoppm -png -r 150 main.pdf rendered_pages\page
```

## Scope Note

The composed report intentionally omits ablation studies because they were not completed. The original instruction file is preserved in `source_materials/` for traceability, including any earlier inconsistent notes.
