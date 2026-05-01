const fs = require("fs");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const inputPath = path.join(repoRoot, "report_v2.txt");
const outputPath = path.join(repoRoot, "FaceXFormer_Report_v2_clean.tex");

const AlignmentType = { LEFT: "left", CENTER: "center", JUSTIFIED: "justify" };
const HeadingLevel = { HEADING_1: 1, HEADING_2: 2, HEADING_3: 3 };
const BorderStyle = { SINGLE: "single", NONE: "none" };
const WidthType = { DXA: "dxa" };
const ShadingType = { CLEAR: "clear" };
const VerticalAlign = { CENTER: "center" };
const PageNumber = { CURRENT: "" };
const LevelFormat = { BULLET: "bullet", DECIMAL: "decimal" };
const UnderlineType = { SINGLE: "single" };

class TextRun {
  constructor(opts = {}) {
    this.type = "run";
    Object.assign(this, opts);
    if (this.text === undefined && Array.isArray(this.children)) {
      this.text = this.children.join("");
    }
  }
}

class Paragraph {
  constructor(opts = {}) {
    this.type = "paragraph";
    Object.assign(this, opts);
  }
}

class Table {
  constructor(opts = {}) {
    this.type = "table";
    Object.assign(this, opts);
  }
}

class TableRow {
  constructor(opts = {}) {
    this.type = "row";
    Object.assign(this, opts);
  }
}

class TableCell {
  constructor(opts = {}) {
    this.type = "cell";
    Object.assign(this, opts);
  }
}

class Header { constructor(opts = {}) { Object.assign(this, opts); } }
class Footer { constructor(opts = {}) { Object.assign(this, opts); } }
class Document { constructor(opts = {}) { Object.assign(this, opts); } }
class ExternalHyperlink { constructor(opts = {}) { Object.assign(this, opts); } }
class TableOfContents { constructor(opts = {}) { Object.assign(this, opts); } }
class PageBreak { constructor() { this.type = "pagebreak"; } }

const Packer = { toBuffer: async () => Buffer.from("") };

function cleanText(value) {
  if (value === undefined || value === null) return "";
  let text = String(value);
  const replacements = [
    ["—", " -- "],
    ["–", "-"],
    ["−", "-"],
    ["→", "->"],
    ["↑", "up"],
    ["↓", "down"],
    ["≈", "approximately"],
    ["≤", "<="],
    ["≥", ">="],
    ["✓", "yes"],
    ["°", " degrees"],
    ["±", "+/-"],
    ["·", "."],
    ["×", "x"],
    ["π", "pi"],
    ["λᵢ", "lambda_i"],
    ["λ", "lambda"],
    ["Σ", "sum"],
    ["ᵢ", "_i"],
    ["₁", "_1"],
    ["ₙ", "_n"],
    ["⁴", "^4"],
    ["ℝ", "R"],
    ["∈", "in"],
    ["…", "..."],
    ["⟨", "<"],
    ["⟩", ">"],
    ["Δ", "Delta"],
    ["•", "-"],
    ["“", "\""],
    ["”", "\""],
    ["‘", "'"],
    ["’", "'"],
    ["â€”", " -- "],
    ["â€“", "-"],
    ["âˆ’", "-"],
    ["â†’", "->"],
    ["â†‘", "up"],
    ["â†“", "down"],
    ["â‰ˆ", "approximately"],
    ["â‰¤", "<="],
    ["â‰¥", ">="],
    ["âœ“", "yes"],
    ["Â°", " degrees"],
    ["Â±", "+/-"],
    ["Â·", "."],
    ["Ã—", "x"],
    ["Ï€", "pi"],
    ["Î»áµ¢", "lambda_i"],
    ["Î»", "lambda"],
    ["Î£", "sum"],
    ["áµ¢", "_i"],
    ["â‚", "_1"],
    ["â‚™", "_n"],
    ["â´", "^4"],
    ["â„", "R"],
    ["âŸ¨", "<"],
    ["âŸ©", ">"],
    ["FÌ‚", "F-hat"],
    ["TÌ‚", "T-hat"],
    ["Ì‚", "-hat"],
    ["Dâ‚œ", "D_t"],
    ["â‚œ", "_t"],
    ["Táµ¢", "T_i"],
    ["Sáµ¢", "S_i"],
    ["Láµ¢", "L_i"],
    ["Î”", "Delta"],
    ["Î³", "gamma"],
    ["â€¢", "-"],
    ["â€˜", "'"],
    ["â€™", "'"],
    ["â€œ", "\""],
    ["â€", "\""],
    ["â€", "\""],
    ["Ã©", "e"],
    ["â»", "-"],
  ];
  for (const [from, to] of replacements) text = text.split(from).join(to);
  text = text.replace(/Â/g, "");
  text = text.replace(/Ã—/g, "x");
  text = text.replace(/[^\x00-\x7F]/g, "");
  text = text.replace(/\s+/g, " ").trim();
  text = text.replace(/\s+([,.;:])/g, "$1");
  text = text.replace(/\(\s+/g, "(").replace(/\s+\)/g, ")");
  return text;
}

function latexEscape(value) {
  let text = cleanText(value);
  const specials = [
    [/\\/g, "\\textbackslash{}"],
    [/&/g, "\\&"],
    [/%/g, "\\%"],
    [/\$/g, "\\$"],
    [/#/g, "\\#"],
    [/_/g, "\\_"],
    [/{/g, "\\{"],
    [/}/g, "\\}"],
    [/\^/g, "\\textasciicircum{}"],
    [/~/g, "\\textasciitilde{}"],
  ];
  for (const [pattern, replacement] of specials) text = text.replace(pattern, replacement);
  text = text.replace(/\b([A-Fa-f0-9]{40,})\b/g, (_match, hash) => `\\seqsplit{${hash}}`);
  return text;
}

function itemText(node) {
  if (!node) return "";
  if (typeof node === "string") return node;
  if (node.type === "run") return node.text || "";
  if (node.type === "paragraph") return (node.children || []).map(itemText).join("");
  if (node.type === "cell") return (node.children || []).map(itemText).join(" ");
  return "";
}

function cellText(cell) {
  return latexEscape(itemText(cell));
}

function paraText(paragraph) {
  return latexEscape(itemText(paragraph));
}

function parseTableCaption(text) {
  const cleaned = cleanText(text);
  const match = cleaned.match(/^(Table\s+[A-Za-z0-9.]+):?\s*(.*)$/i);
  if (match) return { label: match[1], caption: match[2] || cleaned };
  return { label: "", caption: cleaned };
}

function renderTable(table, pendingCaption) {
  const rows = table.rows || [];
  const columnCount = Math.max(...rows.map((row) => (row.children || []).length), 1);
  const colSpec = ">{\\RaggedRight\\arraybackslash}X".repeat(columnCount);
  const caption = parseTableCaption(pendingCaption || "Table");
  const out = [];
  out.push("\\begin{table}[H]");
  out.push("\\centering");
  out.push("\\scriptsize");
  out.push("\\setlength{\\tabcolsep}{3pt}");
  out.push("\\renewcommand{\\arraystretch}{1.15}");
  out.push(`\\caption{${latexEscape(caption.caption)}}`);
  out.push(`\\begin{tabularx}{\\textwidth}{${colSpec}}`);
  out.push("\\toprule");
  rows.forEach((row, index) => {
    const cells = (row.children || []).map(cellText);
    while (cells.length < columnCount) cells.push("");
    out.push(cells.join(" & ") + " \\\\");
    if (index === 0) out.push("\\midrule");
  });
  out.push("\\bottomrule");
  out.push("\\end{tabularx}");
  out.push("\\end{table}");
  return out.join("\n");
}

const figureMap = new Map([
  ["3.1 overall pipeline", ["assets/fig1_facexformer_pipeline.pdf", "FaceXFormer end-to-end pipeline."]],
  ["3.4 facex decoder", ["assets/fig2_facex_decoder_block.pdf", "FaceX decoder block with task self-attention, task-to-face cross-attention, and face-to-task cross-attention."]],
  ["4.1 paper-vs-code gap analysis", ["assets/fig8_gap_analysis_heatmap.pdf", "Structured gap analysis between the paper specification, released code, and this reproduction."]],
  ["4.6 staged training strategy", ["assets/fig3_staged_training_timeline.pdf", "Three-stage 3-to-6-to-8 task co-training strategy."]],
  ["5.1 results", ["assets/fig4_baseline_inference_bars.pdf", "Released-checkpoint baseline inference compared with paper targets."]],
  ["6.1 main results", ["assets/fig5_training_results_flagged.pdf", "Eight-task training results with known gap rows flagged."]],
  ["8.3 the lambda=1 problem", ["assets/fig7_loss_scale_comparison.pdf", "Illustrative loss-scale comparison showing why uniform task weights are unstable."]],
]);

function renderFigure(sectionTitle) {
  const key = cleanText(sectionTitle).toLowerCase();
  const fig = figureMap.get(key);
  if (!fig) return "";
  const [file, caption] = fig;
  return [
    "\\begin{figure}[H]",
    "\\centering",
    `\\includegraphics[width=0.94\\textwidth]{${file}}`,
    `\\caption{${latexEscape(caption)}}`,
    "\\end{figure}",
  ].join("\n");
}

function extractDoc() {
  let code = fs.readFileSync(inputPath, "utf8");
  code = code.replace(/^\s*javascript\s*\r?\n/, "");
  code = code.replace(/const\s*\{[\s\S]*?\}\s*=\s*require\('docx'\);\s*const fs = require\('fs'\);\s*/, "");
  code = code.replace(/Packer\.toBuffer\(doc\)[\s\S]*$/m, "globalThis.__doc = doc;");
  eval(code);
  return globalThis.__doc;
}

function isPageBreak(paragraph) {
  return paragraph && paragraph.type === "paragraph" && (paragraph.children || []).some((child) => child && child.type === "pagebreak");
}

function preamble() {
  return String.raw`\documentclass[11pt]{article}
\usepackage[letterpaper,margin=0.82in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{float}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{seqsplit}
\usepackage[most]{tcolorbox}

\definecolor{facexblue}{HTML}{0D1B2A}
\definecolor{facexteal}{HTML}{0E7C7B}
\definecolor{facexgray}{HTML}{334155}
\definecolor{facexlight}{HTML}{F0F7FA}

\titleformat{\section}{\Large\bfseries\color{facexblue}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries\color{facexteal}}{\thesubsection}{0.65em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{facexblue}}{\thesubsubsection}{0.55em}{}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.48em}
\setlength{\parindent}{0pt}
\setlist[itemize]{leftmargin=1.4em,itemsep=0.15em,topsep=0.25em}
\onehalfspacing

\pagestyle{fancy}
\fancyhf{}
\lhead{\small Reproducing FaceXFormer}
\rhead{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\newcolumntype{Y}{>{\RaggedRight\arraybackslash}X}
\newenvironment{keybox}[1]{\begin{tcolorbox}[colback=facexlight,colframe=facexteal,title=#1,fonttitle=\bfseries,arc=1mm,boxrule=0.5pt]}{\end{tcolorbox}}

\begin{document}
\hypersetup{pageanchor=false}
\pagenumbering{roman}
\begin{titlepage}
\centering
\vspace*{1.0in}
{\Huge\bfseries\color{facexblue} Reproducing FaceXFormer\par}
\vspace{0.25in}
{\LARGE\bfseries\color{facexteal} A Unified Transformer for Multi-Task Facial Analysis\par}
\vspace{0.35in}
\rule{0.78\textwidth}{1.2pt}
\vspace{0.4in}

{\Large\bfseries Preetom Saha Arko \quad Md Kamruzzaman Kamrul\par}
\vspace{0.1in}
{\large Purdue University\par}
\vspace{0.1in}
{\normalsize arko@purdue.edu \quad mkamrul@purdue.edu\par}
\vfill

\begin{keybox}{Report Scope}
\begin{tabularx}{0.92\textwidth}{>{\bfseries}p{0.28\textwidth}Y}
Original paper & Narayan et al., ICCV 2025 -- Johns Hopkins University\\
Task scope & 8 of 10 tasks reproduced; expression and face recognition excluded\\
Test samples & 127,735 samples across 12 dataset-task combinations\\
Training infrastructure & 8 x NVIDIA A100 GPUs; PyTorch DDP; FP16; Purdue cluster\\
Epochs & 12 epochs; AdamW; learning rate 1e-4; decay at epochs 6 and 10\\
Repository & github.com/kamrul28890/FaceXFormer-a-unified-transformer\\
\end{tabularx}
\end{keybox}
\vfill
{\large May 2026\par}
\end{titlepage}
\hypersetup{pageanchor=true}

\tableofcontents
\newpage
\pagenumbering{arabic}
`;
}

function postamble() {
  return "\n\\end{document}\n";
}

function renderContent(children) {
  const out = [];
  let pendingCaption = "";
  let inReferences = false;
  let referencesOpen = false;

  function closeReferences() {
    if (referencesOpen) {
      out.push("\\end{enumerate}");
      referencesOpen = false;
    }
  }

  let started = false;
  for (const child of children) {
    if (!started) {
      if (isPageBreak(child)) started = true;
      continue;
    }
    if (isPageBreak(child)) continue;
    if (!child) continue;

    if (child.type === "paragraph") {
      const text = cleanText(itemText(child));
      if (!text) continue;

      if (child.heading === HeadingLevel.HEADING_1) {
        closeReferences();
        pendingCaption = "";
        inReferences = /^references$/i.test(text);
        out.push("");
        if (/^abstract$/i.test(text)) out.push("\\section*{Abstract}");
        else if (inReferences) out.push("\\section*{References}");
        else out.push(`\\section{${latexEscape(text.replace(/^\d+\.\s*/, ""))}}`);
        continue;
      }

      if (child.heading === HeadingLevel.HEADING_2) {
        closeReferences();
        pendingCaption = "";
        out.push("");
        out.push(`\\subsection{${latexEscape(text.replace(/^\d+\.\d+\s*/, ""))}}`);
        const fig = renderFigure(text);
        if (fig) out.push(fig);
        continue;
      }

      if (child.heading === HeadingLevel.HEADING_3) {
        closeReferences();
        out.push("");
        if (/^[A-Z]\.\d+/.test(text)) {
          out.push(`\\subsection{${latexEscape(text.replace(/^[A-Z]\.\d+\s*/, ""))}}`);
        } else {
          out.push(`\\subsubsection{${latexEscape(text.replace(/^[A-Z]?\d*(\.\d+)*\s*/, ""))}}`);
        }
        continue;
      }

      if (/^Table\s+[A-Za-z0-9.]+:/i.test(text) || /^Greyed rows indicate/i.test(text) || /^Impact:/i.test(text) || /^Without these normalizations/i.test(text) || /^\*/.test(text)) {
        pendingCaption = pendingCaption || text;
        if (!/^Table\s+/i.test(text)) out.push(`\\textit{${latexEscape(text)}}`);
        continue;
      }

      if (inReferences) {
        if (!referencesOpen) {
          out.push("\\begin{enumerate}[leftmargin=1.5em,itemsep=0.25em]");
          referencesOpen = true;
        }
        out.push(`\\item ${latexEscape(text)}`);
        continue;
      }

      if (child.numbering) {
        out.push(`\\begin{itemize}\\item ${latexEscape(text)}\\end{itemize}`);
      } else if (child.shading || child.border) {
        out.push(`\\begin{keybox}{Note}${latexEscape(text)}\\end{keybox}`);
      } else {
        out.push(latexEscape(text));
      }
      continue;
    }

    if (child.type === "table") {
      closeReferences();
      out.push(renderTable(child, pendingCaption));
      pendingCaption = "";
      continue;
    }
  }
  closeReferences();
  return out.join("\n\n");
}

const doc = extractDoc();
const children = doc.sections[0].children;
const tex = preamble() + renderContent(children) + postamble();
fs.writeFileSync(outputPath, tex, "utf8");
console.log(outputPath);
