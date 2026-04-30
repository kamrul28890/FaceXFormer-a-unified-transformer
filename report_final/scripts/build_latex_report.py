from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DRAFT = ROOT / "draft_sources"
OUT = ROOT / "main.tex"


ORDER = [
    "section_01_introduction.md",
    "section_02_related_work.md",
    "section_03_architecture.md",
    "section_04_methodology.md",
    "section_05_baseline_verification.md",
    "section_06_training_results.md",
    "section_08_discussion.md",
    "section_09_conclusion.md",
]


FIGURE_WIDTHS = {
    "fig1_facexformer_pipeline.pdf": "0.98\\textwidth",
    "fig2_facex_decoder_block.pdf": "0.86\\textwidth",
    "fig8_gap_analysis_heatmap.pdf": "0.98\\textwidth",
    "fig3_staged_training_timeline.pdf": "0.88\\textwidth",
    "fig4_baseline_inference_bars.pdf": "0.98\\textwidth",
    "fig5_training_results_flagged.pdf": "0.98\\textwidth",
    "fig7_loss_scale_comparison.pdf": "0.78\\textwidth",
}


def escape_text(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def inline_latex(s: str) -> str:
    parts: list[str] = []
    pattern = re.compile(r"(`[^`]+`|\\\([^)]+\\\)|\*\*[^*]+\*\*)")
    pos = 0
    for m in pattern.finditer(s):
        if m.start() > pos:
            parts.append(escape_text(s[pos:m.start()]))
        token = m.group(0)
        if token.startswith("`"):
            code = token[1:-1]
            parts.append(code_latex(code))
        elif token.startswith(r"\("):
            parts.append(token)
        elif token.startswith("**"):
            parts.append(r"\textbf{" + inline_latex(token[2:-2]) + "}")
        pos = m.end()
    if pos < len(s):
        parts.append(escape_text(s[pos:]))
    return "".join(parts)


def code_latex(code: str) -> str:
    if re.fullmatch(r"[A-Fa-f0-9]{32,}", code):
        chunks = [code[i : i + 16] for i in range(0, len(code), 16)]
        return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"
    return r"\path{" + code + "}"


def clean_heading(line: str) -> str:
    text = line.lstrip("#").strip()
    replacements = {
        "8. Discussion": "7. Discussion",
        "9. Conclusion": "8. Conclusion",
    }
    text = replacements.get(text, text)
    return text


def heading_to_latex(line: str) -> str:
    level = len(line) - len(line.lstrip("#"))
    title = clean_heading(line)
    if re.match(r"^\d+\.\s+", title):
        title = re.sub(r"^\d+\.\s+", "", title)
    if level == 1:
        return "\\section{" + inline_latex(title) + "}\n"
    if level == 2:
        title = re.sub(r"^\d+\.\d+\s+", "", title)
        return "\\subsection{" + inline_latex(title) + "}\n"
    if level == 3:
        return "\\subsubsection{" + inline_latex(title) + "}\n"
    return "\\paragraph{" + inline_latex(title) + "}\n"


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append(lines[i].strip())
        i += 1
    if len(rows) < 2:
        return inline_latex(lines[start]) + "\n", start + 1

    def split(row: str) -> list[str]:
        return [c.strip() for c in row.strip("|").split("|")]

    headers = split(rows[0])
    body = [split(r) for r in rows[2:]]
    n = len(headers)
    if n <= 2:
        width = 0.38
    elif n <= 4:
        width = min(0.28, 0.94 / n)
    else:
        width = max(0.08, min(0.22, 0.94 / n))
    spec = "@{}" + "".join([f">{{\\RaggedRight\\arraybackslash}}p{{{width:.3f}\\textwidth}}" for _ in range(n)]) + "@{}"

    out = ["\\begingroup", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{longtable}}{{{spec}}}", "\\toprule"]
    out.append(" & ".join(inline_latex(h) for h in headers) + r" \\")
    out.append("\\midrule")
    out.append("\\endfirsthead")
    out.append("\\toprule")
    out.append(" & ".join(inline_latex(h) for h in headers) + r" \\")
    out.append("\\midrule")
    out.append("\\endhead")
    for row in body:
        row = row + [""] * (n - len(row))
        out.append(" & ".join(inline_latex(c) for c in row[:n]) + r" \\")
    out.append("\\bottomrule")
    out.append("\\end{longtable}")
    out.append("\\endgroup")
    return "\n".join(out) + "\n", i


def figure_env(caption_line: str, source_line: str) -> str | None:
    m = re.search(r"`([^`]+\.pdf)`", source_line)
    if not m:
        return None
    src = Path(m.group(1)).name
    caption = re.sub(r"^\*\*Figure\s+[^.]+\.\*\*\s*", "", caption_line).strip()
    caption = re.sub(r"^Figure\s+[^.]+\.\s*", "", caption)
    caption = re.sub(r"^\*\*Figure\s+[^.]+\.[^*]*\*\*\s*", "", caption).strip()
    width = FIGURE_WIDTHS.get(src, "0.9\\textwidth")
    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width={width}]{{assets/{src}}}\n"
        f"\\caption{{{inline_latex(caption)}}}\n"
        "\\end{figure}\n"
    )


def convert_markdown(md: str, appendices: bool = False) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    list_env = ""
    in_code = False
    code_lines: list[str] = []
    appendix_started = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_code:
            if stripped.startswith("```"):
                out.append("\\begin{verbatim}")
                out.extend(code_lines)
                out.append("\\end{verbatim}")
                code_lines = []
                in_code = False
            else:
                code_lines.append(line)
            i += 1
            continue

        if stripped.startswith("```"):
            in_code = True
            code_lines = []
            i += 1
            continue

        if stripped == "":
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            next_stripped = lines[j].strip() if j < len(lines) else ""
            next_is_same_list = (
                list_env == "itemize" and next_stripped.startswith("- ")
            ) or (
                list_env == "enumerate" and re.match(r"^\d+\.\s+", next_stripped)
            )
            if list_env and not next_is_same_list:
                out.append(f"\\end{{{list_env}}}")
                list_env = ""
            out.append("")
            i += 1
            continue

        if stripped.startswith("|"):
            if list_env:
                out.append(f"\\end{{{list_env}}}")
                list_env = ""
            table_tex, i = parse_table(lines, i)
            out.append(table_tex)
            continue

        if stripped.startswith("#"):
            if list_env:
                out.append(f"\\end{{{list_env}}}")
                list_env = ""
            if appendices and stripped.startswith("# Appendix"):
                if not appendix_started:
                    out.append("\\appendix")
                    appendix_started = True
                title = stripped.lstrip("#").strip()
                title = re.sub(r"^Appendix\s+[A-Z]:\s*", "", title)
                out.append("\\section{" + inline_latex(title) + "}")
            else:
                out.append(heading_to_latex(stripped))
            i += 1
            continue

        if stripped.startswith("- "):
            if list_env and list_env != "itemize":
                out.append(f"\\end{{{list_env}}}")
                list_env = ""
            if not list_env:
                out.append("\\begin{itemize}[leftmargin=*]")
                list_env = "itemize"
            out.append("\\item " + inline_latex(stripped[2:]))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            if list_env and list_env != "enumerate":
                out.append(f"\\end{{{list_env}}}")
                list_env = ""
            if not list_env:
                out.append("\\begin{enumerate}[leftmargin=*]")
                list_env = "enumerate"
            out.append("\\item " + inline_latex(re.sub(r"^\d+\.\s+", "", stripped)))
            i += 1
            continue

        if stripped.startswith("**Figure") and i + 1 < len(lines) and "Source asset:" in lines[i + 1]:
            fig = figure_env(stripped, lines[i + 1].strip())
            if fig:
                out.append(fig)
                i += 2
                continue

        if stripped.startswith("*Caption:") and stripped.endswith("*"):
            note = stripped.strip("*")
            note = re.sub(r"^Caption:\s*", "Note. ", note)
            out.append("\\emph{" + inline_latex(note) + "}")
            i += 1
            continue

        if stripped.startswith("**Table") and stripped.endswith("**"):
            table_title = stripped.strip("*")
            table_title = table_title.replace(":", ".", 1)
            out.append("\\emph{" + inline_latex(table_title) + "}")
            i += 1
            continue

        if stripped.startswith("*") and stripped.endswith("*"):
            out.append("\\emph{" + inline_latex(stripped.strip("*")) + "}")
        else:
            out.append(inline_latex(stripped) + "\n")
        i += 1

    if list_env:
        out.append(f"\\end{{{list_env}}}")
    return "\n".join(out)


def read(name: str) -> str:
    return (DRAFT / name).read_text(encoding="utf-8")


def main() -> None:
    abstract = read("abstract.md").split("\n", 2)[-1].strip()
    body = []
    for name in ORDER:
        body.append(convert_markdown(read(name)))
    appendices = convert_markdown(read("appendices.md"), appendices=True)

    tex = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,margin=0.85in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{float}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\usepackage{titlesec}
\usepackage{setspace}

\definecolor{facexblue}{HTML}{1F4E79}
\definecolor{facexgray}{HTML}{4B5563}

\titleformat{\section}{\Large\bfseries\color{facexblue}}{\thesection}{0.75em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.65em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{0.55em}{}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\makeatletter
\@addtoreset{figure}{section}
\makeatother
\renewcommand{\thefigure}{\thesection.\arabic{figure}}
\onehalfspacing

\title{\textbf{Reproducing FaceXFormer:}\\A Unified Transformer for Multi-Task Facial Analysis}
\author{Preetom Saha Arko \and Md Kamruzzaman Kamrul\\Purdue University}
\date{April 30, 2026}

\begin{document}
\maketitle

\begin{abstract}
""" + inline_latex(abstract) + r"""
\end{abstract}

\newpage
\tableofcontents
\newpage

""" + "\n\n".join(body) + "\n\n" + appendices + r"""

\end{document}
"""
    OUT.write_text(tex, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
