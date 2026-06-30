# Manual Revision Checklist for `report_updated.tex`

Source checked:

- `report_updated.tex`
- `main.tex` for the abstract, because the abstract is not inside `report_updated.tex`

Important compile note:

- `main.tex` currently uses `\input{sec/report}`, not `report_updated.tex`.
- If you want Overleaf to compile `report_updated.tex`, either copy its final contents into `sec/report.tex` or change the input path in `main.tex`.
- I did not find `main_updated.tex` in this folder.

Metric-table note:

- Your message says "according to the following table", but no new metric table was attached.
- Wherever this checklist says `<paper value>` or `<our value from released checkpoint>`, fill it using your table.
- In the results table, "our value" should mean the metric obtained by running/evaluating the released checkpoint, not the value from the rebuilt training run.

## 1. Abstract: remove staged training, add GitHub repo, and mention released artifacts

File: `main.tex`

Line 29 currently mentions `staged 3-to-6-to-8 task co-training` and does not include the GitHub link.

Replace the full abstract paragraph at line 29 with:

```tex
FaceXFormer proposes a single transformer for ten facial analysis tasks while reporting real-time inference and competitive or state-of-the-art accuracy. The public release includes the model architecture, pretrained weights, and inference code, but not the full training system, including training scripts, dataset loaders, multi-task sampler, loss implementations, task-specific augmentation policy, and fully specified loss coefficients. This paper presents a compact reproduction study of FaceXFormer in the ICCV format. We reconstruct an eight-task training and evaluation pipeline covering face parsing, landmark detection, head pose estimation, attribute prediction, age estimation, gender classification, race classification, and face visibility prediction. We combine a paper-vs-code audit, released-checkpoint verification, eight-task training analysis, and metric normalization analysis. The released checkpoint is broadly validated for segmentation (face parsing), age estimation, and attribute prediction on CelebA, with metrics close to the values claimed in the paper after normalization. The trained eight-task reproduction produces meaningful segmentation and attribute-prediction performance, but landmark, head pose, demographic, and visibility rows require careful qualification because of implementation divergences, evaluation bugs, or protocol ambiguity. Code and artifacts are available at \url{https://github.com/kamrul28890/FaceXFormer-a-unified-transformer}. The central conclusion is that FaceXFormer's architecture is plausible, but a checkpoint-and-inference release is insufficient for independent training reproduction of a modern multi-task vision system.
```

## 2. Introduction contribution list: keep checkpoint wording strict

File: `report_updated.tex`

Line 9 currently says:

```tex
This work studies \method as both a model and a reproducibility artifact. We reconstruct an eight-task version of the training and evaluation pipeline, verify the released checkpoint, train a multi-task model, and analyze the observed performance metrics of the model. Our contributions are:
```

Replace with:

```tex
This work studies \method as both a model and a reproducibility artifact. We reconstruct an eight-task version of the training and evaluation pipeline, evaluate the released checkpoint, train a multi-task model, and analyze which reported metrics are supported by the public release. Our contributions are:
```

Line 14 currently says:

```tex
\item released-checkpoint verification on 129,060 evaluation samples, showing that segmentation, age estimation, and attribute prediction match the performance metrics claimed in the paper closely after normalization;
```

Replace with:

```tex
\item released-checkpoint verification on 129,060 evaluation samples, showing that segmentation (face parsing), age estimation, and attribute prediction on CelebA are close to the metrics claimed in the paper after normalization;
```

## 3. Scope paragraph: use the exact exclusion reason for expression and recognition

File: `report_updated.tex`

Line 49 currently includes the right idea, but also adds an extra data/optimization-stack reason.

Replace line 49 with:

```tex
The original paper reports ten tasks. Our final reproduction covers eight: segmentation (face parsing), landmark detection, head pose estimation, attribute prediction, age estimation, gender classification, race classification, and face visibility prediction. Facial expression recognition and face recognition are excluded from the final training scope because the released checkpoint does not include these tasks. This scope is still large enough to test the central multi-task claim: a shared encoder-decoder can support dense prediction, coordinate regression, continuous regression, binary attributes, categorical demographics, and visibility outputs.
```

## 4. Paper-vs-code audit: avoid calling the release weights-only

File: `report_updated.tex`

Line 80 currently says the release is "weights-only" for training reproducibility. Since the authors released architecture, pretrained weights, and inference code, use more precise wording.

Replace line 80 with:

```tex
The reproduction began with a structured audit. Each item was classified by paper claim, released/reproduced behavior, chosen resolution, and practical impact. Table~\ref{tab:gaps} condenses the high-impact findings. The release is best described as inference-ready but not training-complete: the authors released the model architecture, pretrained weights, and inference code, but not the full training scripts, dataset loaders, sampler, and metric protocols needed to reproduce the optimization process.
```

Optional table-row addition after line 91:

```tex
Released artifacts & Architecture, pretrained weights, and inference path & Model architecture, pretrained weights, and inference code are public & Used for released-checkpoint verification & Available \\
```

If this row makes the table too wide, skip the row and keep only the prose replacement above.

## 5. Table 3 caption and lead-in: make it clearly evaluation, not training

File: `report_updated.tex`

Line 107 currently says:

```tex
The reconstructed pipeline uses the eight task groups listed in Table~\ref{tab:data}. We keep both paper-comparable rows and diagnostic rows. For example, CelebA attributes are paper-comparable, while LFWA attributes are cross-dataset diagnostic. FairFace gender/race rows are useful for debugging demographic heads, but they are not always paper-target rows because label taxonomies differ.
```

Replace with:

```tex
The reconstructed evaluation protocol uses the eight task groups listed in Table~\ref{tab:data}. We keep both paper-comparable rows and diagnostic rows. For example, attribute prediction on CelebA is paper-comparable, while attribute prediction on LFWA is cross-dataset diagnostic. FairFace gender/race rows are useful for debugging demographic heads, but they are not always paper-target rows because label taxonomies differ.
```

Line 110 currently says:

```tex
\caption{\textbf{Datasets, evaluation rows, and normalization rules.} This compact table preserves the important counts and units from the full report.}
```

Replace with:

```tex
\caption{\textbf{Evaluation datasets, reported rows, and metric units.} This compact table preserves the important evaluation counts and normalization rules from the full report.}
```

## 6. Remove staged-training methodology paragraph and figure

File: `report_updated.tex`

Line 144 currently says:

```tex
Multi-task sampling is implemented with dataset upsampling and task-aware batching. Without this, large datasets dominate optimization and small tasks such as COFW visibility appear rarely. The staged schedule starts with spatial tasks, then adds attributes and demographics, and finally trains the full eight-task set. Figure~\ref{fig:method} groups the audit summary with this staged schedule so the missing components and the chosen training resolution are visible together.
```

Replace with:

```tex
Multi-task sampling is implemented with dataset upsampling and task-aware batching. Without this, large datasets dominate optimization and small tasks such as COFW visibility appear rarely. Figure~\ref{fig:method} summarizes the audit findings that shaped the reproduced eight-task training and evaluation pipeline.
```

Lines 146-161 currently include the staged-training timeline image:

```tex
\begin{figure*}[!b]
  \centering
  \begin{subfigure}[t]{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{assets/fig8_gap_analysis_heatmap.png}
    \caption{Gap analysis summary.}
  \end{subfigure}
  \hfill
  \begin{subfigure}[t]{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{assets/fig3_staged_training_timeline.png}
    \caption{3-to-6-to-8 task schedule.}
  \end{subfigure}
  \caption{\textbf{Reproduction methodology.} The audit identifies high-impact missing or divergent components; the training plan then stages optimization from spatial tasks to the full eight-task reproduction.}
  \label{fig:method}
\end{figure*}
```

Replace the whole block with:

```tex
\begin{figure}[!t]
  \centering
  \includegraphics[width=\linewidth]{assets/fig8_gap_analysis_heatmap.png}
  \caption{\textbf{Reproduction methodology audit.} The audit identifies high-impact missing or divergent components that shaped the reproduced eight-task training and evaluation pipeline.}
  \label{fig:method}
\end{figure}
```

This removes every staged-training figure reference from the methodology section.

## 7. Results prose: include age as close if your new table supports it

File: `report_updated.tex`

Lines 200-202 currently say segmentation and CelebA attributes are close, but age is suspicious.

Replace lines 200-202 with:

```tex
The released checkpoint loads successfully and provides the cleanest evidence about the public model. Segmentation (face parsing) reaches 91.77 F1 on CelebAMask-HQ versus the paper target of 92.01. Attribute prediction on CelebA reaches 91.72\% versus the paper target of 91.83\%. Age estimation reaches <our value from released checkpoint> versus the paper target of <paper value>. These released-checkpoint rows are close to the metrics claimed in the paper after the appropriate unit normalization.

Other rows are more nuanced. Landmark NME on 300W full is 6.75 compared with the paper's 4.67, a moderate gap. Head pose is much worse under the current evaluation protocol: 20.65 degrees versus the paper's 3.52. COFW visibility is above the paper target, which may reflect protocol mismatch around Recall@P80.
```

Do not call age a clean match unless the new table values really show it is close.

## 8. Results table: make "our value" explicitly mean released checkpoint

File: `report_updated.tex`

Line 211 currently says:

```tex
\caption{\textbf{Main quantitative summary.} ``CKPT'' is released-checkpoint inference after normalization. ``Train'' is the staged eight-task reproduction. A dash means the row was diagnostic or not directly paper-targeted in the long report. Qualified rows are retained for transparency but not used as unqualified claims.}
```

Replace with:

```tex
\caption{\textbf{Main quantitative summary.} ``Our value'' denotes our evaluation of the released pretrained checkpoint after metric normalization; it is not a separately trained model. ``Train'' denotes the final eight-task training run. A dash means the row was diagnostic or not directly paper-targeted in the long report. Qualified rows are retained for transparency but not used as unqualified claims.}
```

Line 218 currently says:

```tex
Task & Dataset & Metric & Paper & CKPT & Train & Interpretation \\
```

Replace with:

```tex
Task & Dataset & Metric & Paper & \makecell{Our value\\(released ckpt.)} & Train & Interpretation \\
```

Line 220 currently says:

```tex
Seg. & CelebAMask-HQ & F1 (\%) & 92.01 & 91.77 & 85.91 & CKPT matches; trained model functional but below paper \\
```

Replace with:

```tex
Seg. & CelebAMask-HQ & F1 (\%) & 92.01 & 91.77 & 85.91 & Released checkpoint close; trained model functional but below paper \\
```

Line 225 currently says:

```tex
Attr. & CelebA & Acc. (\%) & 91.83 & 91.72 & 91.27 & Clean match for CKPT; close train row \\
```

Replace with:

```tex
Attr. & CelebA & Acc. (\%) & 91.83 & 91.72 & 91.27 & Attribute prediction on CelebA; released checkpoint close \\
```

Line 227 currently says:

```tex
Age & UTKFace & MAE yrs & 4.17 & 1.17 & 35.27 & CKPT/training both qualified by split/bin issues \\
```

Replace with the version consistent with your new table:

```tex
Age & UTKFace & MAE yrs & <paper value> & <our value from released checkpoint> & 35.27 & Released checkpoint close; train row not comparable \\
```

If you keep the current values `4.17` and `1.17`, then do not describe this as "close" without explaining the protocol or unit difference.

## 9. Rename the training subsection if it implies staged training

File: `report_updated.tex`

Line 204 currently says:

```tex
\subsection{Eight-Task Training}
```

This is acceptable. Do not rename it to "Staged Training".

Line 206 currently says:

```tex
The trained reproduction shows that the reconstructed pipeline is functional but not a complete reproduction of the paper. Segmentation reaches 85.91 F1, below the paper but meaningful for a reconstructed eight-task training run. CelebA attributes reach 91.27\%, within 0.56 percentage points of the paper target. LFWA attributes, UTKFace gender, and UTKFace race are useful diagnostic rows.
```

Replace with:

```tex
The trained reproduction shows that the reconstructed pipeline is functional but not a complete reproduction of the paper. Segmentation reaches 85.91 F1, below the paper but meaningful for a reconstructed eight-task training run. Attribute prediction on CelebA reaches 91.27\%, within 0.56 percentage points of the paper target. LFWA attribute prediction, UTKFace gender, and UTKFace race are useful diagnostic rows.
```

## 10. Cleanly supported section: add age and fix attribute wording

File: `report_updated.tex`

Line 240 currently says:

```tex
Three conclusions are strongly supported. First, the released checkpoint is usable and credible for at least the segmentation and CelebA attribute rows. These metrics land close to the paper targets after unit normalization. Second, the reconstructed training system can train an eight-task model end to end: segmentation and attributes do not collapse, and diagnostic demographic outputs are produced. Third, metric normalization is necessary. Raw segmentation and attribute values are fractions, while paper values are percentages; landmark and head-pose values are already in paper-style units after the fixed manifest. Figure~\ref{fig:results} visualizes these results and the loss-scale issue that helps explain why equal task weights are not neutral.
```

Replace with:

```tex
Three conclusions are strongly supported. First, the released checkpoint is usable and credible for segmentation (face parsing), age estimation, and attribute prediction on CelebA. These metrics land close to the paper targets after unit normalization. Second, the reconstructed training system can train an eight-task model end to end: segmentation and attribute prediction do not collapse, and diagnostic demographic outputs are produced. Third, metric normalization is necessary. Raw segmentation and attribute values are fractions, while paper values are percentages; landmark and head-pose values are already in paper-style units after the fixed manifest. Figure~\ref{fig:results} visualizes these results and the loss-scale issue that helps explain why equal task weights are not neutral.
```

## 11. Figure 3: remove staged-training caption

File: `report_updated.tex`

Line 253 currently says:

```tex
\caption{Staged training results.}
```

Replace with:

```tex
\caption{Final eight-task training results.}
```

Line 261 currently says:

```tex
\caption{\textbf{Results and interpretation.} Checkpoint verification supports some original claims, especially segmentation and attributes. Training results identify a functional pipeline but also mark rows that require repair or protocol checks. Loss scale is a likely source of instability in the multi-task setting.}
```

Replace with:

```tex
\caption{\textbf{Results and interpretation.} Released-checkpoint verification supports some original claims, especially segmentation (face parsing), age estimation, and attribute prediction on CelebA. Training results identify a functional pipeline but also mark rows that require repair or protocol checks. Loss scale is a likely source of instability in the multi-task setting.}
```

## 12. Discussion: remove task-staging reference

File: `report_updated.tex`

Line 269 currently says:

```tex
The released \method checkpoint is valuable, but it cannot answer how the model was trained. A multi-task training recipe includes the data loaders, label mappings, sampler, augmentations, loss functions, coefficients, learning-rate schedule, task staging, and metric post-processing. Omitting those components forces independent reproducers to make new decisions, and those decisions become part of the reproduced model.
```

Replace with:

```tex
The released \method checkpoint is valuable, but it cannot answer how the model was trained. A multi-task training recipe includes the data loaders, label mappings, sampler, augmentations, loss functions, coefficients, learning-rate schedule, and metric post-processing. Omitting those components forces independent reproducers to make new decisions, and those decisions become part of the reproduced model.
```

## 13. Limitations and future work: keep expression/recognition exclusion aligned with checkpoint fact

File: `report_updated.tex`

Line 285 currently says:

```tex
The main limitation is that this is an eight-task reproduction rather than the full ten-task paper setting. Expression recognition and face recognition remain to be implemented. The second limitation is that some results require reruns after bug fixes, especially landmark, head pose, and age. The third limitation is protocol verification: BIWI head-pose preprocessing, COFW Recall@P80, FairFace race mapping, and age-bin conversion should be independently checked against the original paper and datasets.
```

Replace with:

```tex
The main limitation is that this is an eight-task reproduction rather than the full ten-task paper setting. Facial expression recognition and face recognition are excluded because the released checkpoint does not include these tasks. The second limitation is that some results require reruns after bug fixes, especially landmark, head pose, and training-side age outputs. The third limitation is protocol verification: BIWI head-pose preprocessing, COFW Recall@P80, FairFace race mapping, and age-bin conversion should be independently checked against the original paper and datasets.
```

Line 287 currently says:

```tex
Future work should implement the paper-described 68-token landmark pathway and 9-token head-pose representation, add expression and recognition training, and run controlled loss-balancing experiments. A reproducibility release should include the exact training scripts, environment files, dataset manifests, split files, normalization rules, and figure-generation scripts.
```

Replace with:

```tex
Future work should implement the paper-described 68-token landmark pathway and 9-token head-pose representation and run controlled loss-balancing experiments. A reproducibility release should include the exact training scripts, environment files, dataset manifests, split files, normalization rules, and figure-generation scripts.
```

## 14. Recommendations checklist: remove staging

File: `report_updated.tex`

Line 315 currently says:

```tex
Training script & Defines optimization, staging, checkpoints \\
```

Replace with:

```tex
Training script & Defines optimization, checkpoints, and logging \\
```

## 15. Conclusion: include age and avoid checkpoint-only wording

File: `report_updated.tex`

Line 329 currently says:

```tex
This paper condenses a full \method reproduction report into an ICCV-style study. The balanced conclusion is that \method is a plausible and partially validated architecture, but the public release does not make the training process independently reproducible. The released checkpoint supports segmentation and attribute claims after correct metric normalization. The reconstructed eight-task training pipeline is functional, but several rows require professional qualification because of architectural divergences, evaluation bugs, or protocol ambiguity.
```

Replace with:

```tex
This paper condenses a full \method reproduction report into an ICCV-style study. The balanced conclusion is that \method is a plausible and partially validated architecture, and the public release provides the model architecture, pretrained weights, and inference code, but it does not make the training process independently reproducible. The released checkpoint supports segmentation (face parsing), age estimation, and attribute-prediction claims after correct metric normalization. The reconstructed eight-task training pipeline is functional, but several rows require professional qualification because of architectural divergences, evaluation bugs, or protocol ambiguity.
```

Line 331 currently says:

```tex
The broader message is that multi-task reproducibility requires more than pretrained weights. It requires the complete training and evaluation system: loaders, samplers, losses, coefficients, schedules, label mappings, and metric rules. For unified face analysis models, those components are not implementation background; they define the experiment.
```

Replace with:

```tex
The broader message is that multi-task reproducibility requires more than architecture files, pretrained weights, and inference code. It requires the complete training and evaluation system: loaders, samplers, losses, coefficients, schedules, label mappings, and metric rules. For unified face analysis models, those components are not implementation background; they define the experiment.
```

## 16. Search terms to verify after manual edits

After applying the changes, search the report and abstract for these terms:

```text
staged
staging
3-to-6-to-8
fig3_staged_training_timeline
CelebA prediction
CKPT
```

Expected result:

- No `staged`, `staging`, `3-to-6-to-8`, or `fig3_staged_training_timeline`.
- No `CelebA prediction`.
- `CKPT` should either be gone or clearly replaced by `Our value (released ckpt.)`.

