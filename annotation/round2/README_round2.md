# Round-2 human annotation: what to send and what to keep

## Send to each annotator (one folder each, nothing else)

| Folder | Contents |
|---|---|
| `FOR_ANNOTATOR_1/` | `annotation_round2_coder_1.xlsx`, `Annotation_Instructions` (.pdf and .docx), `Regret_and_Disappointment_Background` (.pdf and .docx) |
| `FOR_ANNOTATOR_2/` | the same, with `annotation_round2_coder_2.xlsx` |

Before sending, replace **[return date]** in section 4 of `Annotation_Instructions.docx` and re-save the PDF.

## Never send

- `data/attention_checks_round2.csv`: the answer key for the 12 hidden check items.
- Anything in `results/`, the manuscript, or the model outputs.
- The round-1 workbooks.

## The set

397 rows per annotator, in a different random order for each:

| Part | Reviews | Why |
|---|---|---|
| Round-1 random sample (includes the paper's original 100) | 200 | Unbiased prevalence and accuracy |
| Round-1 targeted | 85 | Regret-like cases |
| New targeted | 100 | Every remaining review GPT-4o (C3) labels regret, plus Speculative-RAG-only and original-pipeline-only cases, so regret precision and recall can be estimated |
| Attention checks | 12 | 6 clear regret, 6 clear disappointment; excluded from analysis |

Strata and weights: `data/annotation_set_round2.csv`. Practice reviews (not in the study): `data/practice_round2.csv`.

## When the files come back

1. Save them unchanged in `annotation/round2/returned/` under their original names.
2. Run `python code/annotation_round2.py` once to confirm the set, then in Python:
   `import annotation_round2 as R; R.score()`
   This prints the attention-check scores (pass = 10 or more of 12), checks how each file was written (Excel or a script), and computes Cohen's kappa and Krippendorff's alpha. It also writes the disagreements to `results/04_human_annotation_round2/disagreements_for_adjudication.csv`.
3. Adjudicate the disagreements (third person or the authors). The accuracy tables and supervised baselines follow from the gold labels.
