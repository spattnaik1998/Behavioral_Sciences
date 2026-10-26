# Revision: code, data and results for the reviewer responses

Everything needed to answer Reviewers 1–4 lives in this folder. The original project
(`../Project`, `../Notebook`, `../Dataset`) is untouched.

## Status (10 Oct 2026)

| Part | Status |
|---|---|
| Data preparation, dataset description, exact rebuild of the paper's 100-review sample | **Done**: `results/01_data/` |
| Re-analysis from the paper's reported statistics (Welch, CIs, power, RDS algebra) | **Done**: `results/00_reported_stats/` |
| Human-annotation workbooks + codebook (200 reviews, 2 blind coders) | **Ready to code**: `annotation/` |
| Lexicon and VADER baselines on all 2,019 reviews | **Done**: `results/05_baselines/` |
| Cost estimate for all LLM runs | **Done**: `results/03_llm_outputs/cost_estimate.csv` (≈ $32 indicative) |
| Architecture figure (RAG vs Speculative RAG) and power curve | **Done**: `results/07_figures/` |
| GPT-4o runs | C0–C3 **done**; Speculative RAG ≈ 40%; rest pending: notebooks 01–03 |
| Transformer baselines (NLI, RoBERTa) | **Optional, local** (stage 7b) |
| Agreement, accuracy, error analysis | **After the human coding is returned** |

All code was verified end to end (every stage, every analysis block, every figure, and the
notebook itself) with a deterministic fake model and synthetic coders, under pandas 2.2 and 3.0.
Those test outputs were discarded; nothing in `results/` comes from the fake model.
Rows produced by the fake model carry `backend = "MOCK"`, and `results/06_analysis/STATUS.md`
prints a warning if any are present.

## How to run (locally, in parts, so you never overspend)

The work is split into five notebooks in `notebooks/`. Run them in order. Only 01–03 cost money,
and each of those has its own `BUDGET_USD` cap.

| Notebook | What | OpenAI cost |
|---|---|---|
| `00_Setup_and_Free_Analyses` | setup, data, re-analysis of the reported statistics, lexicon/VADER baselines, annotation sheets, spending plan | free |
| `01_Core_Reviewer_Responses` | **main reviewer answers**: ablation C0–C3 + RDS dimensions | ≈ $1.5 left (C0–C3 already done) |
| `02_Speculative_RAG` | Speculative RAG (C4) | ≈ $5.5 left (≈ 40% done) |
| `03_Robustness_Extras` | original replication, stability, counterfactual focus, external dataset | ≈ $9 |
| `03b_Prompt_vs_KB_and_SpecRAG_Drafter` | why the original pipeline over-detected regret (prompt vs knowledge base, 2×2) + Speculative RAG with a GPT-4o drafter | ≈ $3–4 |
| `04_Annotation_Baselines_Final` | human agreement, supervised/transformer baselines, final analysis and figures | free |

How the spending protection works:
* Every paid notebook first shows what is already paid for and what is left (`Cost check` cell, free).
* `BUDGET_USD` caps **new** spend in that run. When the cap or your OpenAI credit is reached, the run
  stops cleanly. Top up and re-run the *Run* cell to continue from where it stopped.
* Finished calls are cached in `results/llm_cache/` and are never paid for twice. Failed calls
  (e.g. "no credits remaining") are not cached, so they are retried next time.
* Every run is logged in `results/03_llm_outputs/spend_ledger.csv`.
* Terminal alternative: `cd code`, then `python pipeline.py --remaining`, then
  `python pipeline.py --part part1_core --budget 3`.

Keep `Behavioral_Economics/.env` (holds `OPENAI_API_KEY`) out of anything you share.
The literature PDFs are read from `Materials/`.

**Note on the PDFs:** *Kirmani and Rao (2000)* is a scanned PDF with almost no extractable text
(1,627 characters over 14 pages), so it contributes almost nothing to retrieval. The other three
extract well. The Mental Accounting PDF yields 164 chunks, exactly as in the original notebook,
which confirms the chunking replicates.

## Results summary

`results/RESULTS_SUMMARY.md` collects every number for the manuscript and the reply letter, organised by
reviewer comment. It regenerates whenever notebooks 01–04 run (or `python code/summary.py`).

## Human annotation (the key addition: R2.3, R3.2, R4.3)

**Design (285 reviews).** 200 random reviews (the paper's 100 plus 100 stratified by rating), plus 85 targeted
reviews where the systems flag regret or disagree (C3 regret / Speculative-RAG-only regret / original-pipeline-only
regret). Each review carries a stratum weight (`data/annotation_set.csv`), so accuracy and prevalence are reported
(a) on the random 200 (unbiased, primary) and (b) on all 285 with stratum weights. The targeted reviews provide
enough regret-like cases to measure regret precision and recall. Coder B may code only part of the sheet, top to
bottom; agreement is computed on the overlap. Each workbook has a `practice` sheet of 10 non-study reviews for a
15-minute calibration. Required columns (orange headers): primary_emotion, forced_choice, certainty.

* `annotation/annotation_coder_A.xlsx` and `annotation_coder_B.xlsx`: the same 200 reviews, in a
  different random order for each coder, with drop-down fields. Star ratings are hidden.
  The set is the paper's original 100 reviews plus 100 more, sampled at random and stratified
  by rating.
* `annotation/CODEBOOK.md` (also a sheet inside each workbook): definitions grounded in
  Zeelenberg et al. and Marcatto & Ferrante, plus decision rules.
* Coders work independently and save their completed files into `annotation/returned/`.
  `annotation.agreement()` then computes Cohen's κ, Krippendorff's α, weighted κ and ICC; writes
  `annotation/adjudication.xlsx` with the disagreements; and builds `data/gold_labels.csv`.

## Folder map

```
code/            all logic (each module's docstring lists the reviewer comments it answers)
  config.py            paths, seeds, models, the ablation grid
  data_prep.py         stage 1 data flow + description
  reported_stats.py    stage 0 re-analysis from printed numbers
  llm.py               cached/retrying OpenAI client (+ MockClient for testing)
  kb.py                knowledge base: all PDFs, 500/50 chunks, FAISS
  prompts.py           every prompt, verbatim (dumped to results/prompts_appendix.txt)
  tasks.py             classification conditions, Speculative RAG, RDS, counterfactual focus
  pipeline.py          orchestrator + cost estimate
  annotation.py        coder workbooks, agreement, gold labels
  baselines_classic.py lexicon, VADER, TF-IDF+LR (CV)
  baselines_transformer.py zero-shot NLI, RoBERTa (CV and silver-label)
  stats_utils.py       Welch/bootstrap/permutation/McNemar/ICC/calibration helpers
  analysis.py          analysis blocks A-H -> results/06_analysis
  figures.py           publication figures -> results/07_figures
notebooks/00_…04_*.ipynb               local notebooks (run in order; old single notebook in notebooks/_archive)
data/            pool.csv (2,019), original_sample_100.csv, external_sample.csv (500), annotation_set.csv
annotation/      coder workbooks, codebook, returned/
knowledge_base/  optional override for the PDFs (default: ../Materials/)
results/         00 reported stats · 01 data · 02 knowledge base · 03 LLM outputs · 04 human annotation
                 05 baselines · 06 analysis · 07 figures · llm_cache · prompts_appendix.txt
```

## Experimental design (what changed versus the original paper)

| Original | Revision | Why |
|---|---|---|
| Only the first uploaded PDF (mental accounting) indexed | All literature PDFs indexed; original KB kept as replication R0 | Bug fix; R4.2 |
| n = 100 | All 2,019 reviews (rating ≤ 3) | R1.1, R2.2, R3.4 |
| Single condition | C0 labels only · C1 + definitions · C2 retrieval only · C3 retrieval + definitions · C4 Speculative RAG · R0 original | R2.1, R3.1, R3.5, R4.5–7 |
| No ground truth | 200 human-coded reviews, two coders, κ / α / ICC, adjudication | R2.3, R3.2, R4.3 |
| No baselines | Lexicon, VADER, TF-IDF+LR (5-fold CV), zero-shot NLI, RoBERTa (CV and LLM-silver) | R2.1, R4.5, R4.6 |
| Student t-test only | Welch, bootstrap CI, permutation, Mann-Whitney, Hedges' g + CI, power | R2.5, R3.4 |
| Fixed 0.4/0.3/0.3 weights | Closed-form algebra, 7 named + 2,000 random weight sets, data-driven weights, human-dimension validation | R2.4, R3.3 |
| Pearson with rating = "convergent validity" | Spearman/Kendall + CIs, per-dimension, partial on affect, range-restriction framing | R2.6, R3.2 |
| Failed RDS calls coded 0.0 | Failures kept as missing and counted | Data integrity |
| Counterfactual focus described, not implemented | Implemented and tested against labels and human coding | R4.8 |
| Single run | 3 runs on the gold set (test-retest κ), system fingerprints logged | Reproducibility |
| One product line | External 500-review sample (Kindle/Fire/Echo) + per-category results | R2.8, R4.9 |
| — | Error analysis exports (FN/FP cases, error rates by feature) | R2.7 |

## Findings already available (no API needed)

* **Data flow (R1.1):** 28,332 raw → 2,787 with rating ≤ 3 → 2,019 after removing duplicate texts
  → 100 sampled. The paper's Section 3.1 describes a different, 5,000-row file; that text needs correcting.
* **Reported statistics are internally inconsistent.** The printed group SDs (1.69, 1.62) imply
  t = 1.93 (p = 0.056) and an overall SD of 1.65, not the reported t = 1.790 (p = 0.0765) and
  SD 1.78. The reported t is consistent with a pooled SD of about 1.76. Recompute from the raw
  results in the revision.
* **Welch (R2.5):** t = 1.88, df = 20.6, p = 0.075; 95% CI for the mean difference [−0.09, 1.81].
  Cohen's d = 0.53, 95% CI [−0.01, 1.07]. The result is inconclusive (R3.4).
* **Power:** the 16-vs-84 design had about 48% power at the observed d. The full pool gives over 99%
  power at that d, and 80% power down to d ≈ 0.18.
* **RDS algebra (R2.4, R3.3):** the paper's formula reduces exactly to
  RDS = 0.4·CF + 0.6·Resp + 0.6·Ctrl − 0.4·Affect − 4.8. The (8 − x) reverse scoring adds nothing.
* **Lexicon baseline:** only 1.5% of the 2,019 reviews contain explicit first-person regret cues
  ("I should have", "my fault", "I wish I had"…).
