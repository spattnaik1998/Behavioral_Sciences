# When Retrieval Does Not Help: Benchmarking LLMs on Regret and Disappointment in Customer Reviews

Code, data, model outputs and human annotations for the paper by Sarthak Pattnaik, Chhayank Jain and Eugene Pinsky (Boston University), submitted to *Analytics* (MDPI).

The paper tests whether retrieval-augmented generation (RAG) helps GPT-4o tell regret (the reviewer blames their own choice) from disappointment (the reviewer blames the product) in 2,019 low-rated Amazon reviews. Two independent human annotators labelled 385 of them (Cohen's κ = 0.80). Against those labels:

| System | κ with human labels | F1 for regret |
|---|---|---|
| C0 GPT-4o, label names only | 0.73 | 0.75 |
| C1 GPT-4o + definitions | 0.72 | 0.75 |
| C3 RAG + definitions | 0.45 | 0.51 |
| C4 Speculative RAG (GPT-4o-mini drafter) | 0.20 | 0.31 |
| R0 First version's pipeline | 0.16 | 0.28 |
| B3 TF-IDF + logistic regression (CV) | 0.11 | 0.12 |

The annotators found regret in 2.0% of random reviews. Full results: `results/04_human_annotation_round2/` and `results/RESULTS_SUMMARY.md`.

## Repository layout

```
code/            all analysis code (each module's docstring says what it does)
notebooks/       00-05: the order the analyses were run (outputs cleared)
data/            derived review files, annotation sets, human gold labels
annotation/      codebook, annotator instructions, blank and returned workbooks
results/         every table and figure in the paper, plus the API response cache
knowledge_base/  put the four literature PDFs here (not included, see its README)
dataset/         put the raw Datafiniti CSVs here (optional, see its README)
original_v1/     the notebook behind the first submitted version
docs/            working notes from the revision
```

## Reproduce the results

The analysis scripts read the parsed model outputs in `results/03_llm_outputs/`, so the paper's numbers can be regenerated without an API key or any cost.

```bash
pip install -r requirements.txt
cd code
python analysis.py            # system comparisons, composite score, robustness
python figures.py             # figures that need no human labels
python analysis_round2.py     # everything against the human labels, and Figures 2-4 of the paper
```

To rerun the models themselves, copy `.env.example` to `.env`, add your OpenAI key, put the PDFs in `knowledge_base/`, and run the notebooks in order. Each paid notebook has a budget cap. `notebooks/05_RoBERTa_Baseline_optional.ipynb` needs `requirements-transformers.txt` and downloads `roberta-base`.

## What is and is not included

- **API cache.** The cached requests keep their model settings and responses, but the prompt text is removed, because it contains passages from copyrighted papers. Lookups use a hash of the full request, so a rerun with the same PDFs in `knowledge_base/` rebuilds the same prompts and still hits the cache instead of paying again. `results/prompts_appendix.txt` gives every prompt template.
- **Literature PDFs.** Not included (copyright). `knowledge_base/README.md` lists them.
- **Raw dataset.** Not included. `data/` holds the derived files the code needs; `dataset/README.md` says where to get the originals.
- **Human annotations.** `annotation/round2/returned/` holds both annotators' codes as CSV files. The signed workbooks, which contain the annotators' declarations and signatures, are kept private.
- **First annotation round.** `annotation/round1/returned_automated/` holds the first round's workbooks. Their files showed signs of automated labelling, so the paper does not use them as ground truth (Section 4.7).

## Citation

If you use the benchmark or code, please cite the paper (citation details will be added on publication).
