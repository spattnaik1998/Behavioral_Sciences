"""
Analysis of the two returned annotation workbooks, treated as AUTOMATED annotation.

Provenance (see results/04_annotation_automated/PROVENANCE.md): both workbooks were
written by the openpyxl library, not saved from Excel, at the same second, with a
full-sentence model-style rationale in every row. The annotators could not be reached
to confirm how the labels were produced. We therefore treat them as labels from two
unverified, probably LLM-based, annotators ("annotator A", "annotator B") and never as
human ground truth.

Outputs: results/04_annotation_automated/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

import config as C
import stats_utils as S

OUT = C.RESULTS_DIR / "04_annotation_automated"
RET = C.ANNOT_RETURNED_DIR
SYSTEMS = ["C0_labels_only", "C1_definitions", "C2_rag_only", "C3_rag_full", "C4_speculative",
           "R0_original_replication"]
DIMS = ["counterfactual", "responsibility", "controllability", "affect"]


def krippendorff_nominal(a, b) -> float:
    """Krippendorff's alpha, nominal, two coders, no missing values."""
    vals = pd.Series(list(a) + list(b))
    cats = vals.unique()
    o = pd.DataFrame(0.0, index=cats, columns=cats)
    for x, y in zip(a, b):
        o.loc[x, y] += 1; o.loc[y, x] += 1
    n_c = o.sum(1); n = n_c.sum()
    do = (o.values.sum() - np.trace(o.values)) / n
    de = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return 1 - do / de


def load():
    A = pd.read_excel(RET / "annotation_coder_A.xlsx", sheet_name="coding")
    B = pd.read_excel(RET / "annotation_coder_B.xlsx", sheet_name="coding")
    keep = ["review_uid", "primary_emotion", "forced_choice", "cf_focus", *DIMS, "certainty"]
    m = A[keep].merge(B[keep], on="review_uid", suffixes=("_A", "_B"))
    meta = pd.read_csv(C.DATA_DIR / "annotation_set.csv")
    pool = pd.read_csv(C.RESULTS_DIR / "03_llm_outputs" / "pool_master.csv")
    lex = pd.read_csv(C.RESULTS_DIR / "05_baselines" / "pool_lexicon_vader.csv")[["review_uid", "lexicon_label"]]
    m = m.merge(meta, on="review_uid").merge(pool, on="review_uid", suffixes=("", "_pool")).merge(lex, on="review_uid")
    m["random"] = m["gold_origin"] != "targeted"
    m["consensus"] = np.where(m.forced_choice_A == m.forced_choice_B, m.forced_choice_A, None)
    return m


def wilson(k, n):
    lo, hi = S.wilson_ci(k, n) if hasattr(S, "wilson_ci") else (np.nan, np.nan)
    return f"[{100*lo:.1f}, {100*hi:.1f}]"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load()
    res = {}

    # 1. Agreement between the two annotators
    agr = []
    for f in ["primary_emotion", "forced_choice", "cf_focus"]:
        for sub, d in [("all 285", m), ("random 200", m[m.random])]:
            a, b = d[f"{f}_A"].astype(str), d[f"{f}_B"].astype(str)
            agr.append({"field": f, "subset": sub, "n": len(d), "pct_agree": round(100 * (a == b).mean(), 1),
                        "cohen_kappa": round(cohen_kappa_score(a, b), 3),
                        "krippendorff_alpha": round(krippendorff_nominal(a, b), 3)})
    for dim in DIMS:
        r = stats.spearmanr(m[f"{dim}_A"], m[f"{dim}_B"]).statistic
        agr.append({"field": dim, "subset": "all 285", "n": len(m), "spearman_A_B": round(r, 3),
                    "mean_A": round(m[f"{dim}_A"].mean(), 2), "mean_B": round(m[f"{dim}_B"].mean(), 2)})
    agr = pd.DataFrame(agr); agr.to_csv(OUT / "annotator_agreement.csv", index=False)

    # Scale check: responsibility should be LOW for disappointment, HIGH for regret
    chk = []
    for c in "AB":
        for lab in ["REGRET", "DISAPPOINTMENT"]:
            d = m[m[f"primary_emotion_{c}"] == lab]
            chk.append({"annotator": c, "own_label": lab, "n": len(d),
                        **{f"mean_{dim}": round(d[f"{dim}_{c}"].mean(), 2) for dim in DIMS}})
    pd.DataFrame(chk).to_csv(OUT / "dimension_scale_check.csv", index=False)

    # 2. Regret prevalence on the random 200
    rnd = m[m.random]
    prev = []
    for name, s in [("annotator A", rnd.forced_choice_A), ("annotator B", rnd.forced_choice_B),
                    ("A and B agree", rnd.consensus.dropna())] + \
                   [(k, rnd[f"{k}__label"]) for k in SYSTEMS] + [("B1_lexicon", rnd.lexicon_label)]:
        k, n = int((s == "REGRET").sum()), int(s.notna().sum())
        lo, hi = S.wilson_ci(k, n)
        prev.append({"labeler": name, "n": n, "n_regret": k, "pct_regret": round(100 * k / n, 1), "ci95": f"[{100*lo:.1f}, {100*hi:.1f}]"})
    pd.DataFrame(prev).to_csv(OUT / "prevalence_random200.csv", index=False)

    # 3. Agreement of each system with annotator A, B and the consensus (random 200, primary)
    rows = []
    for ref in ["A", "B", "consensus"]:
        for sub, d in [("random 200", rnd), ("all 285", m)]:
            col = f"forced_choice_{ref}" if ref != "consensus" else "consensus"
            d = d.dropna(subset=[col])
            for k in SYSTEMS + ["B1_lexicon"]:
                pcol = f"{k}__label" if k != "B1_lexicon" else "lexicon_label"
                met = S.metrics_with_ci(d[col].values, d[pcol].values)
                yt, yp = d[col] == "REGRET", d[pcol] == "REGRET"
                tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
                rows.append({"reference": ref, "subset": sub, "system": k, "n": len(d),
                             "n_regret_reference": int(yt.sum()),
                             "accuracy": round(met["accuracy"], 3), "kappa": round(met["kappa_vs_human"], 3),
                             "macro_f1": round(met["macro_f1"], 3), "f1_regret": round(met["f1_REGRET"], 3),
                             "precision_regret": round(tp / (tp + fp), 3) if tp + fp else np.nan,
                             "recall_regret": round(tp / (tp + fn), 3) if tp + fn else np.nan,
                             **{k2: v for k2, v in met.items() if k2.endswith("_ci95")}})
    acc = pd.DataFrame(rows); acc.to_csv(OUT / "system_vs_annotators.csv", index=False)

    # 4. McNemar on correctness vs consensus (all 285 where A and B agree)
    d = m.dropna(subset=["consensus"])
    mc = []
    for a, b in [("C1_definitions", "C3_rag_full"), ("C3_rag_full", "C4_speculative"),
                 ("R0_original_replication", "C3_rag_full"), ("C0_labels_only", "C1_definitions")]:
        ca, cb = d[f"{a}__label"] == d.consensus, d[f"{b}__label"] == d.consensus
        t = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
        mc.append({"A": a, "B": b, "n": len(d), "acc_A": round(ca.mean(), 3), "acc_B": round(cb.mean(), 3),
                   "only_A_right": t[0][1], "only_B_right": t[1][0], "p": mcnemar(t, exact=True).pvalue})
    pd.DataFrame(mc).to_csv(OUT / "mcnemar_correctness_vs_consensus.csv", index=False)

    # 5. Composite score by consensus label (scores from GPT-4o, labels from other annotators)
    d = m.dropna(subset=["consensus", "rds_paper"])
    x, y = d.loc[d.consensus == "REGRET", "rds_paper"], d.loc[d.consensus == "DISAPPOINTMENT", "rds_paper"]
    rep = S.two_group_report(x.values, y.values)
    rep = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)) for k, v in rep.items()}
    json.dump(rep, open(OUT / "composite_by_consensus.json", "w"), indent=1, default=str)

    # 6. Counterfactual focus: GPT-4o vs annotators
    cf = []
    for ref in "AB":
        dd = m.dropna(subset=["cf_focus"])
        cf.append({"pair": f"GPT-4o vs {ref}", "n": len(dd),
                   "pct_agree": round(100 * (dd.cf_focus == dd[f"cf_focus_{ref}"]).mean(), 1),
                   "kappa": round(cohen_kappa_score(dd.cf_focus, dd[f"cf_focus_{ref}"]), 3)})
    pd.DataFrame(cf).to_csv(OUT / "cf_focus_agreement.csv", index=False)

    m.to_csv(OUT / "merged_annotations.csv", index=False)
    return agr, acc, mc, rep, cf


if __name__ == "__main__":
    agr, acc, mc, rep, cf = run()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(agr.to_string()); print()
    print(pd.read_csv(OUT / "dimension_scale_check.csv").to_string()); print()
    print(pd.read_csv(OUT / "prevalence_random200.csv").to_string()); print()
    print(acc[["reference", "subset", "system", "n", "n_regret_reference", "accuracy", "kappa", "macro_f1",
               "f1_regret", "precision_regret", "recall_regret", "kappa_vs_human_ci95", "f1_REGRET_ci95"]].to_string()); print()
    print(pd.DataFrame(mc).to_string()); print()
    print(json.dumps(rep, indent=1, default=str)); print(pd.DataFrame(cf).to_string())
