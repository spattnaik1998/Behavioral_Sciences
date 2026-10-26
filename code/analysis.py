"""
Stage 4 - analysis of all outputs. Every block checks that its inputs exist
and is skipped (with a note in results/06_analysis/STATUS.md) if they do not,
so this can be run repeatedly as results arrive.

Blocks and the reviewer comments they answer
  A  prevalence by condition, inter-condition agreement ........ R2.1 R3.5 R4.5-7
  B  accuracy vs human gold, McNemar ablation tests, calibration  R2.1 R2.3 R3.2 R3.5 R4.5-7
  C  test-retest stability of the proposed system ................ R4.2 (reproducibility)
  D  RDS: group comparison (robust), human validation, rating
     correlation with range-restriction analysis, weight
     sensitivity, data-driven weights ............................ R2.4 R2.5 R2.6 R3.2 R3.3 R3.4
  E  counterfactual focus vs labels .............................. R4.8
  F  error analysis exports ........................................ R2.7
  G  generalisability (product categories, external dataset) ..... R2.8 R4.9
  H  Speculative RAG diagnostics ................................... R1.2 R3.1 R4.7
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import stats_utils as S
import tasks as T

IN = C.RESULTS_DIR / "03_llm_outputs"
OUT = C.RESULTS_DIR / "06_analysis"
STATUS: list[str] = []

SYSTEM_NAMES = {
    "C0_labels_only": "GPT-4o, labels only (no retrieval, no definitions)",
    "C1_definitions": "GPT-4o + definitions (no retrieval)",
    "C2_rag_only": "RAG without definitions",
    "C3_rag_full": "RAG + definitions (proposed)",
    "C4_speculative": "Speculative RAG",
    "R0_original_replication": "Original paper pipeline (replication): original prompt + original KB",
    "R1_origprompt_newKB": "Original prompt + revised KB (decomposition)",
    "R2_newprompt_origKB": "Revised prompt + original KB (decomposition)",
    "C5_speculative_4o_drafter": "Speculative RAG with GPT-4o drafter",
    "B1_lexicon": "Regret-cue lexicon",
    "B3_tfidf_logreg_cv": "TF-IDF + logistic regression (5-fold CV)",
    "B4_nli_zeroshot": "Zero-shot NLI (BART-large-MNLI)",
    "B5_roberta_cv": "RoBERTa-base fine-tuned (5-fold CV)",
    "B6_roberta_silver": "RoBERTa-base trained on LLM silver labels",
}


def _note(msg):
    print("  -", msg); STATUS.append(msg)


def _save(df: pd.DataFrame, name: str, index=False):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=index)


def load_predictions() -> dict[str, pd.DataFrame]:
    preds = {}
    for f in sorted(IN.glob("pool__*__run0.csv")):
        cond = f.stem.split("__")[1]
        preds[cond] = pd.read_csv(f)
    b = C.RESULTS_DIR / "05_baselines"
    if (b / "pool_lexicon_vader.csv").exists():
        d = pd.read_csv(b / "pool_lexicon_vader.csv")
        preds["B1_lexicon"] = d.rename(columns={"lexicon_label": "label"})[["review_uid", "label"]]
    for name, f in [("B3_tfidf_logreg_cv", "gold_tfidf_logreg_cv.csv"), ("B4_nli_zeroshot", "pool_nli_zeroshot.csv"),
                    ("B5_roberta_cv", "gold_roberta_cv.csv"), ("B6_roberta_silver", "gold_roberta_silver.csv")]:
        if (b / f).exists():
            preds[name] = pd.read_csv(b / f)
    return preds


def pool_n() -> int:
    return len(pd.read_csv(C.DATA_DIR / "pool.csv", usecols=["review_uid"]))


def id_sets() -> dict[str, set]:
    """Named review subsets used across blocks."""
    out = {"paper_sample_100": set(pd.read_csv(C.DATA_DIR / "original_sample_100.csv")["review_uid"])}
    f = C.DATA_DIR / "annotation_set.csv"
    if f.exists():
        a = pd.read_csv(f)
        out["annotation_all"] = set(a["review_uid"])
        out["annotation_random"] = set(a.loc[a.get("gold_origin", pd.Series("", index=a.index)) != "targeted", "review_uid"])
    return out


def load_gold() -> pd.DataFrame | None:
    f = C.DATA_DIR / "gold_labels.csv"
    return pd.read_csv(f) if f.exists() else None


# ==========================================================================
# A. Prevalence and inter-condition agreement (no human labels needed)
# ==========================================================================
def block_a(preds):
    from sklearn.metrics import cohen_kappa_score
    rows = []
    N = pool_n()
    full = {c: d for c, d in preds.items() if d["review_uid"].nunique() >= 0.99 * N}
    for c, d in full.items():
        lab = d["label"].dropna(); k = int((lab == "REGRET").sum()); n = len(lab)
        lo, hi = S.wilson_ci(k, n)
        rows.append({"system": c, "description": SYSTEM_NAMES.get(c, c), "n_classified": n,
                     "n_errors": int(d["label"].isna().sum()), "n_regret": k, "pct_regret": 100 * k / n,
                     "pct_regret_ci95": f"[{100*lo:.1f}, {100*hi:.1f}]"})
    if not rows:
        _note("A: no predictions yet"); return
    _save(pd.DataFrame(rows), "A_prevalence_by_system.csv")
    conds = list(full)
    kap = pd.DataFrame(index=conds, columns=conds, dtype=float)
    agr = kap.copy()
    for a, b in combinations(conds, 2):
        m = preds[a][["review_uid", "label"]].merge(preds[b][["review_uid", "label"]], on="review_uid").dropna()
        kap.loc[a, b] = kap.loc[b, a] = cohen_kappa_score(m.label_x, m.label_y)
        agr.loc[a, b] = agr.loc[b, a] = (m.label_x == m.label_y).mean()
    _save(kap.round(3), "A_kappa_between_systems.csv", index=True)
    _save(agr.round(3), "A_pct_agreement_between_systems.csv", index=True)
    # label flips caused by adding retrieval (C1 -> C3) and by adding definitions (C2 -> C3)
    flips = []
    for a, b, what in [("C1_definitions", "C3_rag_full", "add retrieval"), ("C2_rag_only", "C3_rag_full", "add definitions"),
                       ("C0_labels_only", "C1_definitions", "add definitions (no retrieval)"),
                       ("C3_rag_full", "C4_speculative", "speculative drafting + verification"),
                       ("R0_original_replication", "C3_rag_full", "original -> corrected pipeline")]:
        if a in preds and b in preds:
            m = preds[a][["review_uid", "label"]].merge(preds[b][["review_uid", "label"]], on="review_uid").dropna()
            ct = pd.crosstab(m.label_x, m.label_y)
            flips.append({"change": what, "from": a, "to": b, "n": len(m), "pct_changed": 100 * (m.label_x != m.label_y).mean(),
                          "D_to_R": int(((m.label_x == "DISAPPOINTMENT") & (m.label_y == "REGRET")).sum()),
                          "R_to_D": int(((m.label_x == "REGRET") & (m.label_y == "DISAPPOINTMENT")).sum()),
                          "mcnemar_p_label_shift": S.mcnemar(m.label_x == "REGRET", m.label_y == "REGRET")["exact_p"]})
    if flips:
        _save(pd.DataFrame(flips), "A_label_changes_per_component.csv")


# ==========================================================================
# B. Accuracy against the human gold standard
# ==========================================================================
def block_b(preds, gold):
    if gold is None:
        _note("B: waiting for human gold labels (run annotation.agreement())"); return
    g = gold.dropna(subset=["forced_choice"]).copy()
    if "gold_origin" not in g:
        g["gold_origin"] = "random"; g["weight"] = 1.0
    g["weight"] = g["weight"].fillna(1.0)
    clear = gold[gold["primary_emotion"].isin(["REGRET", "DISAPPOINTMENT"])]
    rows, correct, correct_rand = [], {}, {}
    for c, d in preds.items():
        m = g[["review_uid", "forced_choice", "gold_origin", "weight"]].merge(d[["review_uid", "label"]], on="review_uid")
        if m.empty:
            continue
        m["label"] = m["label"].fillna("ERROR")      # errors count as wrong
        rnd = m[m["gold_origin"] != "targeted"]
        if len(rnd):
            rows.append({"system": c, "description": SYSTEM_NAMES.get(c, c),
                         "subset": "random sample (unbiased; primary result)",
                         **S.metrics_with_ci(rnd["forced_choice"].values, rnd["label"].values)})
            correct_rand[c] = rnd.set_index("review_uid").eval("forced_choice == label")
        rows.append({"system": c, "description": SYSTEM_NAMES.get(c, c),
                     "subset": "all coded, stratum-weighted (point estimates)",
                     **S.classification_metrics(m["forced_choice"].values, m["label"].values, sample_weight=m["weight"].values)})
        tg = m[m["gold_origin"] == "targeted"]
        if len(tg):
            rows.append({"system": c, "description": SYSTEM_NAMES.get(c, c),
                         "subset": "targeted reviews only (enriched; diagnostic)",
                         **S.classification_metrics(tg["forced_choice"].values, tg["label"].values)})
        correct[c] = m.set_index("review_uid").eval("forced_choice == label")
        mc = clear[["review_uid", "primary_emotion"]].merge(d[["review_uid", "label"]], on="review_uid").dropna()
        if len(mc):
            rows.append({"system": c, "description": SYSTEM_NAMES.get(c, c), "subset": "clear cases (human primary = R or D)",
                         **S.classification_metrics(mc["primary_emotion"].values, mc["label"].values)})
    res = pd.DataFrame(rows)
    _save(res, "B_accuracy_vs_human.csv")

    # paired ablation tests (same items)
    tests = [("C3_rag_full", "C1_definitions", "retrieval effect"), ("C3_rag_full", "C2_rag_only", "definitions effect (with RAG)"),
             ("C1_definitions", "C0_labels_only", "definitions effect (no RAG)"), ("C4_speculative", "C3_rag_full", "speculative effect"),
             ("C3_rag_full", "R0_original_replication", "revised vs original pipeline"),
             ("C5_speculative_4o_drafter", "C4_speculative", "GPT-4o vs GPT-4o-mini drafter"),
             ("R1_origprompt_newKB", "R0_original_replication", "KB effect (original prompt)"),
             ("R2_newprompt_origKB", "R0_original_replication", "prompt effect (original KB)")]
    tests += [("C3_rag_full", b, "proposed vs baseline") for b in preds if b.startswith("B")]
    out = []
    for subset, corr in [("random sample", correct_rand), ("all coded", correct)]:
        for a, b, what in tests:
            if a in corr and b in corr:
                j = pd.concat([corr[a], corr[b]], axis=1, join="inner").dropna()
                out.append({"subset": subset, "comparison": what, "A": a, "B": b, "n_paired": len(j),
                            "acc_A": j.iloc[:, 0].mean(), "acc_B": j.iloc[:, 1].mean(), **S.mcnemar(j.iloc[:, 0], j.iloc[:, 1])})
    if out:
        t = pd.DataFrame(out)
        from statsmodels.stats.multitest import multipletests
        t["holm_p"] = np.nan
        for sub, idx in t.groupby("subset").groups.items():
            t.loc[idx, "holm_p"] = multipletests(t.loc[idx, "exact_p"], method="holm")[1]
        _save(t, "B_mcnemar_ablation_tests.csv")

    # confusion matrices + calibration
    cm_rows, cal = [], []
    for c, d in preds.items():
        m = g[["review_uid", "forced_choice"]].merge(d, on="review_uid")
        if m.empty or "label" not in m:
            continue
        ct = pd.crosstab(m["forced_choice"], m["label"].fillna("ERROR"))
        for (h, p), v in ct.stack().items():
            cm_rows.append({"system": c, "human": h, "predicted": p, "n": int(v)})
        y = (m["forced_choice"] == "REGRET").astype(int)
        for col in ["p_regret_logprob", "confidence_verbal"]:
            if col in m and m[col].notna().sum() > 10:
                p = m[col].astype(float)
                if col == "confidence_verbal":          # convert confidence-in-chosen-label to P(REGRET)
                    p = np.where(m["label"] == "REGRET", p, 1 - p)
                from sklearn.metrics import roc_auc_score, brier_score_loss
                ok = ~np.isnan(p)
                if y[ok].nunique() == 2:
                    cal.append({"system": c, "score": col, "n": int(ok.sum()), "auc": roc_auc_score(y[ok], p[ok]),
                                "brier": brier_score_loss(y[ok], np.clip(p[ok], 0, 1)), "ece_10bin": S.ece(p[ok], y[ok])})
    if cm_rows:
        _save(pd.DataFrame(cm_rows), "B_confusion_matrices_long.csv")
    if cal:
        _save(pd.DataFrame(cal), "B_calibration.csv")

    # VADER: does generic polarity separate human regret from disappointment? (R4.6)
    b = C.RESULTS_DIR / "05_baselines" / "pool_lexicon_vader.csv"
    if b.exists():
        v = pd.read_csv(b).merge(g[["review_uid", "forced_choice"]], on="review_uid")
        x = v.loc[v.forced_choice == "REGRET", "vader_compound"]; y = v.loc[v.forced_choice == "DISAPPOINTMENT", "vader_compound"]
        if len(x) > 2 and len(y) > 2:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score((v.forced_choice == "REGRET").astype(int), v.vader_compound)
            rep = S.two_group_report(x, y)
            rep["auc_vader_for_regret"] = auc
            _save(S.to_table(rep), "B_vader_separates_regret_vs_disappointment.csv")
    # human prevalence: random sample (Wilson CI) and stratum-weighted estimate over all coded reviews
    rnd = g[g["gold_origin"] != "targeted"]
    k = int((rnd.forced_choice == "REGRET").sum()); n = len(rnd); lo, hi = S.wilson_ci(k, n)
    y = (g.forced_choice == "REGRET").astype(float).values; w = g["weight"].values
    pw = float((w * y).sum() / w.sum())
    r = S.rng(); strata = g["stratum"].fillna("all").values if "stratum" in g else np.array(["all"] * len(g))
    boots = []
    for _ in range(2000):
        idx = np.concatenate([r.choice(np.where(strata == h)[0], (strata == h).sum()) for h in np.unique(strata)])
        boots.append((w[idx] * y[idx]).sum() / w[idx].sum())
    prev = {"random_sample_n": n, "random_sample_regret": k, "pct": 100 * k / n if n else None,
            "ci95": [100 * lo, 100 * hi],
            "weighted_all_coded_n": int(len(g)), "weighted_pct": 100 * pw,
            "weighted_ci95_stratified_bootstrap": [100 * float(np.percentile(boots, 2.5)), 100 * float(np.percentile(boots, 97.5))],
            "human_primary_counts_random": rnd["primary_emotion"].value_counts().to_dict() if "primary_emotion" in rnd else {},
            "human_primary_counts_all": gold["primary_emotion"].value_counts().to_dict()}
    (OUT / "B_human_prevalence.json").write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")


# ==========================================================================
# C. Test-retest stability
# ==========================================================================
def block_c():
    from sklearn.metrics import cohen_kappa_score
    runs = sorted(IN.glob(f"pool__{C.MAIN_CONDITION}__run*.csv"))
    if len(runs) < 2:
        _note("C: stability reruns not available"); return
    ds = [pd.read_csv(f)[["review_uid", "label", "p_regret_logprob", "system_fingerprint"]].assign(run=f.stem[-1]) for f in runs]
    gold_ids = id_sets().get("annotation_random", set())
    ds = [d[d.review_uid.isin(gold_ids)] for d in ds]
    rows = []
    for a, b in combinations(range(len(ds)), 2):
        m = ds[a].merge(ds[b], on="review_uid").dropna(subset=["label_x", "label_y"])
        rows.append({"run_a": a, "run_b": b, "n": len(m), "pct_identical": (m.label_x == m.label_y).mean(),
                     "kappa": cohen_kappa_score(m.label_x, m.label_y),
                     "mean_abs_diff_p_regret": (m.p_regret_logprob_x - m.p_regret_logprob_y).abs().mean()})
    allm = ds[0][["review_uid"]]
    for i, d in enumerate(ds):
        allm = allm.merge(d[["review_uid", "label"]].rename(columns={"label": f"run_{i}"}), on="review_uid")
    rows.append({"run_a": "all", "run_b": "all", "n": len(allm),
                 "pct_identical": (allm.filter(regex=r"^run_").nunique(axis=1) == 1).mean()})
    fps = pd.concat(ds)["system_fingerprint"].value_counts().to_dict()
    t = pd.DataFrame(rows); t["system_fingerprints_seen"] = json.dumps(fps)
    _save(t, "C_test_retest_stability.csv")


# ==========================================================================
# D. RDS analyses
# ==========================================================================
WEIGHTS = {"paper (0.4/0.3/0.3)": (0.4, 0.3, 0.3, 0.4), "equal (0.33 each, affect 0.33)": (1/3, 1/3, 1/3, 1/3),
           "no counterfactual": (0.0, 0.5, 0.5, 0.4), "no affect": (0.4, 0.3, 0.3, 0.0),
           "attribution only (resp+ctrl)": (0.0, 0.5, 0.5, 0.0), "counterfactual heavy (0.6/0.2/0.2)": (0.6, 0.2, 0.2, 0.4),
           "responsibility only": (0.0, 1.0, 0.0, 0.0)}


def block_d(gold):
    mf = IN / "pool_master.csv"
    if not mf.exists():
        _note("D: pool_master.csv not available"); return
    m = pd.read_csv(mf)
    if "rds_paper" not in m:
        _note("D: RDS scores not available"); return
    lab = f"{C.MAIN_CONDITION}__label"
    ok = m.dropna(subset=["rds_paper"])
    summary = {"n_rds": len(ok), "n_rds_errors": int(m["rds_error"].sum()) if "rds_error" in m else None,
               "rds_mean": ok.rds_paper.mean(), "rds_sd": ok.rds_paper.std(), "rds_median": ok.rds_paper.median(),
               "pct_rds_below_0": 100 * (ok.rds_paper < 0).mean()}
    (OUT / "D_rds_descriptives.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _save(ok[[f"rds_{d}" for d in T.DIMS]].describe().T.round(3), "D_rds_dimension_descriptives.csv", index=True)
    _save(ok[[f"rds_{d}" for d in T.DIMS] + ["rds_paper", "reviews.rating"]].corr("spearman").round(3),
          "D_spearman_dimensions_rating.csv", index=True)

    # D1. RDS by LLM label (as in the paper) - robust statistics
    reps = []
    if lab in ok:
        x = ok.loc[ok[lab] == "REGRET", "rds_paper"]; y = ok.loc[ok[lab] == "DISAPPOINTMENT", "rds_paper"]
        if len(x) > 2:
            reps.append({"grouping": f"LLM label ({C.MAIN_CONDITION}), full pool", **S.two_group_report(x, y)})
        s100 = set(pd.read_csv(C.DATA_DIR / "original_sample_100.csv")["review_uid"])
        o = ok[ok.review_uid.isin(s100)]
        x = o.loc[o[lab] == "REGRET", "rds_paper"]; y = o.loc[o[lab] == "DISAPPOINTMENT", "rds_paper"]
        if len(x) > 2:
            reps.append({"grouping": f"LLM label ({C.MAIN_CONDITION}), paper's 100 reviews", **S.two_group_report(x, y)})
    # D2. RDS by HUMAN label (non-circular)
    if gold is not None:
        h = ok.merge(gold[["review_uid", "forced_choice", "primary_emotion"] + [f"human_{d}" for d in T.DIMS]], on="review_uid")
        x = h.loc[h.forced_choice == "REGRET", "rds_paper"]; y = h.loc[h.forced_choice == "DISAPPOINTMENT", "rds_paper"]
        if len(x) > 2:
            reps.append({"grouping": "HUMAN forced-choice label (gold set)", **S.two_group_report(x, y)})
        # LLM dimensions vs human dimensions
        dv = []
        for d in T.DIMS:
            a, b = h[f"rds_{d}"], h[f"human_{d}"]
            okk = a.notna() & b.notna()
            if okk.sum() > 5:
                cr = S.corr_report(a[okk], b[okk])
                dv.append({"dimension": d, "n": int(okk.sum()), "spearman_rho": cr["spearman_rho"],
                           "spearman_ci95": cr["spearman_ci95_boot"], "pearson_r": cr["pearson_r"],
                           "icc2_1_llm_vs_human_mean": S.icc2_1(a[okk].values, b[okk].values),
                           "mean_llm": a[okk].mean(), "mean_human": b[okk].mean()})
        if dv:
            _save(pd.DataFrame(dv), "D_llm_vs_human_dimensions.csv")
        # human-dimension RDS vs human label: is the composite valid when humans supply the inputs?
        hh = h.copy()
        for d in T.DIMS:
            hh[f"rds_{d}"] = h[f"human_{d}"]
        hh["rds_human_inputs"] = T.rds_composite(hh)
        x = hh.loc[hh.forced_choice == "REGRET", "rds_human_inputs"].dropna(); y = hh.loc[hh.forced_choice == "DISAPPOINTMENT", "rds_human_inputs"].dropna()
        if len(x) > 2:
            reps.append({"grouping": "HUMAN label, RDS computed from HUMAN dimension ratings", **S.two_group_report(x, y)})
    if reps:
        _save(pd.DataFrame(reps).T, "D_rds_group_comparisons.csv", index=True)

    # D3. Correlation with star rating + range restriction (R2.6, R3.2)
    rows = []
    cr = S.corr_report(ok["reviews.rating"], ok["rds_paper"]); rows.append({"subset": "pool (ratings 1-3)", **cr})
    if lab in ok:
        for L in ["REGRET", "DISAPPOINTMENT"]:
            s = ok[ok[lab] == L]
            if len(s) > 10:
                rows.append({"subset": f"within LLM label {L}", **S.corr_report(s["reviews.rating"], s["rds_paper"])})
    for d in T.DIMS:
        rows.append({"subset": f"rating vs {d} only", **S.corr_report(ok["reviews.rating"], ok[f"rds_{d}"])})
    # partial correlation controlling for negative affect: is the rating link just intensity?
    import statsmodels.formula.api as smf
    z = ok.rename(columns={"reviews.rating": "rating"})
    fit = smf.ols("rds_paper ~ rating + rds_affect", data=z).fit()
    res_r = smf.ols("rating ~ rds_affect", data=z).fit().resid; res_s = smf.ols("rds_paper ~ rds_affect", data=z).fit().resid
    rows.append({"subset": "partial: rating vs RDS | affect", **S.corr_report(res_r, res_s)})
    # attribution-only index (no affect term) vs rating
    z["attr_only"] = T.rds_composite(z, 0.0, 0.5, 0.5, 0.0)
    rows.append({"subset": "rating vs attribution-only index (resp+ctrl)", **S.corr_report(z["rating"], z["attr_only"])})
    _save(pd.DataFrame(rows), "D_rating_correlations.csv")
    # rating distribution of the pool vs full dataset (range restriction is by design)
    _save(ok.groupby("reviews.rating")["rds_paper"].agg(["count", "mean", "std", "median"]).round(3),
          "D_rds_by_rating.csv", index=True)

    # D4. Weight sensitivity (R2.4, R3.3)
    sens = []
    hum = gold[["review_uid", "forced_choice"]] if gold is not None else None
    for name, (wc, wr, wk, wa) in WEIGHTS.items():
        score = T.rds_composite(ok, wc, wr, wk, wa)
        row = {"weights": name, "w_cf": wc, "w_resp": wr, "w_ctrl": wk, "w_affect": wa,
               "spearman_with_paper_rds": stats.spearmanr(score, ok["rds_paper"]).statistic,
               "spearman_with_rating": stats.spearmanr(score, ok["reviews.rating"]).statistic}
        if lab in ok:
            x, y = score[ok[lab] == "REGRET"], score[ok[lab] == "DISAPPOINTMENT"]
            if len(x) > 2:
                row["hedges_g_llm_label"] = S.hedges_g(x, y)[1]
        if hum is not None:
            hh = ok.assign(s=score).merge(hum, on="review_uid").dropna(subset=["forced_choice"])
            if hh.forced_choice.nunique() == 2:
                from sklearn.metrics import roc_auc_score
                row["auc_vs_human_label"] = roc_auc_score((hh.forced_choice == "REGRET").astype(int), hh.s)
                row["hedges_g_human_label"] = S.hedges_g(hh.s[hh.forced_choice == "REGRET"], hh.s[hh.forced_choice == "DISAPPOINTMENT"])[1]
        sens.append(row)
    # random weight draws (Dirichlet) for a distribution of outcomes
    r = S.rng()
    draws = []
    for _ in range(2000):
        w = r.dirichlet([1, 1, 1]); wa = r.uniform(0, 0.6)
        score = T.rds_composite(ok, w[0], w[1], w[2], wa)
        d = {"w_cf": w[0], "w_resp": w[1], "w_ctrl": w[2], "w_affect": wa,
             "spearman_with_paper_rds": stats.spearmanr(score, ok["rds_paper"]).statistic}
        if lab in ok:
            x, y = score[ok[lab] == "REGRET"], score[ok[lab] == "DISAPPOINTMENT"]
            if len(x) > 2:
                d["hedges_g_llm_label"] = S.hedges_g(x, y)[1]
        draws.append(d)
    dr = pd.DataFrame(draws)
    _save(pd.DataFrame(sens), "D_weight_sensitivity.csv")
    _save(dr.describe(percentiles=[.025, .5, .975]).T.round(3), "D_weight_sensitivity_random_draws_summary.csv", index=True)
    dr.to_csv(OUT / "D_weight_sensitivity_random_draws.csv", index=False)

    # D5. Data-driven weights: logistic regression of the HUMAN label on the 4 LLM dimensions
    if gold is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
        hh = ok.merge(gold[["review_uid", "forced_choice"]], on="review_uid").dropna(subset=["forced_choice"])
        if hh.forced_choice.nunique() == 2 and (hh.forced_choice == "REGRET").sum() >= 5:
            X = hh[[f"rds_{d}" for d in T.DIMS]].values; y = (hh.forced_choice == "REGRET").astype(int).values
            k = min(5, int(y.sum()))
            p = cross_val_predict(LogisticRegression(max_iter=1000), X, y,
                                  cv=StratifiedKFold(k, shuffle=True, random_state=C.SEED), method="predict_proba")[:, 1]
            full = LogisticRegression(max_iter=1000).fit((X - X.mean(0)) / X.std(0), y)
            dd = {"n": len(y), "cv_auc_learned_weights": roc_auc_score(y, p),
                  "auc_paper_weights": roc_auc_score(y, hh["rds_paper"]),
                  "standardised_coefficients": dict(zip(T.DIMS, full.coef_[0].round(3).tolist()))}
            (OUT / "D_data_driven_weights.json").write_text(json.dumps(dd, indent=2, default=float), encoding="utf-8")


# ==========================================================================
# E. Counterfactual focus
# ==========================================================================
def block_e(gold):
    mf = IN / "pool_master.csv"
    if not mf.exists():
        return
    m = pd.read_csv(mf)
    if "cf_focus" not in m:
        _note("E: counterfactual focus not available"); return
    lab = f"{C.MAIN_CONDITION}__label"
    rows = []
    for name, col, df in [("LLM label", lab, m), ("human label", "forced_choice",
                                          m.merge(gold.drop(columns=["cf_focus"], errors="ignore"), on="review_uid") if gold is not None else None)]:
        if df is None or col not in df:
            continue
        ct = pd.crosstab(df[col], df["cf_focus"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi = stats.chi2_contingency(ct)
            v = np.sqrt(chi.statistic / (ct.values.sum() * (min(ct.shape) - 1)))
            rows.append({"by": name, "chi2": chi.statistic, "df": chi.dof, "p": chi.pvalue, "cramers_v": v})
        _save(ct, f"E_cf_focus_by_{name.replace(' ', '_')}.csv", index=True)
    if gold is not None and "cf_focus" in gold:
        from sklearn.metrics import cohen_kappa_score
        j = m[["review_uid", "cf_focus"]].merge(gold[["review_uid", "cf_focus"]], on="review_uid", suffixes=("_llm", "_human")).dropna()
        if len(j):
            rows.append({"by": "LLM cf_focus vs human cf_focus", "kappa": cohen_kappa_score(j.cf_focus_llm, j.cf_focus_human),
                         "pct_agreement": (j.cf_focus_llm == j.cf_focus_human).mean(), "n": len(j)})
    if rows:
        _save(pd.DataFrame(rows), "E_cf_focus_tests.csv")


# ==========================================================================
# F. Error analysis exports
# ==========================================================================
def block_f(preds, gold):
    if gold is None or C.MAIN_CONDITION not in preds:
        _note("F: needs gold labels and the main system's predictions"); return
    pool = pd.read_csv(C.DATA_DIR / "pool.csv")[["review_uid", "reviews.text", "reviews.rating", "name"]]
    d = preds[C.MAIN_CONDITION][["review_uid", "label", "p_regret_logprob", "confidence_verbal", "rationale", "retrieved"]]
    m = gold.merge(d, on="review_uid").merge(pool, on="review_uid")
    lex = C.RESULTS_DIR / "05_baselines" / "pool_lexicon_vader.csv"
    if lex.exists():
        m = m.merge(pd.read_csv(lex)[["review_uid", "lexicon_hits", "vader_compound"]], on="review_uid", how="left")
    mf = IN / "pool_master.csv"
    if mf.exists():
        mm = pd.read_csv(mf)
        cols = ["review_uid"] + [c for c in mm if c.startswith("rds_") or c == "cf_focus"]
        m = m.merge(mm[cols], on="review_uid", how="left")
    m["outcome"] = np.select([(m.forced_choice == "REGRET") & (m.label == "REGRET"),
                              (m.forced_choice == "DISAPPOINTMENT") & (m.label == "DISAPPOINTMENT"),
                              (m.forced_choice == "REGRET") & (m.label == "DISAPPOINTMENT"),
                              (m.forced_choice == "DISAPPOINTMENT") & (m.label == "REGRET")],
                             ["TP_regret", "TN_disappointment", "FN_missed_regret", "FP_false_regret"], "other")
    m["n_words"] = m["reviews.text"].astype(str).str.split().str.len()
    m["has_self_blame_cue"] = m.get("lexicon_hits", pd.Series("", index=m.index)).fillna("").astype(str).str.len() > 0
    m["mentions_return_refund"] = m["reviews.text"].str.contains(r"\b(return|refund|send it back|sent it back)", case=False, regex=True)
    m["human_mixed_or_neither"] = m["primary_emotion"].isin(["BOTH", "NEITHER"])
    m.sort_values(["outcome", "p_regret_logprob"]).to_csv(OUT / "F_error_analysis_cases.csv", index=False)
    feats = ["has_self_blame_cue", "mentions_return_refund", "human_mixed_or_neither"]
    rows = []
    for f in feats:
        for v in [True, False]:
            s = m[m[f] == v]
            if len(s):
                rows.append({"feature": f, "value": v, "n": len(s), "accuracy": (s.forced_choice == s.label).mean(),
                             "pct_FN": (s.outcome == "FN_missed_regret").mean(), "pct_FP": (s.outcome == "FP_false_regret").mean()})
    for lo, hi in [(0, 10), (10, 30), (30, 10_000)]:
        s = m[(m.n_words >= lo) & (m.n_words < hi)]
        if len(s):
            rows.append({"feature": "n_words", "value": f"[{lo},{hi})", "n": len(s), "accuracy": (s.forced_choice == s.label).mean(),
                         "pct_FN": (s.outcome == "FN_missed_regret").mean(), "pct_FP": (s.outcome == "FP_false_regret").mean()})
    for r_ in [1, 2, 3]:
        s = m[m["reviews.rating"] == r_]
        if len(s):
            rows.append({"feature": "rating", "value": r_, "n": len(s), "accuracy": (s.forced_choice == s.label).mean(),
                         "pct_FN": (s.outcome == "FN_missed_regret").mean(), "pct_FP": (s.outcome == "FP_false_regret").mean()})
    _save(pd.DataFrame(rows), "F_error_rates_by_feature.csv")
    _save(m["outcome"].value_counts().rename_axis("outcome").reset_index(), "F_outcome_counts.csv")
    # a compact, human-readable sample for the paper (up to 5 per outcome)
    ex = m.groupby("outcome").head(5)
    ex[["outcome", "review_uid", "reviews.text", "forced_choice", "primary_emotion", "label", "p_regret_logprob",
        "rationale"]].to_csv(OUT / "F_examples_for_paper.csv", index=False)


# ==========================================================================
# G. Generalisability
# ==========================================================================
def block_g():
    mf = IN / "pool_master.csv"
    if not mf.exists():
        return
    m = pd.read_csv(mf)
    lab = f"{C.MAIN_CONDITION}__label"
    rows = []

    def grp(name, s):
        k = int((s[lab] == "REGRET").sum()); n = int(s[lab].notna().sum()); lo, hi = S.wilson_ci(k, n)
        r = {"group": name, "n": n, "pct_regret": 100 * k / n if n else np.nan, "ci95": f"[{100*lo:.1f}, {100*hi:.1f}]"}
        if "rds_paper" in s and s["rds_paper"].notna().sum() > 10:
            r["rds_mean"] = s["rds_paper"].mean()
            r["spearman_rating_rds"] = stats.spearmanr(s["reviews.rating"], s["rds_paper"], nan_policy="omit").statistic
        return r

    if lab in m:
        rows.append(grp("main pool - all", m))
        for cat, s in m.groupby("primaryCategories"):
            if len(s) >= 30:
                rows.append(grp(f"main pool - {cat}", s))
        for r_, s in m.groupby("reviews.rating"):
            rows.append(grp(f"main pool - rating {int(r_)}", s))
        ct = pd.crosstab(m["primaryCategories"], m[lab])
        ct = ct[ct.sum(1) >= 30]
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi = stats.chi2_contingency(ct)
            (OUT / "G_category_chi2.json").write_text(json.dumps({"chi2": chi.statistic, "df": chi.dof, "p": chi.pvalue}, default=float), encoding="utf-8")
    ef = IN / f"external__{C.MAIN_CONDITION}__run0.csv"
    if ef.exists():
        e = pd.read_csv(C.DATA_DIR / "external_sample.csv").merge(pd.read_csv(ef)[["review_uid", "label"]], on="review_uid")
        e = e.rename(columns={"label": lab})
        rf = IN / "external__rds.csv"
        if rf.exists():
            r = pd.read_csv(rf); e = e.merge(r, on="review_uid", how="left"); e["rds_paper"] = T.rds_composite(e)
        rows.append(grp("EXTERNAL - Amazon devices (1429_1)", e))
        for r_, s in e.groupby("reviews.rating"):
            rows.append(grp(f"EXTERNAL - rating {int(r_)}", s))
    if rows:
        _save(pd.DataFrame(rows), "G_generalisability.csv")


# ==========================================================================
# I. Same-review comparison of ALL systems (incl. annotation-set-only runs) and the
#    2x2 decomposition of the original pipeline (prompt x knowledge base)
# ==========================================================================
def block_i(preds):
    ids = id_sets()
    subsets = [(k, ids[k]) for k in ["paper_sample_100", "annotation_random", "annotation_all"] if k in ids]
    rows = []
    for c, d in preds.items():
        if c.startswith(("B3", "B5", "B6")):
            continue
        for name, s_ids in subsets:
            dd = d[d["review_uid"].isin(s_ids)].dropna(subset=["label"])
            if len(dd) < 0.95 * len(s_ids):
                continue
            k = int((dd["label"] == "REGRET").sum()); n = len(dd); lo, hi = S.wilson_ci(k, n)
            rows.append({"system": c, "description": SYSTEM_NAMES.get(c, c), "subset": name, "n": n, "n_regret": k,
                         "pct_regret": 100 * k / n, "ci95": f"[{100*lo:.1f}, {100*hi:.1f}]"})
    if rows:
        t = pd.DataFrame(rows)
        t.loc[len(t)] = {"system": "PAPER (as published)", "description": "reported in the manuscript", "subset": "paper_sample_100",
                         "n": 100, "n_regret": 16, "pct_regret": 16.0, "ci95": "[10.1, 24.4]"}
        _save(t, "I_systems_on_same_reviews.csv")
    need = ["R0_original_replication", "R1_origprompt_newKB", "R2_newprompt_origKB", "C3_rag_full"]
    if not all(k in preds for k in need):
        _note("I: decomposition runs (part 4) not available"); return
    cells = {"R0_original_replication": ("original prompt", "original KB (Mental Accounting only)"),
             "R1_origprompt_newKB": ("original prompt", "revised KB (all PDFs)"),
             "R2_newprompt_origKB": ("revised prompt", "original KB (Mental Accounting only)"),
             "C3_rag_full": ("revised prompt", "revised KB (all PDFs)")}
    out = []
    for name, s_ids in subsets:
        for c, (pr, kb) in cells.items():
            d = preds[c][preds[c]["review_uid"].isin(s_ids)].dropna(subset=["label"])
            out.append({"subset": name, "prompt": pr, "knowledge_base": kb, "system": c, "n": len(d),
                        "pct_regret": 100 * (d["label"] == "REGRET").mean() if len(d) else np.nan})
    t = pd.DataFrame(out)
    _save(t, "I_decomposition_long.csv")
    for name, _ in subsets:
        piv = t[t.subset == name].pivot(index="prompt", columns="knowledge_base", values="pct_regret").round(1)
        _save(piv, f"I_decomposition_2x2_{name}.csv", index=True)
    tests = []
    for a, b, what in [("R0_original_replication", "R1_origprompt_newKB", "KB effect | original prompt"),
                       ("R2_newprompt_origKB", "C3_rag_full", "KB effect | revised prompt"),
                       ("R0_original_replication", "R2_newprompt_origKB", "prompt effect | original KB"),
                       ("R1_origprompt_newKB", "C3_rag_full", "prompt effect | revised KB")]:
        for name, s_ids in subsets:
            m = preds[a][["review_uid", "label"]].merge(preds[b][["review_uid", "label"]], on="review_uid")
            m = m[m.review_uid.isin(s_ids)].dropna()
            tests.append({"comparison": what, "subset": name, "from": a, "to": b, "n": len(m),
                          "pct_regret_from": 100 * (m.label_x == "REGRET").mean(), "pct_regret_to": 100 * (m.label_y == "REGRET").mean(),
                          "R_to_D": int(((m.label_x == "REGRET") & (m.label_y != "REGRET")).sum()),
                          "D_to_R": int(((m.label_x != "REGRET") & (m.label_y == "REGRET")).sum()),
                          "mcnemar_p": S.mcnemar(m.label_x == "REGRET", m.label_y == "REGRET")["exact_p"]})
    _save(pd.DataFrame(tests), "I_decomposition_tests.csv")


# ==========================================================================
# F2. Where the systems disagree (qualitative error analysis before human labels)
# ==========================================================================
def block_f2(preds):
    keys = [k for k in ["C3_rag_full", "C4_speculative", "R0_original_replication", "C1_definitions"] if k in preds]
    if len(keys) < 2:
        return
    pool = pd.read_csv(C.DATA_DIR / "pool.csv")[["review_uid", "reviews.text", "reviews.rating", "name"]]
    m = pool.copy()
    for k in keys:
        cols = ["review_uid", "label"] + (["rationale"] if "rationale" in preds[k] else [])
        d = preds[k][cols].rename(columns={"label": f"{k}__label", "rationale": f"{k}__rationale"})
        m = m.merge(d, on="review_uid", how="left")
    labs = m[[f"{k}__label" for k in keys]]
    m["pattern"] = labs.apply(lambda r: " / ".join(f"{k.split('_')[0]}={str(v)[:1]}" for k, v in zip(keys, r)), axis=1)
    m["n_systems_regret"] = (labs == "REGRET").sum(axis=1)
    pat = m["pattern"].value_counts().rename_axis("pattern").reset_index(name="n")
    _save(pat, "F2_disagreement_patterns.csv")
    dis = m[(m["n_systems_regret"] > 0) & (m["n_systems_regret"] < len(keys))]
    ex = dis.sample(frac=1, random_state=C.SEED).groupby("pattern").head(8)
    ex.sort_values(["pattern"]).to_csv(OUT / "F2_disagreement_examples.csv", index=False)


# ==========================================================================
# H. Speculative RAG diagnostics
# ==========================================================================
def block_h():
    rows = []
    ids = id_sets()
    for cond in ["C4_speculative", "C5_speculative_4o_drafter"]:
        f = IN / f"pool__{cond}__drafts.csv"; s = IN / f"pool__{cond}__run0.csv"
        if not f.exists() or not s.exists():
            continue
        d = pd.read_csv(f); sm = pd.read_csv(s)
        for name, s_ids in [("all reviews run", None)] + [(k, ids[k]) for k in ["paper_sample_100", "annotation_random"] if k in ids]:
            ss = sm if s_ids is None else sm[sm.review_uid.isin(s_ids)]
            dd = d if s_ids is None else d[d.review_uid.isin(s_ids)]
            if len(ss) == 0:
                continue
            rows.append({"system": cond, "subset": name, "n_reviews": len(ss), "n_drafts": len(dd),
                         "draft_regret_rate": (dd["label"] == "REGRET").mean(),
                         "majority_vote_regret_rate": (ss["majority_label"] == "REGRET").mean(),
                         "selected_regret_rate": (ss["label"] == "REGRET").mean(),
                         "pct_all_drafts_agree": 100 * ss["drafts_agree"].mean(),
                         "pct_selected_differs_from_majority": 100 * (ss["label"] != ss["majority_label"]).mean(),
                         "pct_unanimous_regret_drafts": 100 * (ss["n_drafts_regret"] == C.SPEC_N_DRAFTS).mean(),
                         "mean_rho_draft": dd["rho_draft"].mean(), "mean_rho_sc": dd["rho_sc"].mean(),
                         "mean_rho_sr": dd["rho_sr"].mean(),
                         "tokens_in_per_review": ss["tokens_in"].mean(), "tokens_out_per_review": ss["tokens_out"].mean()})
        if cond == "C4_speculative":
            _save(sm["n_drafts_regret"].value_counts().sort_index().rename_axis("n_regret_drafts_of_4").reset_index(),
                  "H_draft_vote_distribution.csv")
            out = rows[0].copy()
            m3 = IN / f"pool__{C.MAIN_CONDITION}__run0.csv"
            if m3.exists():
                out["tokens_in_per_review_C3"] = pd.read_csv(m3)["tokens_in"].mean()
            (OUT / "H_speculative_rag_diagnostics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    if not rows:
        _note("H: speculative RAG outputs not available"); return
    _save(pd.DataFrame(rows), "H_speculative_rag_by_subset.csv")


# ==========================================================================
def run():
    OUT.mkdir(parents=True, exist_ok=True); STATUS.clear()
    preds, gold = load_predictions(), load_gold()
    for name, fn in [("A", lambda: block_a(preds)), ("B", lambda: block_b(preds, gold)), ("C", block_c),
                     ("D", lambda: block_d(gold)), ("E", lambda: block_e(gold)), ("F", lambda: block_f(preds, gold)),
                     ("G", block_g), ("H", block_h),
                     ("I", lambda: block_i(preds)), ("F2", lambda: block_f2(preds))]:
        print(f"[analysis] block {name}")
        try:
            fn()
        except Exception as e:  # keep going; record the failure
            import traceback; traceback.print_exc(); _note(f"{name}: FAILED - {e!r}")
    backend = set()
    for d in preds.values():
        if "backend" in d:
            backend |= set(d["backend"].dropna().unique())
    hdr = "# Analysis status\n\nLLM backend(s) in these results: " + (", ".join(sorted(backend)) or "n/a") + "\n\n"
    if "MOCK" in backend:
        hdr += "**WARNING: MOCK outputs - for code testing only, not results.**\n\n"
    (OUT / "STATUS.md").write_text(hdr + "\n".join(f"- {s}" for s in STATUS) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
