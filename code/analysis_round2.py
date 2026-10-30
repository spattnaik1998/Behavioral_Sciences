"""
Round-2 analysis against the verified human labels (Reviewers 2 and 3, second round).

Gold standard
  * Primary: the forced choice on which both human annotators agree (no adjudication needed).
  * If results/04_human_annotation_round2/disagreements_for_adjudication.csv has its
    'adjudicated_forced_choice' column filled, those reviews are added (gold_source = 'adjudicated').
  * Sensitivity: every metric is recomputed against annotator 1 alone and annotator 2 alone (all 385).
Attention-check items are excluded everywhere.

Outputs: results/04_human_annotation_round2/, data/gold_labels_round2.csv (used by the RoBERTa notebook),
         and the paper's Figures 2-4 (R8, R3, R5) in results/07_figures/. Run figures.py first, then this.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

import annotation_round2 as R2
import config as C
import stats_utils as S
from automated_annotation import krippendorff_nominal

OUT = C.RESULTS_DIR / "04_human_annotation_round2"
DIMS = ["counterfactual", "responsibility", "controllability", "affect"]
FULL = ["C0_labels_only", "C1_definitions", "C2_rag_only", "C3_rag_full", "C4_speculative", "R0_original_replication"]
SUBSET = {"C5_speculative_4o_drafter": "pool__C5_speculative_4o_drafter__run0.csv",
          "R1_origprompt_newKB": "pool__R1_origprompt_newKB__run0.csv",
          "R2_newprompt_origKB": "pool__R2_newprompt_origKB__run0.csv"}
NAMES = {"C0_labels_only": "C0 Labels only", "C1_definitions": "C1 Definitions", "C2_rag_only": "C2 RAG, no definitions",
         "C3_rag_full": "C3 RAG with definitions", "C4_speculative": "C4 Speculative RAG",
         "C5_speculative_4o_drafter": "C5 Speculative RAG, GPT-4o drafter", "R0_original_replication": "R0 Original pipeline",
         "R1_origprompt_newKB": "R1 Original prompt, revised KB", "R2_newprompt_origKB": "R2 Revised prompt, original KB",
         "B1_lexicon": "B1 Lexicon", "B3_tfidf_logreg_cv": "B3 TF-IDF + logistic regression (CV)",
         "B5_roberta_cv": "B5 RoBERTa (CV)", "B6_roberta_silver": "B6 RoBERTa trained on GPT-4o labels",
         "AUTO_A": "Round-1 automated annotator A", "AUTO_B": "Round-1 automated annotator B"}


# --------------------------------------------------------------------------
def build_table(attention_s1: bool = False) -> pd.DataFrame:
    """The 385 study reviews (default), or, with attention_s1=True, the attention checks drawn from stratum S1.

    Both annotators coded the attention checks blind, like every other review. The six from S1 were chosen as
    clear regret, so leaving them out would bias S1's regret share downward; they enter the pool-level
    estimates only (POOL_EXTRA), never the unweighted 372-review gold set."""
    a, b = R2._read("1"), R2._read("2")
    att = pd.read_csv(C.DATA_DIR / "attention_checks_round2.csv")
    keep = ["review_uid", "primary_emotion", "forced_choice", "cf_focus", *DIMS, "certainty"]
    m = a[keep].merge(b[keep], on="review_uid", suffixes=("_h1", "_h2"))
    s = pd.read_csv(C.DATA_DIR / "annotation_set_round2.csv")
    if attention_s1:
        import annotation as A_
        pool = pd.read_csv(C.DATA_DIR / "pool.csv")
        pool = pool.assign(stratum=A_.assign_strata(pool))
        pa = pool[pool["review_uid"].isin(att["review_uid"]) & (pool["stratum"] == "S1_C3_regret")]
        s = pa[["review_uid", "stratum", "reviews.rating", "primaryCategories"]].assign(
            gold_origin="attention_check_S1", stratum_N=int((pool["stratum"] == "S1_C3_regret").sum()))
        m = m[m["review_uid"].isin(s["review_uid"])]
    else:
        m = m[~m["review_uid"].isin(att["review_uid"])]
    pm = pd.read_csv(C.RESULTS_DIR / "03_llm_outputs" / "pool_master.csv")
    lex = pd.read_csv(C.RESULTS_DIR / "05_baselines" / "pool_lexicon_vader.csv")[["review_uid", "lexicon_label"]]
    m = m.merge(s, on="review_uid").merge(pm, on="review_uid", suffixes=("", "_pool")).merge(lex, on="review_uid")
    m = m.rename(columns={f"{k}__label": k for k in FULL}).rename(columns={"lexicon_label": "B1_lexicon"})
    for k, f in SUBSET.items():
        p = C.RESULTS_DIR / "03_llm_outputs" / f
        if p.exists():
            m = m.merge(pd.read_csv(p)[["review_uid", "label"]].rename(columns={"label": k}), on="review_uid", how="left")
    auto = C.RESULTS_DIR / "04_annotation_automated" / "merged_annotations.csv"
    if auto.exists():
        au = pd.read_csv(auto)[["review_uid", "forced_choice_A", "forced_choice_B"]].rename(
            columns={"forced_choice_A": "AUTO_A", "forced_choice_B": "AUTO_B"})
        m = m.merge(au, on="review_uid", how="left")
    # gold
    m["gold"] = np.where(m["forced_choice_h1"] == m["forced_choice_h2"], m["forced_choice_h1"], None)
    m["gold_source"] = np.where(m["gold"].notna(), "agreed", "disagreement")
    adj = OUT / "disagreements_for_adjudication.csv"
    if adj.exists():
        d = pd.read_csv(adj)
        if "adjudicated_forced_choice" in d and d["adjudicated_forced_choice"].notna().any():
            d = d.dropna(subset=["adjudicated_forced_choice"])
            d = d[d["adjudicated_forced_choice"].isin(["REGRET", "DISAPPOINTMENT"])]
            mp = dict(zip(d["review_uid"], d["adjudicated_forced_choice"]))
            hit = m["review_uid"].isin(mp) & m["gold"].isna()
            m.loc[hit, "gold"] = m.loc[hit, "review_uid"].map(mp); m.loc[hit, "gold_source"] = "adjudicated"
    m["gold_primary"] = np.where(m["primary_emotion_h1"] == m["primary_emotion_h2"], m["primary_emotion_h1"], None)
    m["random"] = m["gold_origin"].str.startswith("random")
    for d_ in DIMS:
        m[f"human_{d_}"] = m[[f"{d_}_h1", f"{d_}_h2"]].mean(axis=1)
    return m


def weighted_metrics(y, p, w):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w, float)
    yt, yp = y == "REGRET", p == "REGRET"
    acc = np.sum(w * (yt == yp)) / w.sum()
    tp, fp, fn = np.sum(w * (yt & yp)), np.sum(w * (~yt & yp)), np.sum(w * (yt & ~yp))
    prec = tp / (tp + fp) if tp + fp else np.nan; rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return acc, prec, rec, f1


POOL_EXTRA = pd.DataFrame()   # gold-labelled S1 attention checks, set in run()


def pool_frame(d: pd.DataFrame, col: str | None = None) -> pd.DataFrame:
    """Rows for pool-level estimates: the gold set plus the gold-labelled S1 attention checks, with weights."""
    x = POOL_EXTRA.dropna(subset=["gold"] + ([col] if col else [])) if len(POOL_EXTRA) else POOL_EXTRA
    out = pd.concat([d, x], ignore_index=True) if len(x) else d.copy()
    return out.assign(weight=gold_weights(out))


def gold_weights(d: pd.DataFrame) -> pd.Series:
    """w_h = N_h / m_h, where N_h is the stratum's size in the 2019-review pool and m_h the number of
    gold-labelled reviews from that stratum. Every gold-labelled review in stratum h gets w_h."""
    m_h = d.groupby("stratum")["review_uid"].transform("size")
    return d["stratum_N"] / m_h


def strat_boot(df, col, ref, n=2000, seed=C.SEED):
    rng = np.random.default_rng(seed); res = []
    groups = [g for _, g in df.groupby("stratum")]
    for _ in range(n):
        bs = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for g in groups])
        res.append(weighted_metrics(bs[ref], bs[col], bs["weight"]))
    r = np.array(res, float)
    return {k: f"[{np.nanpercentile(r[:, i], 2.5):.3f}, {np.nanpercentile(r[:, i], 97.5):.3f}]"
            for i, k in enumerate(["w_accuracy_ci95", "w_precision_ci95", "w_recall_ci95", "w_f1_ci95"])}


def system_metrics(m, systems, ref="gold", subset_name="all agreed"):
    rows = []
    d0 = m.dropna(subset=[ref])
    for k in systems:
        if k not in d0:
            continue
        d = d0.dropna(subset=[k])
        if d.empty:
            continue
        met = S.metrics_with_ci(d[ref].values, d[k].values)
        yt, yp = d[ref] == "REGRET", d[k] == "REGRET"
        tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
        bal = 0.5 * (tp / max(tp + fn, 1) + int((~yt & ~yp).sum()) / max(int((~yt).sum()), 1))
        r = {"reference": ref, "subset": subset_name, "system": k, "name": NAMES.get(k, k), "n": len(d),
             "n_regret_ref": int(yt.sum()), "n_regret_pred": int(yp.sum()), "tp": tp, "fp": fp, "fn": fn,
             "accuracy": met["accuracy"], "accuracy_ci95": met.get("accuracy_ci95"),
             "kappa": met["kappa_vs_human"], "kappa_ci95": met.get("kappa_vs_human_ci95"),
             "macro_f1": met["macro_f1"], "macro_f1_ci95": met.get("macro_f1_ci95"),
             "balanced_accuracy": bal, "balanced_accuracy_ci95": met.get("balanced_accuracy_ci95"),
             "precision_regret": tp / (tp + fp) if tp + fp else np.nan, "precision_regret_ci95": met.get("precision_REGRET_ci95"),
             "recall_regret": tp / (tp + fn) if tp + fn else np.nan, "recall_regret_ci95": met.get("recall_REGRET_ci95"),
             "f1_regret": met["f1_REGRET"], "f1_regret_ci95": met.get("f1_REGRET_ci95")}
        if subset_name == "all agreed" and ref == "gold" and len(d) == len(d0):
            # Pool weights: stratum pool size / number of gold-labelled (agreed) reviews in that stratum,
            # the same weights the prevalence estimate uses (Reviewer 3, third round, comment 1).
            d = pool_frame(d, k)
            wa, wp, wr, wf = weighted_metrics(d[ref], d[k], d["weight"])
            r.update({"w_accuracy": wa, "w_precision": wp, "w_recall": wr, "w_f1": wf, **strat_boot(d, k, ref)})
        rows.append(r)
    return pd.DataFrame(rows)


def supplementary_tables(m, g, met):
    """Third-round requests of Reviewer 3: weighting procedure, minority-class intervals, four-way label."""
    # (a) stratum table: everything needed to reproduce the pool-level estimates by hand
    st = m.groupby("stratum").agg(N_pool=("stratum_N", "first"), n_coded=("review_uid", "size"))
    pf = pool_frame(g)
    st["n_attention_checks_used"] = pf[pf.gold_origin == "attention_check_S1"].groupby("stratum").size()
    st["n_gold"] = pf.groupby("stratum").size()
    st["n_regret_gold"] = pf[pf.gold == "REGRET"].groupby("stratum").size()
    st = st.fillna(0).astype({"n_gold": int, "n_regret_gold": int, "n_attention_checks_used": int})
    st["n_random_coded"] = m[m.random].groupby("stratum").size().reindex(st.index).fillna(0).astype(int)
    st["weight_w_h"] = st.N_pool / st.n_gold
    st["regret_rate_in_stratum"] = st.n_regret_gold / st.n_gold
    st["estimated_regret_in_pool"] = st.N_pool * st.regret_rate_in_stratum
    tot = st.sum(numeric_only=True); tot[["weight_w_h", "regret_rate_in_stratum"]] = np.nan
    st.loc["TOTAL"] = tot
    st.loc["TOTAL", "regret_rate_in_stratum"] = st.loc["TOTAL", "estimated_regret_in_pool"] / st.loc["TOTAL", "N_pool"]
    st.to_csv(OUT / "H_stratum_weights.csv")
    # (b) Table 5 with every interval, plus macro-F1 and balanced accuracy (supplementary table)
    cols = ["system", "name", "n", "n_regret_ref", "n_regret_pred", "tp", "fp", "fn",
            "accuracy", "accuracy_ci95", "kappa", "kappa_ci95",
            "precision_regret", "precision_regret_ci95", "recall_regret", "recall_regret_ci95",
            "f1_regret", "f1_regret_ci95", "macro_f1", "macro_f1_ci95", "balanced_accuracy", "balanced_accuracy_ci95",
            "w_accuracy", "w_accuracy_ci95", "w_precision", "w_precision_ci95", "w_recall", "w_recall_ci95",
            "w_f1", "w_f1_ci95"]
    sel = met[(met.reference == "gold") & (met.subset == "all agreed")]
    sel[[c for c in cols if c in sel]].to_csv(OUT / "J_metrics_with_all_intervals.csv", index=False)
    # per-stratum confusion counts: with H_stratum_weights.csv these reproduce every pool-weighted metric
    rows = []
    for k in sel.system:
        for h, d in pool_frame(m.dropna(subset=["gold", k]), k).groupby("stratum"):
            yt, yp = d.gold == "REGRET", d[k] == "REGRET"
            rows.append({"system": k, "stratum": h, "n_gold": len(d), "tp": int((yt & yp).sum()), "fp": int((~yt & yp).sum()),
                         "fn": int((yt & ~yp).sum()), "tn": int((~yt & ~yp).sum())})
    pd.DataFrame(rows).to_csv(OUT / "K_confusion_by_stratum.csv", index=False)
    # (c) four-way primary emotion versus the binary forced choice
    from statsmodels.stats.proportion import proportion_confint
    rows = []
    for sub, d in [("random 200", m[m.random]), ("all 385", m)]:
        a = d.dropna(subset=["gold_primary"])
        for lab, cnt in a.gold_primary.value_counts().items():
            lo, hi = proportion_confint(cnt, len(a), method="wilson")
            rows.append({"subset": sub, "n_primary_agreed": len(a), "primary_emotion": lab, "n": int(cnt),
                         "pct": 100 * cnt / len(a), "ci95": f"[{100*lo:.1f}, {100*hi:.1f}]"})
        ct = pd.crosstab(d.gold_primary, d.gold).reset_index().assign(subset=sub)
        ct.to_csv(OUT / f"I_primary_by_forced_choice_{sub.replace(' ', '_')}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "I_fourway_primary_emotion.csv", index=False)


def BC_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B3 labels for reviews outside the 372: the same TF-IDF model, trained on all 372 gold reviews."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    if test.empty:
        return np.array([], dtype=object)
    mdl = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                        LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
    mdl.fit(train["reviews.text"], train["gold"])
    return mdl.predict(test["reviews.text"])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    global POOL_EXTRA
    m = build_table(); res = {}
    m.to_csv(OUT / "merged_round2.csv", index=False)
    g = m.dropna(subset=["gold"])
    pool = pd.read_csv(C.DATA_DIR / "pool.csv")
    POOL_EXTRA = build_table(attention_s1=True)
    POOL_EXTRA.to_csv(OUT / "merged_attention_S1.csv", index=False)
    g = g.assign(weight=gold_weights(pd.concat([g, POOL_EXTRA.dropna(subset=["gold"])]))[: len(g)].values)
    g[["review_uid", "gold", "gold_source", "gold_primary", "stratum", "weight", "gold_origin"]].rename(
        columns={"gold": "forced_choice", "gold_primary": "primary_emotion"}).merge(
        pool[["review_uid", "reviews.text"]], on="review_uid").to_csv(C.DATA_DIR / "gold_labels_round2.csv", index=False)

    # 1. inter-annotator agreement
    ia = []
    for f in ["forced_choice", "primary_emotion", "cf_focus"]:
        for sub, d in [("all 385", m), ("random 200", m[m.random])]:
            x, y = d[f"{f}_h1"].astype(str).tolist(), d[f"{f}_h2"].astype(str).tolist()
            ia.append({"field": f, "subset": sub, "n": len(d), "pct_agree": 100 * np.mean(np.array(x) == np.array(y)),
                       "kappa": cohen_kappa_score(x, y), "alpha": krippendorff_nominal(x, y)})
    for dd in DIMS:
        ia.append({"field": dd, "subset": "all 385", "n": len(m),
                   "spearman": stats.spearmanr(m[f"{dd}_h1"], m[f"{dd}_h2"]).statistic,
                   "kappa": cohen_kappa_score(m[f"{dd}_h1"], m[f"{dd}_h2"], weights="quadratic")})
    pd.DataFrame(ia).to_csv(OUT / "A_interannotator.csv", index=False)
    res["gold"] = {"n_385": len(m), "n_gold": len(g), "n_regret": int((g.gold == "REGRET").sum()),
                   "n_disagree": int(m.gold.isna().sum()), "sources": g.gold_source.value_counts().to_dict(),
                   "gold_primary_counts": m.gold_primary.value_counts().to_dict()}

    # 2. prevalence
    rnd = g[g.random]; k = int((rnd.gold == "REGRET").sum()); lo, hi = S.wilson_ci(k, len(rnd))
    N = len(pool)
    by = pool_frame(g).groupby("stratum").apply(lambda d: pd.Series({"N": d["stratum_N"].iloc[0], "p": (d.gold == "REGRET").mean(), "n": len(d)}))
    p_w = float((by.N * by.p).sum() / by.N.sum())
    # interval: bootstrap that resamples reviews within each stratum (same scheme as the weighted metrics)
    pf = pool_frame(g); rng = np.random.default_rng(C.SEED); bs = []
    grp = {h: (d.gold == "REGRET").values for h, d in pf.groupby("stratum")}
    for _ in range(2000):
        bs.append(sum(by.N[h] * rng.choice(v, len(v), replace=True).mean() for h, v in grp.items()) / by.N.sum())
    # S0 has no regret among its annotated reviews, so it adds no sampling variance; report the pool share
    # if S0's regret share were at the upper Wilson bound for 0 of n
    s0_hi = S.wilson_ci(0, int(by.n["S0_rest"]))[1] if "S0_rest" in by.index else np.nan
    p_s0hi = float((by.N * by.p.where(by.index != "S0_rest", s0_hi)).sum() / by.N.sum())
    res["prevalence"] = {"random_n": len(rnd), "random_regret": k, "random_pct": 100 * k / len(rnd),
                         "random_ci95": [100 * lo, 100 * hi], "weighted_pct": 100 * p_w,
                         "weighted_ci95": [100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))],
                         "weighted_pct_if_S0_at_wilson_upper": 100 * p_s0hi, "S0_wilson_upper_pct": 100 * s0_hi,
                         "random_primary_counts": m[m.random].gold_primary.value_counts().to_dict()}
    rows = []
    for sys in FULL + ["B1_lexicon"]:
        kk = int((m[m.random][sys] == "REGRET").sum()); rows.append({"system": sys, "random_200_pct": 100 * kk / m.random.sum()})
    pd.DataFrame(rows).to_csv(OUT / "B_prevalence_random200_systems.csv", index=False)

    # 3. supervised baseline (TF-IDF, CV on the gold set) + optional RoBERTa
    import baselines_classic as BC
    tf = BC.tfidf_cv(g.rename(columns={"gold": "forced_choice"}).merge(pool[["review_uid"]], on="review_uid")
                      .assign(**{"reviews.text": lambda d: d["reviews.text"]}) if "reviews.text" in g else
                      g.rename(columns={"gold": "forced_choice"}).merge(pool[["review_uid", "reviews.text"]], on="review_uid"))
    m = m.merge(tf[["review_uid", "label"]].rename(columns={"label": "B3_tfidf_logreg_cv"}), on="review_uid", how="left")
    POOL_EXTRA = POOL_EXTRA.assign(B3_tfidf_logreg_cv=BC_fit_predict(
        g.drop(columns="reviews.text", errors="ignore").merge(pool[["review_uid", "reviews.text"]], on="review_uid"),
        POOL_EXTRA.drop(columns="reviews.text", errors="ignore").merge(pool[["review_uid", "reviews.text"]], on="review_uid")))
    extra = ["B3_tfidf_logreg_cv"]
    for k, f in [("B5_roberta_cv", "gold_roberta_cv.csv"), ("B6_roberta_silver", "gold_roberta_silver.csv")]:
        p = C.RESULTS_DIR / "05_baselines" / f
        if p.exists():
            r = pd.read_csv(p)
            if set(r["review_uid"]) >= set(g["review_uid"]) - set():
                m = m.merge(r[["review_uid", "label"]].rename(columns={"label": k}), on="review_uid", how="left"); extra.append(k)

    # 4. system metrics
    systems = FULL + list(SUBSET) + ["B1_lexicon"] + extra + ["AUTO_A", "AUTO_B"]
    tabs = [system_metrics(m, FULL + ["B1_lexicon"] + extra, "gold", "all agreed"),
            system_metrics(m[m.random], FULL + ["B1_lexicon"] + extra, "gold", "random agreed"),
            system_metrics(m[m.review_uid.isin(m.dropna(subset=["C5_speculative_4o_drafter"]).review_uid)] if "C5_speculative_4o_drafter" in m else m.iloc[:0],
                           ["C3_rag_full", "C4_speculative"] + list(SUBSET) + ["R0_original_replication"], "gold", "285 round-1 reviews"),
            system_metrics(m[m.review_uid.isin(m.dropna(subset=["AUTO_A"]).review_uid)] if "AUTO_A" in m else m.iloc[:0],
                           ["AUTO_A", "AUTO_B", "C0_labels_only", "C3_rag_full"], "gold", "285 round-1 reviews (automated annotators)")]
    for ref in ["forced_choice_h1", "forced_choice_h2"]:
        tabs.append(system_metrics(m, FULL + ["B1_lexicon"], ref, "all 385, single annotator"))
    met = pd.concat(tabs, ignore_index=True)
    met.to_csv(OUT / "C_system_metrics.csv", index=False)
    supplementary_tables(m, g, met)

    # 5. McNemar on correctness vs gold
    mc = []
    pairs = [("C0_labels_only", "C1_definitions"), ("C1_definitions", "C3_rag_full"), ("C2_rag_only", "C3_rag_full"),
             ("C0_labels_only", "C2_rag_only"), ("C3_rag_full", "C4_speculative"), ("C3_rag_full", "C5_speculative_4o_drafter"),
             ("C4_speculative", "C5_speculative_4o_drafter"), ("R0_original_replication", "C3_rag_full"),
             ("R2_newprompt_origKB", "C3_rag_full"), ("C1_definitions", "B3_tfidf_logreg_cv")] + \
            ([("C1_definitions", "B5_roberta_cv")] if "B5_roberta_cv" in m else [])
    for a_, b_ in pairs:
        if a_ not in m or b_ not in m:
            continue
        d = g.merge(m[["review_uid", a_, b_]].rename(columns={a_: "_a", b_: "_b"}), on="review_uid").dropna(subset=["_a", "_b"])
        ca, cb = d["_a"] == d.gold, d["_b"] == d.gold
        t = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
        mc.append({"A": a_, "B": b_, "n": len(d), "acc_A": ca.mean(), "acc_B": cb.mean(), "only_A_right": t[0][1],
                   "only_B_right": t[1][0], "p": mcnemar(t, exact=True).pvalue})
    pd.DataFrame(mc).to_csv(OUT / "D_mcnemar_correctness.csv", index=False)

    # 6. composite score vs human label, human vs LLM dimensions, data-driven weights
    d = g.dropna(subset=["rds_paper"])
    rep = S.two_group_report(d.loc[d.gold == "REGRET", "rds_paper"].values, d.loc[d.gold == "DISAPPOINTMENT", "rds_paper"].values)
    res["composite_by_human"] = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)) for k, v in rep.items()}
    y = (d.gold == "REGRET").astype(int).values
    res["composite_auc"] = {"paper_weights": roc_auc_score(y, d.rds_paper),
                            **{f"llm_{x}": roc_auc_score(y, d[f"rds_{x}"]) for x in DIMS}}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X = d[[f"rds_{x}" for x in DIMS]].values
    pcv = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED), method="predict_proba")[:, 1]
    fit = LogisticRegression(max_iter=1000).fit((X - X.mean(0)) / X.std(0), y)
    res["composite_auc"]["learned_cv"] = roc_auc_score(y, pcv)
    res["learned_std_coefs"] = dict(zip(DIMS, fit.coef_[0].round(3).tolist()))
    res["human_vs_llm_dims_spearman"] = {x: stats.spearmanr(m[f"human_{x}"], m[f"rds_{x}"], nan_policy="omit").statistic for x in DIMS}
    hr = m.merge(pool[["review_uid", "reviews.rating"]], on="review_uid", suffixes=("", "_r"))
    res["human_dims_vs_rating_spearman"] = {x: stats.spearmanr(hr[f"human_{x}"], hr["reviews.rating"]).statistic for x in DIMS}

    # 7. counterfactual focus: GPT-4o vs humans
    cf = []
    for ref in ["cf_focus_h1", "cf_focus_h2"]:
        dd = m.dropna(subset=["cf_focus", ref])
        cf.append({"pair": f"GPT-4o vs {ref}", "n": len(dd), "pct_agree": 100 * (dd.cf_focus == dd[ref]).mean(),
                   "kappa": cohen_kappa_score(dd.cf_focus, dd[ref])})
    agreed_cf = m[m.cf_focus_h1 == m.cf_focus_h2]
    tab = pd.crosstab(agreed_cf.dropna(subset=["gold"]).gold, agreed_cf.dropna(subset=["gold"]).cf_focus_h1)
    tab.to_csv(OUT / "E_human_cf_by_human_label.csv")
    ch = stats.chi2_contingency(tab.values)
    res["cf_human"] = {"agreement": cf, "table": tab.to_dict(), "chi2": ch[0], "p": ch[1],
                       "cramers_v": float(np.sqrt(ch[0] / (tab.values.sum() * (min(tab.shape) - 1))))}

    # 8. error analysis of C1 and C3
    ex = g.merge(pool[["review_uid", "reviews.text"]], on="review_uid", suffixes=("", "_t"))
    txt = "reviews.text" if "reviews.text" in ex else "reviews.text_t"
    err = []
    for sys in ["C1_definitions", "C3_rag_full", "C4_speculative"]:
        for kind, cond in [("false_positive", (ex[sys] == "REGRET") & (ex.gold == "DISAPPOINTMENT")),
                           ("false_negative", (ex[sys] == "DISAPPOINTMENT") & (ex.gold == "REGRET"))]:
            for _, r in ex[cond].iterrows():
                err.append({"system": sys, "error": kind, "review_uid": r.review_uid, "text": r[txt],
                            "human_primary": r.gold_primary, "stratum": r.stratum})
    pd.DataFrame(err).to_csv(OUT / "F_errors_C1_C3_C4.csv", index=False)
    res["errors"] = pd.DataFrame(err).groupby(["system", "error"]).size().to_dict() if err else {}
    res["errors"] = {f"{a}|{b}": int(v) for (a, b), v in res["errors"].items()}
    # length effect on C3 errors
    ex["len"] = ex[txt].astype(str).str.split().str.len()
    ex["c3_err"] = ex["C3_rag_full"] != ex.gold
    res["c3_error_rate_by_length"] = ex.groupby(pd.cut(ex["len"], [0, 10, 25, 50, 1000]), observed=True)["c3_err"].mean().round(3).astype(float).to_dict()
    res["c3_error_rate_by_length"] = {str(k): v for k, v in res["c3_error_rate_by_length"].items()}

    json.dump(res, open(OUT / "G_summary.json", "w"), indent=1, default=str)
    m.to_csv(OUT / "merged_round2.csv", index=False)
    return res, met


# --------------------------------------------------------------------------
def make_figures(m: pd.DataFrame, met: pd.DataFrame, res: dict):
    """Paper figures that use the human labels: R8 (accuracy), R3 (prevalence + human band), R5 (composite)."""
    import json as _json, tempfile
    from pathlib import Path
    import matplotlib.pyplot as plt
    import figures as F
    a = met[(met.subset == "all agreed") & (met.reference == "gold")].set_index("system")
    order = [k for k in ["C0_labels_only", "C1_definitions", "C2_rag_only", "C3_rag_full", "C4_speculative",
                         "R0_original_replication", "B1_lexicon", "B3_tfidf_logreg_cv", "B5_roberta_cv"] if k in a.index]
    a = a.loc[order].reset_index()
    ci = lambda s_: tuple(float(v) for v in str(s_).strip("[]").split(","))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True); y = np.arange(len(a))[::-1]
    for ax, (col, cc, title) in zip(axes, [("kappa", "kappa_ci95", "Cohen's κ with human labels"), ("f1_regret", "f1_regret_ci95", "F1 for regret")]):
        lo = np.array([ci(v)[0] for v in a[cc]]); hi = np.array([ci(v)[1] for v in a[cc]])
        colr = [F.BLUE if s_ in ("C0_labels_only", "C1_definitions") else (F.ORANGE if s_ in ("C2_rag_only", "C3_rag_full", "C4_speculative", "R0_original_replication") else F.MUTED) for s_ in a.system]
        ax.hlines(y, lo, hi, color=colr, lw=2.2); ax.scatter(a[col], y, color=colr, s=28, zorder=3)
        for yy, v in zip(y, a[col]):
            ax.text(v, yy + 0.25, f"{v:.2f}", ha="center", fontsize=7.5, color=F.INK2)
        ax.set_xlim(-0.05, 1); ax.set_title(title, loc="left", color=F.INK, fontsize=9.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels(a.name.str.replace(" (CV)", ""), fontsize=8.5)
    fig.text(0.01, -0.03, f"{int(a.n.iloc[0])} reviews on which both human annotators agree ({int(a.n_regret_ref.iloc[0])} regret). Bars: bootstrap 95% CI. Blue: no retrieval; orange: retrieval-based.", fontsize=7.5, color=F.INK2)
    F._save(fig, "R8_accuracy_vs_human.png")
    # R3 with the human estimate as a band
    t = Path(tempfile.mkdtemp()); d = pd.read_csv(F.A / "A_prevalence_by_system.csv")
    d["description"] = d["system"].map({"C0_labels_only": "C0 GPT-4o, label names only", "C1_definitions": "C1 GPT-4o + definitions",
        "C2_rag_only": "C2 RAG, no definitions", "C3_rag_full": "C3 RAG + definitions", "C4_speculative": "C4 Speculative RAG (GPT-4o-mini drafter)",
        "R0_original_replication": "R0 Original pipeline (replication)", "B1_lexicon": "B1 Regret-cue lexicon"}).fillna(d["description"])
    d.to_csv(t / "A_prevalence_by_system.csv", index=False)
    _json.dump({"pct": res["prevalence"]["weighted_pct"], "ci95": res["prevalence"]["weighted_ci95"]}, open(t / "B_human_prevalence.json", "w"))
    old = F.A; F.A = t
    try:
        F.fig_prevalence()
    finally:
        F.A = old
    # R5: composite by GPT-4o label and by human label
    pm = pd.read_csv(F.IN / "pool_master.csv"); g = pd.read_csv(C.DATA_DIR / "gold_labels_round2.csv")[["review_uid", "forced_choice"]]
    panels = [("(a) GPT-4o label (C3), all 2019", pm, "C3_rag_full__label"),
              (f"(b) Human label (both agree), n = {len(g)}", pm.merge(g, on="review_uid"), "forced_choice")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True); r = np.random.default_rng(0)
    for ax, (title, df, col) in zip(axes, panels):
        for i, L in enumerate(["REGRET", "DISAPPOINTMENT"]):
            v = df.loc[df[col] == L, "rds_paper"].dropna()
            ax.boxplot(v, positions=[i], widths=.45, showfliers=False, patch_artist=True, boxprops=dict(fc="white", ec=F.COL[L], lw=1.2),
                       medianprops=dict(color=F.COL[L], lw=2), whiskerprops=dict(color=F.COL[L]), capprops=dict(color=F.COL[L]))
            ax.scatter(i + r.uniform(-.17, .17, len(v)), v, s=5 if len(v) > 300 else 9, color=F.COL[L], alpha=.35, lw=0)
            ax.text(i, 6.4, f"n = {len(v)}\nmean {v.mean():.2f}", ha="center", fontsize=7.5, color=F.INK2)
        ax.axhline(0, color=F.MUTED, lw=1, ls="--"); ax.set_xticks([0, 1]); ax.set_xticklabels(["REGRET", "DISAPPOINTMENT"])
        ax.set_title(title, loc="left", color=F.INK, fontsize=9); ax.set_ylim(-6.5, 7.8)
    axes[0].set_ylabel("Composite score (GPT-4o ratings)")
    F._save(fig, "R5_rds_by_llm_and_human_label.png")


if __name__ == "__main__":
    res, met = run()
    make_figures(pd.read_csv(OUT / "merged_round2.csv"), met, res)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(json.dumps(res, indent=1, default=str)[:6000])
    cols = ["subset", "reference", "name", "n", "n_regret_ref", "n_regret_pred", "accuracy", "kappa", "kappa_ci95", "precision_regret",
            "recall_regret", "f1_regret", "f1_regret_ci95", "macro_f1", "balanced_accuracy", "w_accuracy", "w_f1"]
    print(met[[c for c in cols if c in met]].round(3).to_string())
    print(pd.read_csv(OUT / "D_mcnemar_correctness.csv").round(4).to_string())
    print(pd.read_csv(OUT / "A_interannotator.csv").round(3).to_string())
