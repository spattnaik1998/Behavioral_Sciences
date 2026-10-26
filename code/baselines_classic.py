"""
Stage 3a - conventional (non-LLM) baselines. Run on CPU, no API.

Answers R2.1 / R4.5 (conventional baselines) and R4.6 (does the system
identify regret/disappointment rather than general sentiment?).

  * Regret-cue lexicon: transparent rule-based classifier built from the
    self-blame / behaviour-counterfactual markers in Zeelenberg et al. (1998).
  * VADER: general-purpose sentiment polarity. If VADER cannot separate human
    REGRET from DISAPPOINTMENT reviews, polarity is not what the task measures.
  * TF-IDF + logistic regression: supervised classifier, stratified 5-fold
    cross-validation on the human gold set (needs data/gold_labels.csv).

Outputs: results/05_baselines/
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config as C

OUT = C.RESULTS_DIR / "05_baselines"

# First-person self-blame / behaviour-focused counterfactual cues
REGRET_PATTERNS = [
    r"\bi (should|shoulda|should've|should have)\b",
    r"\bi (shouldn'?t|should not) have\b",
    r"\bi wish i (had|hadn'?t|would have|would'?ve|did|didn'?t|bought|got|went|spent|read|listened)\b",
    r"\bif only i\b",
    r"\b(my|own) (fault|mistake|bad|error)\b",
    r"\bi regret\b", r"\bregret (buying|purchasing|ordering|getting|this)\b",
    r"\bshould have (listened|read|checked|known|bought|gone|stuck|spent|paid|returned)\b",
    r"\blesson learned\b", r"\blearned my lesson\b",
    r"\bi (was|am) (stupid|dumb|foolish)\b",
    r"\b(next time|from now on) i('ll| will)\b",
    r"\bi (made|make) (a|the) (mistake|wrong choice)\b",
    r"\bshould'?ve (bought|gone|stuck|listened|read|spent|got)\b",
    r"\bwish(ed)? i (had )?(bought|gone|stuck|spent|listened)\b",
]
_RX = re.compile("|".join(REGRET_PATTERNS), re.I)


def lexicon_predict(texts: pd.Series) -> pd.DataFrame:
    hits = texts.astype(str).apply(lambda t: [m.group(0) for m in _RX.finditer(t)])
    return pd.DataFrame({"lexicon_hits": hits.apply(lambda h: "|".join(h)),
                         "lexicon_n_hits": hits.apply(len),
                         "lexicon_label": np.where(hits.apply(len) > 0, "REGRET", "DISAPPOINTMENT")},
                        index=texts.index)


def vader_scores(texts: pd.Series) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    s = texts.astype(str).apply(sia.polarity_scores)
    return pd.DataFrame({"vader_compound": s.apply(lambda d: d["compound"]),
                         "vader_neg": s.apply(lambda d: d["neg"]),
                         "vader_pos": s.apply(lambda d: d["pos"])}, index=texts.index)


def run_pool(dataset: str = "pool") -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    f = C.DATA_DIR / ("pool.csv" if dataset == "pool" else "external_sample.csv")
    df = pd.read_csv(f)
    res = pd.concat([df[["review_uid", "reviews.rating"]], lexicon_predict(df["reviews.text"]),
                     vader_scores(df["reviews.text"])], axis=1)
    res.to_csv(OUT / f"{dataset}_lexicon_vader.csv", index=False)
    from stats_utils import wilson_ci
    k = int((res["lexicon_label"] == "REGRET").sum()); n = len(res)
    lo, hi = wilson_ci(k, n)
    summary = pd.DataFrame([{"dataset": dataset, "n": n, "lexicon_regret_n": k,
                             "lexicon_regret_pct": round(100 * k / n, 2),
                             "ci95": f"[{100*lo:.1f}%, {100*hi:.1f}%]",
                             "vader_compound_mean": round(res["vader_compound"].mean(), 3),
                             "spearman_vader_vs_rating": round(res[["vader_compound", "reviews.rating"]].corr("spearman").iloc[0, 1], 3)}])
    summary.to_csv(OUT / f"{dataset}_lexicon_vader_summary.csv", index=False)
    return res


def tfidf_cv(gold: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold predictions on the gold set (y = forced_choice)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline

    g = gold.dropna(subset=["forced_choice"]).reset_index(drop=True)
    y = g["forced_choice"].values
    pred = np.empty(len(g), dtype=object); prob = np.full(len(g), np.nan); fold = np.zeros(len(g), int)
    n_splits = min(n_splits, pd.Series(y).value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    for k, (tr, te) in enumerate(skf.split(g, y)):
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                          LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
        m.fit(g.loc[tr, "reviews.text"], y[tr])
        pred[te] = m.predict(g.loc[te, "reviews.text"])
        prob[te] = m.predict_proba(g.loc[te, "reviews.text"])[:, list(m.classes_).index("REGRET")]
        fold[te] = k
    out = pd.DataFrame({"review_uid": g["review_uid"], "label": pred, "p_regret_logprob": prob, "fold": fold,
                        "condition": "B3_tfidf_logreg_cv"})
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "gold_tfidf_logreg_cv.csv", index=False)
    return out


if __name__ == "__main__":
    for ds in ["pool", "external"]:
        r = run_pool(ds)
        print(ds, r["lexicon_label"].value_counts().to_dict())
    print(pd.read_csv(OUT / "pool_lexicon_vader_summary.csv").T)
