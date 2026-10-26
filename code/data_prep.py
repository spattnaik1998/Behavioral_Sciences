"""
Stage 1 - data preparation and dataset description.

Answers:  R1.1 (how many records), R4.4 (source, size, categories, language,
          preprocessing, class distribution), R2.2/R3.4 (size of the pool the
          100 reviews were drawn from).

Outputs (results/01_data/):
  data_flow.csv               record counts after every preprocessing step
  pool_description.csv        rating / category / length / language summary
  pool_vs_sample.csv          pool vs the paper's 100-review sample
  category_by_rating.csv
  external_description.csv    the same for the external (Kindle/Fire/Echo) set
Data written to data/:
  pool.csv                    all 2,019 preprocessed reviews, with review_uid
  original_sample_100.csv     exact rebuild of the paper's random sample
  external_sample.csv         500 low-rated reviews from the external dataset
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C

OUT = C.RESULTS_DIR / "01_data"


# --------------------------------------------------------------------------
def _language_detector():
    try:
        from lingua import LanguageDetectorBuilder
        return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    except Exception:            # lingua not installed -> language column left blank
        return None


def _detect_language(texts: pd.Series) -> pd.Series:
    det = _language_detector()
    if det is None:
        return pd.Series(["unknown"] * len(texts), index=texts.index)
    out = []
    for t in texts:
        lang = det.detect_language_of(str(t))
        out.append(lang.name.lower() if lang is not None else "undetermined")
    return pd.Series(out, index=texts.index)


# --------------------------------------------------------------------------
def build_pool(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicates the original notebook's preprocessing exactly (cell 10)."""
    raw = pd.read_csv(C.main_raw_path(), low_memory=False)
    flow = [("Raw file (Datafiniti May19)", len(raw))]

    keep = ["id", "reviews.rating", "primaryCategories", "reviews.text", "reviews.date"]
    extra = [c for c in ["name", "brand", "reviews.title", "reviews.didPurchase",
                         "reviews.doRecommend", "reviews.numHelpful"] if c in raw.columns]
    df = raw[keep + extra].copy()

    df = df[df["reviews.rating"] <= C.RATING_MAX]
    flow.append((f"Rating <= {C.RATING_MAX}", len(df)))
    df = df.dropna(subset=["reviews.text"])
    flow.append(("Non-empty review text", len(df)))
    df = df.drop_duplicates(subset=["reviews.text"])
    flow.append(("Duplicate texts removed", len(df)))
    df["reviews.date"] = pd.to_datetime(df["reviews.date"], errors="coerce")
    df = df.reset_index(drop=True)
    df.insert(0, "review_uid", [f"P{i:04d}" for i in range(len(df))])

    flow_df = pd.DataFrame(flow, columns=["step", "n_records"])
    if write:
        C.ensure_dirs(); OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(C.DATA_DIR / "pool.csv", index=False)
        flow_df.to_csv(OUT / "data_flow.csv", index=False)
    return df, flow_df


def original_sample(pool: pd.DataFrame) -> pd.DataFrame:
    """Exact rebuild of df_analysis.sample(n=100, random_state=42) from the paper's notebook.
    Row selection depends only on row order and count, both of which are reproduced."""
    return pool.sample(n=min(C.ORIGINAL_SAMPLE_N, len(pool)), random_state=C.SEED)


def describe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    txt = df["reviews.text"].astype(str)
    words = txt.str.split().str.len()
    rows = {
        "dataset": name,
        "n_reviews": len(df),
        "n_products": df["id"].nunique() if "id" in df else np.nan,
        "date_min": str(pd.to_datetime(df["reviews.date"], errors="coerce").min())[:10],
        "date_max": str(pd.to_datetime(df["reviews.date"], errors="coerce").max())[:10],
        "words_mean": round(words.mean(), 1),
        "words_median": float(words.median()),
        "words_p10": float(words.quantile(.1)),
        "words_p90": float(words.quantile(.9)),
    }
    for r in sorted(df["reviews.rating"].dropna().unique()):
        rows[f"rating_{int(r)}_n"] = int((df["reviews.rating"] == r).sum())
    if "language" in df:
        for lang, n in df["language"].value_counts().items():
            rows[f"lang_{lang}_n"] = int(n)
    if "reviews.didPurchase" in df:
        rows["didPurchase_true_n"] = int((df["reviews.didPurchase"] == True).sum())  # noqa: E712
        rows["didPurchase_missing_n"] = int(df["reviews.didPurchase"].isna().sum())
    return pd.DataFrame([rows])


def build_external(pool: pd.DataFrame, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Low-rated reviews of a different product line (Kindle / Fire / Echo devices)."""
    if not C.EXTERNAL_RAW.exists():
        print(f"[external] {C.EXTERNAL_RAW} not found - skipped")
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.read_csv(C.EXTERNAL_RAW, low_memory=False)
    flow = [("Raw file (Datafiniti 1429_1)", len(raw))]
    df = raw[raw["reviews.rating"] <= C.RATING_MAX].copy(); flow.append((f"Rating <= {C.RATING_MAX}", len(df)))
    df = df.dropna(subset=["reviews.text"]); flow.append(("Non-empty review text", len(df)))
    df = df.drop_duplicates(subset=["reviews.text"]); flow.append(("Duplicate texts removed", len(df)))
    df = df[~df["reviews.text"].isin(set(pool["reviews.text"]))]; flow.append(("Overlap with main pool removed", len(df)))
    df["primaryCategories"] = "Amazon devices (Kindle/Fire/Echo)"
    df = df.reset_index(drop=True)
    df.insert(0, "review_uid", [f"X{i:04d}" for i in range(len(df))])
    sample = df.sample(n=min(C.EXTERNAL_SAMPLE_N, len(df)), random_state=C.SEED)
    flow.append((f"Random sample (seed={C.SEED})", len(sample)))
    keep = ["review_uid", "id", "name", "reviews.rating", "primaryCategories", "reviews.text", "reviews.date"]
    sample = sample[keep]
    flow_df = pd.DataFrame(flow, columns=["step", "n_records"])
    if write:
        sample.to_csv(C.DATA_DIR / "external_sample.csv", index=False)
        flow_df.to_csv(OUT / "external_data_flow.csv", index=False)
    return sample, flow_df


def run() -> dict:
    C.ensure_dirs(); OUT.mkdir(parents=True, exist_ok=True)
    pool, flow = build_pool()
    pool["language"] = _detect_language(pool["reviews.text"])
    pool.to_csv(C.DATA_DIR / "pool.csv", index=False)

    s100 = original_sample(pool)
    s100.to_csv(C.DATA_DIR / "original_sample_100.csv", index=False)

    desc = pd.concat([describe(pool, "pool (rating<=3, cleaned)"),
                      describe(s100, "paper sample (n=100, seed=42)")], ignore_index=True)
    desc.T.to_csv(OUT / "pool_vs_sample.csv", header=False)

    cats = pd.crosstab(pool["primaryCategories"], pool["reviews.rating"], margins=True)
    cats.to_csv(OUT / "category_by_rating.csv")
    cat_cmp = pd.DataFrame({
        "pool_n": pool["primaryCategories"].value_counts(),
        "pool_pct": (pool["primaryCategories"].value_counts(normalize=True) * 100).round(1),
        "sample_n": s100["primaryCategories"].value_counts(),
        "sample_pct": (s100["primaryCategories"].value_counts(normalize=True) * 100).round(1),
    }).fillna(0)
    cat_cmp.to_csv(OUT / "category_pool_vs_sample.csv")

    prods = (pool.groupby(["primaryCategories", "name"]).size()
             .sort_values(ascending=False).rename("n_reviews").reset_index())
    prods.head(25).to_csv(OUT / "top_products_in_pool.csv", index=False)

    ext, ext_flow = build_external(pool)
    if len(ext):
        ext["language"] = _detect_language(ext["reviews.text"])
        ext.to_csv(C.DATA_DIR / "external_sample.csv", index=False)
        describe(ext, "external sample (devices)").T.to_csv(OUT / "external_description.csv", header=False)

    # The 5,000-row file the paper's Section 3.1 describes (for correcting the text)
    note = {}
    if C.ORIGINAL_5000_RAW.exists():
        f5 = pd.read_csv(C.ORIGINAL_5000_RAW, low_memory=False)
        note = {"file": C.ORIGINAL_5000_RAW.name, "rows": len(f5), "cols": f5.shape[1],
                "rating_le3": int((f5["reviews.rating"] <= 3).sum())}
    summary = {
        "data_flow": flow.to_dict("records"),
        "pool_n": len(pool),
        "original_sample_n": len(s100),
        "original_sample_rating_counts": s100["reviews.rating"].value_counts().sort_index().to_dict(),
        "external_sample_n": len(ext),
        "file_described_in_paper_section_3_1": note,
    }
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
