"""
Runners for every LLM task. Each returns a tidy DataFrame (one row per review)
and never silently imputes: failures are kept as NaN with an `error` flag
(the original notebook coded failed RDS calls as 0.0 = "neutral").
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config as C
import prompts as P
from llm import label_probability, mean_token_prob, parse_json, yes_probability

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_):
        return x


def _norm_label(x) -> str | None:
    if not isinstance(x, str):
        return None
    x = x.strip().upper()
    if x.startswith("REG"):
        return "REGRET"
    if x.startswith("DIS"):
        return "DISAPPOINTMENT"
    return None


def _pmap(fn, items, desc):
    """Parallel map. Stops cleanly (cancelling queued work) when the budget or credit runs out;
    everything finished before that is already in the cache."""
    from llm import BudgetExceeded, OutOfCredits
    out = [None] * len(items)
    ex = ThreadPoolExecutor(max_workers=C.MAX_WORKERS)
    try:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        done = 0
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            out[futs[f]] = f.result()
            done += 1
    except (BudgetExceeded, OutOfCredits):
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        ex.shutdown(wait=True)
    return out


# ==========================================================================
# Retrieval for a set of reviews (query = review text, as in the paper)
# ==========================================================================
def retrieve_for(client, kb, df: pd.DataFrame, k: int, name: str) -> dict[str, list[int]]:
    q = client.embed(df["reviews.text"].astype(str).tolist(), name=f"reviews_{name}")
    idx = kb.search(q, k)
    return {uid: list(map(int, row)) for uid, row in zip(df["review_uid"], idx)}


# ==========================================================================
# Classification conditions C0-C3 and the original-prompt replication
# ==========================================================================
def run_classification(client, df: pd.DataFrame, condition: str, retrieval: bool, definitions: bool,
                       kb=None, retrieved: dict | None = None, run: int = 0,
                       original_prompt: bool = False, dataset: str = "pool") -> pd.DataFrame:
    ns = f"{dataset}__{condition}__run{run}"

    def one(row):
        uid, text = row["review_uid"], str(row["reviews.text"])
        ids = retrieved[uid] if retrieval else []
        if original_prompt:
            msgs = P.original_messages(text, kb.context_langchain_repr(ids))
            r = client.chat(ns, msgs, json_mode=False, logprobs=True, top_logprobs=5,
                            max_tokens=60, salt=f"run{run}")
        else:
            ctx = kb.context(ids) if retrieval else None
            msgs = P.classification_messages(text, ctx, definitions)
            r = client.chat(ns, msgs, json_mode=True, logprobs=True, top_logprobs=5,
                            max_tokens=200, salt=f"run{run}")
        obj = parse_json(r.get("text")) or {}
        label = _norm_label(obj.get("emotion"))
        conf = pd.to_numeric(obj.get("confidence"), errors="coerce")
        return {
            "review_uid": uid, "condition": condition, "run": run, "dataset": dataset,
            "label": label, "confidence_verbal": conf,
            "p_regret_logprob": label_probability(r.get("logprobs")),
            "rationale": obj.get("rationale"),
            "retrieved": "|".join(kb.sources(ids)) if retrieval else "",
            "error": label is None, "raw": r.get("text"),
            "system_fingerprint": r.get("system_fingerprint"),
            "tokens_in": r["usage"]["in"], "tokens_out": r["usage"]["out"],
            "backend": client.backend,
        }

    rows = _pmap(one, [r for _, r in df.iterrows()], desc=f"{dataset}:{condition}:run{run}")
    return pd.DataFrame(rows)


# ==========================================================================
# Speculative RAG (Wang et al., 2024)
#   1. retrieve n docs; 2. k-means them into k clusters (on their embeddings);
#   3. build m subsets with one doc sampled from each cluster;
#   4. drafter writes (rationale, answer) for each subset  -> rho_draft
#   5. verifier: self-consistency P(answer | review, rationale) -> rho_sc
#                self-reflection  P("Yes" | rationale supports answer) -> rho_sr
#   6. select draft maximising rho_draft * rho_sc * rho_sr
# API approximation (documented in the paper): rho_draft is the drafter's
# length-normalised sequence probability, rho_sc the verifier's label
# probability, rho_sr the verifier's P(Yes) - all read from token log-probs.
# ==========================================================================
def _subsets(kb, ids: list[int], seed: int) -> list[list[int]]:
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    k = min(C.SPEC_N_CLUSTERS, len(ids))
    vecs = kb.vectors[ids]
    # identical chunks (e.g. repeated reference lists) can give fewer distinct clusters than k
    n_distinct = len({tuple(np.round(v, 6)) for v in vecs})
    k = max(1, min(k, n_distinct))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lab = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit_predict(vecs)
    clusters = [cl for cl in ([ids[i] for i in range(len(ids)) if lab[i] == c] for c in range(k)) if cl]
    return [[int(rng.choice(cl)) for cl in clusters] for _ in range(C.SPEC_N_DRAFTS)]


def run_speculative(client, df: pd.DataFrame, kb, retrieved_n: dict, condition="C4_speculative",
                    run: int = 0, dataset: str = "pool", drafter_model: str = C.DRAFTER_MODEL
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    ns_d, ns_v = f"{dataset}__{condition}__drafter__run{run}", f"{dataset}__{condition}__verifier__run{run}"

    def one(row):
        uid, text = row["review_uid"], str(row["reviews.text"])
        seed = int(uid[1:]) + 1000 * run
        drafts = []
        for j, sub in enumerate(_subsets(kb, retrieved_n[uid], seed)):
            rd = client.chat(ns_d, P.drafter_messages(text, kb.context(sub)), model=drafter_model,
                             json_mode=True, logprobs=True, temperature=C.SPEC_DRAFT_TEMPERATURE,
                             max_tokens=200, salt=f"run{run}-draft{j}")
            obj = parse_json(rd.get("text")) or {}
            lab, rat = _norm_label(obj.get("emotion")), str(obj.get("rationale") or "")
            if lab is None:
                drafts.append({"review_uid": uid, "draft": j, "label": None, "error": True}); continue
            rho_draft = mean_token_prob(rd.get("logprobs"))
            sc = client.chat(ns_v, P.verifier_sc_messages(text, rat), model=C.MAIN_MODEL, json_mode=True,
                             logprobs=True, top_logprobs=5, max_tokens=20, salt=f"run{run}-sc{j}")
            p_reg = label_probability(sc.get("logprobs"))
            rho_sc = None if p_reg is None else (p_reg if lab == "REGRET" else 1 - p_reg)
            sr = client.chat(ns_v, P.verifier_sr_messages(text, lab, rat), model=C.MAIN_MODEL, json_mode=False,
                             logprobs=True, top_logprobs=5, max_tokens=1, salt=f"run{run}-sr{j}")
            rho_sr = yes_probability(sr.get("logprobs"))
            parts = [x for x in (rho_draft, rho_sc, rho_sr) if x is not None]
            score = float(np.prod(parts)) if len(parts) == 3 else np.nan
            drafts.append({"review_uid": uid, "draft": j, "label": lab, "rationale": rat,
                           "docs": "|".join(kb.sources(sub)), "rho_draft": rho_draft, "rho_sc": rho_sc,
                           "rho_sr": rho_sr, "score": score, "error": False,
                           "tokens_in": rd["usage"]["in"] + sc["usage"]["in"] + sr["usage"]["in"],
                           "tokens_out": rd["usage"]["out"] + sc["usage"]["out"] + sr["usage"]["out"]})
        ok = [d for d in drafts if not d["error"]]
        if not ok:
            best = {"label": None, "score": np.nan, "rationale": None}
        else:
            scored = [d for d in ok if not np.isnan(d["score"])] or ok
            best = max(scored, key=lambda d: d.get("score") if not np.isnan(d.get("score", np.nan)) else -1)
        votes = pd.Series([d["label"] for d in ok]).value_counts() if ok else pd.Series(dtype=int)
        p_reg_sel = None
        if ok and not np.isnan(best["score"]):
            # selection confidence: normalised score mass on REGRET drafts
            tot = sum(d["score"] for d in ok if not np.isnan(d["score"]))
            p_reg_sel = sum(d["score"] for d in ok if d["label"] == "REGRET" and not np.isnan(d["score"])) / tot if tot else None
        summary = {"review_uid": uid, "condition": condition, "run": run, "dataset": dataset,
                   "label": best["label"], "confidence_verbal": np.nan, "p_regret_logprob": p_reg_sel,
                   "rationale": best.get("rationale"), "best_score": best["score"],
                   "n_drafts_ok": len(ok), "n_drafts_regret": int(votes.get("REGRET", 0)),
                   "majority_label": votes.idxmax() if len(votes) else None,
                   "drafts_agree": len(votes) == 1, "error": best["label"] is None,
                   "tokens_in": sum(d.get("tokens_in", 0) for d in drafts),
                   "tokens_out": sum(d.get("tokens_out", 0) for d in drafts), "backend": client.backend}
        return summary, drafts

    res = _pmap(one, [r for _, r in df.iterrows()], desc=f"{dataset}:{condition}:run{run}")
    summ = pd.DataFrame([s for s, _ in res])
    drafts = pd.DataFrame([d for _, ds in res for d in ds])
    return summ, drafts


# ==========================================================================
# RDS dimensions and counterfactual focus
# ==========================================================================
DIMS = ["counterfactual", "responsibility", "controllability", "affect"]


def run_rds(client, df: pd.DataFrame, dataset: str = "pool") -> pd.DataFrame:
    ns = f"{dataset}__rds"

    def one(row):
        r = client.chat(ns, P.rds_messages(str(row["reviews.text"])), json_mode=True, max_tokens=80)
        obj = parse_json(r.get("text")) or {}
        vals = {d: pd.to_numeric(obj.get(d), errors="coerce") for d in DIMS}
        vals = {d: (v if pd.notna(v) and 1 <= v <= 7 else np.nan) for d, v in vals.items()}
        return {"review_uid": row["review_uid"], **{f"rds_{d}": v for d, v in vals.items()},
                "rds_error": any(pd.isna(v) for v in vals.values()), "backend": client.backend,
                "tokens_in": r["usage"]["in"], "tokens_out": r["usage"]["out"]}

    return pd.DataFrame(_pmap(one, [r for _, r in df.iterrows()], desc=f"{dataset}:rds"))


def run_cf_focus(client, df: pd.DataFrame, dataset: str = "pool") -> pd.DataFrame:
    ns = f"{dataset}__cf_focus"

    def one(row):
        r = client.chat(ns, P.cf_messages(str(row["reviews.text"])), json_mode=True, max_tokens=120)
        obj = parse_json(r.get("text")) or {}
        f = str(obj.get("focus", "")).strip().upper()
        f = f if f in ("BEHAVIOR", "SITUATION", "NONE") else None
        return {"review_uid": row["review_uid"], "cf_focus": f, "cf_evidence": obj.get("evidence"),
                "cf_error": f is None, "backend": client.backend,
                "tokens_in": r["usage"]["in"], "tokens_out": r["usage"]["out"]}

    return pd.DataFrame(_pmap(one, [r for _, r in df.iterrows()], desc=f"{dataset}:cf_focus"))


# ==========================================================================
# RDS composite (the paper's Equations 1-3) and alternatives
# ==========================================================================
def rds_composite(d: pd.DataFrame, w_cf=0.4, w_resp=0.3, w_ctrl=0.3, w_aff=0.4) -> pd.Series:
    """Paper's formula. Algebraically:
       RDS = w_cf*CF - w_aff*AFF + 2*w_resp*RESP + 2*w_ctrl*CTRL - 8*(w_resp+w_ctrl)
       i.e. with the paper's weights RDS = 0.4CF + 0.6RESP + 0.6CTRL - 0.4AFF - 4.8."""
    regret = d["rds_counterfactual"] * w_cf + d["rds_responsibility"] * w_resp + d["rds_controllability"] * w_ctrl
    disap = d["rds_affect"] * w_aff + (8 - d["rds_responsibility"]) * w_resp + (8 - d["rds_controllability"]) * w_ctrl
    return regret - disap
