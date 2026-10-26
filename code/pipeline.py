"""
Orchestrates every LLM stage. Usage (notebook or terminal):

    python pipeline.py --estimate            # token / cost estimate, no API calls
    python pipeline.py --all                 # run everything (resumable)
    python pipeline.py --stage classify      # one stage: kb | classify | speculative | original
                                             #            | stability | rds | cf | external
    python pipeline.py --all --mock          # smoke test with fake model (never use outputs)
    python pipeline.py --part part1_core --budget 3      # one paid part with a hard $ cap
    python pipeline.py --remaining           # what each part still costs (cache-aware)

Outputs: results/03_llm_outputs/
  <dataset>__<condition>__run<k>.csv     one row per review
  pool__C4_speculative__drafts.csv       every Speculative-RAG draft and its scores
  pool__rds.csv, pool__cf_focus.csv      dimension scores, counterfactual focus
  pool_master.csv                        wide table: all labels + RDS + ratings
  usage_and_cost.json
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import config as C
import tasks as T
from kb import KnowledgeBase, retrieval_diagnostics
from llm import BudgetExceeded, OutOfCredits, cache_usage, get_client

OUT = C.RESULTS_DIR / "03_llm_outputs"


def _save(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    if "error" in df:
        print(f"  saved {name}.csv  ({len(df)} rows, {int(df['error'].sum())} errors)")
    else:
        print(f"  saved {name}.csv  ({len(df)} rows)")


def gold_ids(random_only: bool = False) -> list[str]:
    a = pd.read_csv(C.DATA_DIR / "annotation_set.csv")
    if random_only and "gold_origin" in a:
        a = a[a["gold_origin"] != "targeted"]
    return a["review_uid"].tolist()


def load_sets():
    pool = pd.read_csv(C.DATA_DIR / "pool.csv")
    gold = pool[pool["review_uid"].isin(set(gold_ids()))]
    ext = pd.read_csv(C.DATA_DIR / "external_sample.csv") if (C.DATA_DIR / "external_sample.csv").exists() else None
    return pool, gold, ext


class Runner:
    def __init__(self, mock=False, api_key=None):
        C.ensure_dirs()
        self.client = get_client(mock=mock, api_key=api_key) if not mock else get_client(mock=True)
        self.pool, self.gold, self.ext = load_sets()
        self.gold_random = self.pool[self.pool["review_uid"].isin(set(gold_ids(random_only=True)))]
        self.kb = self.kb_orig = None
        self._retr = {}

    # ------------------------------------------------------------------ kb
    def stage_kb(self):
        pdfs = C.kb_pdfs()
        print("  revised KB PDFs :", [p.name for p in pdfs])
        self.kb = KnowledgeBase.build(self.client, pdfs, "revised")
        if self.kb is None:
            raise RuntimeError(f"No literature PDFs found in {C.KB_DIR} or {C.MATERIALS_DIR}.")
        if C.RUN_ORIGINAL_REPLICATION:
            orig = C.original_kb_pdfs()
            print("  original KB PDF :", [p.name for p in orig])
            self.kb_orig = KnowledgeBase.build(self.client, orig, "original") if orig else None

    def retrieved(self, dataset: str, df: pd.DataFrame, k: int, kb=None):
        kb = kb or self.kb
        key = (dataset, k, kb.name)
        if key not in self._retr:
            self._retr[key] = T.retrieve_for(self.client, kb, df, k, dataset)
        return self._retr[key]

    def _df(self, scope):
        return self.pool if scope == "pool" else self.gold

    # ------------------------------------------------------------ classify
    def stage_classify(self):
        if self.kb is None:
            self.stage_kb()
        for name, cfg in C.CONDITIONS.items():
            if not cfg["enabled"] or cfg["speculative"]:
                continue
            df = self._df(cfg["scope"])
            retr = self.retrieved("pool", self.pool, C.TOP_K) if cfg["retrieval"] else None
            res = T.run_classification(self.client, df, name, cfg["retrieval"], cfg["definitions"],
                                       kb=self.kb, retrieved=retr)
            _save(res, f"pool__{name}__run0")
        retrieval_diagnostics(self.kb, self.retrieved("pool", self.pool, C.TOP_K))

    def stage_speculative(self):
        cfg = C.CONDITIONS.get("C4_speculative")
        if not cfg or not cfg["enabled"]:
            return
        if self.kb is None:
            self.stage_kb()
        df = self._df(cfg["scope"])
        retr_n = self.retrieved("pool", self.pool, C.SPEC_N_RETRIEVE)
        summ, drafts = T.run_speculative(self.client, df, self.kb, retr_n)
        _save(summ, "pool__C4_speculative__run0"); _save(drafts, "pool__C4_speculative__drafts")

    def stage_original(self):
        """The paper's exact prompt and original single-PDF knowledge base, on the whole pool."""
        if self.kb is None:
            self.stage_kb()
        if self.kb_orig is None:
            print("  [original] original PDF not found - replication skipped"); return
        retr = self.retrieved("pool", self.pool, C.TOP_K, kb=self.kb_orig)
        res = T.run_classification(self.client, self.pool, "R0_original_replication", True, True,
                                   kb=self.kb_orig, retrieved=retr, original_prompt=True)
        _save(res, "pool__R0_original_replication__run0")
        retrieval_diagnostics(self.kb_orig, retr)

    def stage_stability(self):
        """Re-runs of the proposed system on the gold set (temperature 0 is not bit-exact)."""
        if self.kb is None:
            self.stage_kb()
        cfg = C.CONDITIONS[C.MAIN_CONDITION]
        retr = self.retrieved("pool", self.pool, C.TOP_K)
        for run in range(1, C.N_STABILITY_RUNS):
            res = T.run_classification(self.client, self.gold_random, C.MAIN_CONDITION, cfg["retrieval"],
                                       cfg["definitions"], kb=self.kb, retrieved=retr, run=run)
            _save(res, f"pool__{C.MAIN_CONDITION}__run{run}")

    def stage_decomp(self):
        """2x2 decomposition on the annotation set: which part of the original pipeline (prompt or
        knowledge base) produced its high regret rate? R0 (orig prompt + orig KB) and C3 (new prompt +
        new KB) already exist; this adds the two mixed cells."""
        if self.kb is None:
            self.stage_kb()
        if self.kb_orig is None:
            print("  [decomp] original PDF not found - skipped"); return
        retr_new = self.retrieved("pool", self.pool, C.TOP_K)
        retr_orig = self.retrieved("pool", self.pool, C.TOP_K, kb=self.kb_orig)
        r1 = T.run_classification(self.client, self.gold, "R1_origprompt_newKB", True, True, kb=self.kb,
                                  retrieved=retr_new, original_prompt=True)
        _save(r1, "pool__R1_origprompt_newKB__run0")
        r2 = T.run_classification(self.client, self.gold, "R2_newprompt_origKB", True, True, kb=self.kb_orig,
                                  retrieved=retr_orig)
        _save(r2, "pool__R2_newprompt_origKB__run0")

    def stage_spec4o(self):
        """Speculative RAG with GPT-4o as the drafter (annotation set): tests whether C4's high
        regret rate comes from the GPT-4o-mini drafter."""
        if self.kb is None:
            self.stage_kb()
        retr_n = self.retrieved("pool", self.pool, C.SPEC_N_RETRIEVE)
        summ, drafts = T.run_speculative(self.client, self.gold, self.kb, retr_n, condition="C5_speculative_4o_drafter",
                                         drafter_model=C.MAIN_MODEL)
        _save(summ, "pool__C5_speculative_4o_drafter__run0"); _save(drafts, "pool__C5_speculative_4o_drafter__drafts")

    def stage_rds(self):
        _save(T.run_rds(self.client, self.pool), "pool__rds")

    def stage_cf(self):
        _save(T.run_cf_focus(self.client, self.pool), "pool__cf_focus")

    def stage_external(self):
        if self.ext is None:
            print("  [external] no external sample"); return
        if self.kb is None:
            self.stage_kb()
        retr = self.retrieved("external", self.ext, C.TOP_K)
        for name in ["C1_definitions", C.MAIN_CONDITION]:
            cfg = C.CONDITIONS[name]
            res = T.run_classification(self.client, self.ext, name, cfg["retrieval"], cfg["definitions"],
                                       kb=self.kb, retrieved=retr if cfg["retrieval"] else None, dataset="external")
            _save(res, f"external__{name}__run0")
        _save(T.run_rds(self.client, self.ext, dataset="external"), "external__rds")

    # -------------------------------------------------------------- master
    def build_master(self):
        m = self.pool[["review_uid", "id", "name", "primaryCategories", "reviews.rating", "reviews.text", "reviews.date"]].copy()
        for f in sorted(OUT.glob("pool__*__run0.csv")):
            cond = f.stem.split("__")[1]
            d = pd.read_csv(f)[["review_uid", "label", "p_regret_logprob", "confidence_verbal"]]
            d.columns = ["review_uid"] + [f"{cond}__{c}" for c in d.columns[1:]]
            m = m.merge(d, on="review_uid", how="left")
        for f, cols in [("pool__rds.csv", ["review_uid"] + [f"rds_{d}" for d in T.DIMS] + ["rds_error"]),
                        ("pool__cf_focus.csv", ["review_uid", "cf_focus"])]:
            if (OUT / f).exists():
                m = m.merge(pd.read_csv(OUT / f)[cols], on="review_uid", how="left")
        if "rds_counterfactual" in m:
            m["rds_paper"] = T.rds_composite(m)
        m["in_gold_set"] = m["review_uid"].isin(self.gold["review_uid"])
        m.to_csv(OUT / "pool_master.csv", index=False)
        print(f"  saved pool_master.csv ({m.shape})")
        return m

    def save_usage(self):
        rep = self.client.cost_report()
        rep["backend"] = self.client.backend
        (OUT / "usage_and_cost_this_session.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print("  usage:", rep)

    def run_part(self, part: str, budget_usd: float | None) -> dict:
        """Runs one part of C.PARTS with a hard cap on NEW spend. Stops cleanly when the cap or
        the OpenAI credit is reached; finished calls stay cached, so re-running continues."""
        spec = C.PARTS[part]
        self.client.set_budget(budget_usd)
        status, msg = "complete", ""
        print(f"=== {spec['title']}  |  budget: " + (f"${budget_usd:.2f}" if budget_usd is not None else "none"))
        try:
            for s in spec["stages"]:
                if s == "kb" and self.kb is not None:
                    continue
                print(f"\n--- stage: {s}")
                getattr(self, f"stage_{s}")()
        except BudgetExceeded as e:
            status, msg = "stopped_budget", str(e)
        except OutOfCredits as e:
            status, msg = "stopped_out_of_credits", str(e)
        except Exception as e:
            status, msg = "error", repr(e)
            raise
        finally:
            try:
                self.build_master()
            except Exception as e:  # nothing to merge yet
                print("  (master table not built:", repr(e)[:120], ")")
            self.save_usage()
            spent = self.client.session_usd()
            ledger = OUT / "spend_ledger.csv"
            row = pd.DataFrame([{"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"), "part": part,
                                 "budget_usd": budget_usd, "spent_this_session_usd": round(spent, 2),
                                 "status": status, "backend": self.client.backend}])
            row.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
        if status != "complete":
            print(f"\n*** STOPPED: {msg}\n    Spent this session: ${spent:.2f}. Everything finished so far is cached;"
                  f" add credit / raise the budget and re-run this cell to continue.")
        else:
            print(f"\n*** {part} complete. Spent this session: ${spent:.2f}")
        return {"part": part, "status": status, "spent_usd": round(spent, 2), "message": msg}

    def run_all(self):
        for s in ["kb", "classify", "speculative", "original", "stability", "rds", "cf", "external"]:
            print(f"\n=== stage: {s}")
            getattr(self, f"stage_{s}")()
        self.build_master(); self.save_usage()


# --------------------------------------------------------------------------
# Cache-aware cost of what is still to be done, per part
# --------------------------------------------------------------------------
_DEFAULT_TOKENS = {  # (in, out) per call; replaced by measured averages once a namespace has data
    "C0_labels_only": (125, 45), "C1_definitions": (155, 47), "C2_rag_only": (650, 49), "C3_rag_full": (680, 47),
    "drafter": (490, 62), "verifier": (167, 5), "R0_original_replication": (850, 20),
    "rds": (130, 30), "cf_focus": (200, 35),
}


def _stage_jobs(stage: str, pool, gold, ext) -> list[tuple[str, str, int, tuple]]:
    """(namespace, model, expected_calls, default_tokens) for one stage."""
    N, G, E = len(pool), len(gold), (len(ext) if ext is not None else 0)
    jobs = []
    if stage == "classify":
        for name, cfg in C.CONDITIONS.items():
            if cfg["enabled"] and not cfg["speculative"]:
                n = N if cfg["scope"] == "pool" else G
                jobs.append((f"pool__{name}__run0", C.MAIN_MODEL, n, _DEFAULT_TOKENS[name]))
    elif stage == "speculative" and C.CONDITIONS["C4_speculative"]["enabled"]:
        n = N if C.CONDITIONS["C4_speculative"]["scope"] == "pool" else G
        jobs += [("pool__C4_speculative__drafter__run0", C.DRAFTER_MODEL, n * C.SPEC_N_DRAFTS, _DEFAULT_TOKENS["drafter"]),
                 ("pool__C4_speculative__verifier__run0", C.MAIN_MODEL, n * C.SPEC_N_DRAFTS * 2, _DEFAULT_TOKENS["verifier"])]
    elif stage == "original" and C.RUN_ORIGINAL_REPLICATION and C.original_kb_pdfs():
        jobs.append(("pool__R0_original_replication__run0", C.MAIN_MODEL, N, _DEFAULT_TOKENS["R0_original_replication"]))
    elif stage == "stability":
        GR = len(gold_ids(random_only=True))
        for r in range(1, C.N_STABILITY_RUNS):
            jobs.append((f"pool__{C.MAIN_CONDITION}__run{r}", C.MAIN_MODEL, GR, _DEFAULT_TOKENS["C3_rag_full"]))
    elif stage == "decomp" and C.original_kb_pdfs():
        jobs += [("pool__R1_origprompt_newKB__run0", C.MAIN_MODEL, G, _DEFAULT_TOKENS["R0_original_replication"]),
                 ("pool__R2_newprompt_origKB__run0", C.MAIN_MODEL, G, _DEFAULT_TOKENS["C3_rag_full"])]
    elif stage == "spec4o":
        jobs += [("pool__C5_speculative_4o_drafter__drafter__run0", C.MAIN_MODEL, G * C.SPEC_N_DRAFTS, _DEFAULT_TOKENS["drafter"]),
                 ("pool__C5_speculative_4o_drafter__verifier__run0", C.MAIN_MODEL, G * C.SPEC_N_DRAFTS * 2, _DEFAULT_TOKENS["verifier"])]
    elif stage == "rds":
        jobs.append(("pool__rds", C.MAIN_MODEL, N, _DEFAULT_TOKENS["rds"]))
    elif stage == "cf":
        jobs.append(("pool__cf_focus", C.MAIN_MODEL, N, _DEFAULT_TOKENS["cf_focus"]))
    elif stage == "external" and E:
        jobs += [("external__C1_definitions__run0", C.MAIN_MODEL, E, _DEFAULT_TOKENS["C1_definitions"]),
                 (f"external__{C.MAIN_CONDITION}__run0", C.MAIN_MODEL, E, _DEFAULT_TOKENS["C3_rag_full"]),
                 ("external__rds", C.MAIN_MODEL, E, _DEFAULT_TOKENS["rds"])]
    return jobs


def remaining_estimate(parts: list[str] | None = None) -> pd.DataFrame:
    """For each part/stage: calls already cached, calls left, and the indicative $ still to spend."""
    pool, gold, ext = load_sets()
    used = cache_usage().set_index("namespace")
    rows = []
    for part in parts or list(C.PARTS):
        for stage in C.PARTS[part]["stages"]:
            for ns, model, expected, (din, dout) in _stage_jobs(stage, pool, gold, ext):
                done = int(used.loc[ns, "valid_calls"]) if ns in used.index else 0
                if done >= 20:
                    din = used.loc[ns, "tokens_in"] / done; dout = used.loc[ns, "tokens_out"] / done
                left = max(0, expected - done)
                pr = C.PRICES.get(model, {"in": 0, "out": 0})
                rows.append({"part": part, "stage": stage, "namespace": ns, "model": model,
                             "calls_expected": expected, "calls_cached": min(done, expected), "calls_left": left,
                             "pct_done": round(100 * min(done, expected) / expected, 1) if expected else 100.0,
                             "usd_spent_so_far": float(used.loc[ns, "usd_indicative"]) if ns in used.index else 0.0,
                             "usd_left_indicative": round(left * (din / 1e6 * pr["in"] + dout / 1e6 * pr["out"]), 2)})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "remaining_cost_by_part.csv", index=False)
    return df


def remaining_summary(parts: list[str] | None = None) -> pd.DataFrame:
    d = remaining_estimate(parts)
    s = d.groupby("part", sort=False).agg(calls_left=("calls_left", "sum"), usd_spent_so_far=("usd_spent_so_far", "sum"),
                                          usd_left_indicative=("usd_left_indicative", "sum")).round(2)
    s.loc["TOTAL"] = s.sum()
    return s


# --------------------------------------------------------------------------
def estimate() -> pd.DataFrame:
    """Rough token / cost estimate (4 characters ~ 1 token) - no API calls."""
    import prompts as P
    pool, gold, ext = load_sets()
    tok = lambda s: len(s) / 4
    ctx5 = "x" * (C.CHUNK_SIZE * C.TOP_K) ; ctx3 = "x" * (C.CHUNK_SIZE * C.SPEC_N_CLUSTERS)
    rows = []
    def add(task, model, df, msgs_fn, out_tokens, calls_per_review=1):
        n_in = sum(tok("".join(m["content"] for m in msgs_fn(str(t)))) for t in df["reviews.text"]) * calls_per_review
        rows.append({"task": task, "model": model, "reviews": len(df), "calls": len(df) * calls_per_review,
                     "tokens_in": int(n_in), "tokens_out": int(out_tokens * len(df) * calls_per_review)})
    for name, cfg in C.CONDITIONS.items():
        if not cfg["enabled"] or cfg["speculative"]:
            continue
        df = pool if cfg["scope"] == "pool" else gold
        add(name, C.MAIN_MODEL, df, lambda t, c=cfg: P.classification_messages(t, ctx5 if c["retrieval"] else None, c["definitions"]), 45)
    if C.CONDITIONS["C4_speculative"]["enabled"]:
        df = pool if C.CONDITIONS["C4_speculative"]["scope"] == "pool" else gold
        add("C4 drafter", C.DRAFTER_MODEL, df, lambda t: P.drafter_messages(t, ctx3), 60, C.SPEC_N_DRAFTS)
        add("C4 verifier", C.MAIN_MODEL, df, lambda t: P.verifier_sc_messages(t, "r" * 200), 8, C.SPEC_N_DRAFTS)
        add("C4 verifier", C.MAIN_MODEL, df, lambda t: P.verifier_sr_messages(t, "REGRET", "r" * 200), 1, C.SPEC_N_DRAFTS)
    add("R0 original replication", C.MAIN_MODEL, pool, lambda t: P.original_messages(t, ctx5 + "x" * 600), 20)
    add("stability reruns", C.MAIN_MODEL, gold, lambda t: P.classification_messages(t, ctx5, True), 45, C.N_STABILITY_RUNS - 1)
    add("RDS", C.MAIN_MODEL, pool, P.rds_messages, 30)
    add("CF focus", C.MAIN_MODEL, pool, P.cf_messages, 35)
    if ext is not None:
        add("external C1", C.MAIN_MODEL, ext, lambda t: P.classification_messages(t, None, True), 45)
        add("external C3", C.MAIN_MODEL, ext, lambda t: P.classification_messages(t, ctx5, True), 45)
        add("external RDS", C.MAIN_MODEL, ext, P.rds_messages, 30)
    emb_tokens = (sum(map(len, pool["reviews.text"].astype(str))) + (sum(map(len, ext["reviews.text"].astype(str))) if ext is not None else 0)) / 4 + 400_000 / 4
    rows.append({"task": "embeddings (reviews + KB)", "model": C.EMBED_MODEL, "reviews": len(pool), "calls": 0,
                 "tokens_in": int(emb_tokens), "tokens_out": 0})
    est = pd.DataFrame(rows)
    est["usd_indicative"] = [round(r.tokens_in / 1e6 * C.PRICES[r.model]["in"] + r.tokens_out / 1e6 * C.PRICES[r.model]["out"], 2)
                             for r in est.itertuples()]
    OUT.mkdir(parents=True, exist_ok=True)
    est.to_csv(OUT / "cost_estimate.csv", index=False)
    return est


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true"); ap.add_argument("--stage")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--part", choices=list(C.PARTS)); ap.add_argument("--budget", type=float)
    ap.add_argument("--remaining", action="store_true")
    a = ap.parse_args()
    if a.remaining:
        print(remaining_estimate().to_string()); print(remaining_summary().to_string())
    elif a.part:
        Runner(mock=a.mock).run_part(a.part, a.budget)
    elif a.estimate:
        e = estimate(); print(e.to_string()); print("TOTAL indicative USD:", e["usd_indicative"].sum())
    else:
        r = Runner(mock=a.mock)
        if a.all:
            r.run_all()
        elif a.stage:
            getattr(r, f"stage_{a.stage}")(); r.build_master(); r.save_usage()
