"""
Thin, cached, retrying wrapper around the OpenAI API.

* Every request/response is appended to a JSONL cache under results/llm_cache/,
  keyed by a hash of the full request. Re-running a stage therefore costs
  nothing for reviews that were already processed, and a crashed session
  resumes where it stopped.
* `system_fingerprint`, token usage and (compact) token log-probabilities are
  stored with each response for reproducibility reporting.
* `MockClient` returns deterministic fake outputs for smoke-testing the code
  without an API key. Every row it produces is tagged backend="MOCK".
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as C

N_TOP_LOGPROB_TOKENS = 48     # keep alternatives only for the first N output tokens


class OutOfCredits(RuntimeError):
    """OpenAI reports no remaining credit / quota. The run stops; nothing is cached."""


class BudgetExceeded(RuntimeError):
    """This session reached its BUDGET_USD cap. The run stops; finished calls stay cached."""


_QUOTA_MARKERS = ("insufficient_quota", "no credits remaining", "exceeded your current quota", "billing_hard_limit")


def _is_quota_error(e) -> bool:
    return any(m in repr(e).lower() for m in _QUOTA_MARKERS)


def _valid(resp: dict | None) -> bool:
    return bool(resp) and not resp.get("error") and resp.get("text") is not None


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]


def text_hash(t: str) -> str:
    return hashlib.sha256(str(t).encode()).hexdigest()[:24]


class _JsonlCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.mem: dict[str, dict] = {}
        self.loaded: set[str] = set()

    def _load(self, ns: str):
        if ns in self.loaded:
            return
        f = self.dir / f"{ns}.jsonl"
        if f.exists():
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        if _valid(rec["response"]):      # failed calls are retried, never reused
                            self.mem[rec["key"]] = rec["response"]
                    except Exception:
                        continue          # tolerate a half-written last line
        self.loaded.add(ns)

    def get(self, ns: str, key: str):
        with self.lock:
            self._load(ns)
            return self.mem.get(key)

    def put(self, ns: str, key: str, request: dict, response: dict):
        if not _valid(response):
            return
        with self.lock:
            self._load(ns)
            self.mem[key] = response
            with (self.dir / f"{ns}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "ts": time.time(), "request": request,
                                     "response": response}, ensure_ascii=False) + "\n")


class LLMClient:
    backend = "openai"

    def __init__(self, cache_dir: Path = C.CACHE_DIR, api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.cache = _JsonlCache(cache_dir)
        self.usage_lock = threading.Lock()
        self.usage: dict[str, dict[str, int]] = {}
        self.budget_usd: float | None = None
        self.stop_reason: str | None = None

    # ------------------------------------------------------------- budget
    def set_budget(self, usd: float | None):
        """Hard cap on NEW spend in this session (cached calls are free). None = no cap."""
        self.budget_usd = usd
        self.stop_reason = None

    def session_usd(self) -> float:
        tot = 0.0
        with self.usage_lock:
            for m, u in self.usage.items():
                p = C.PRICES.get(m, {"in": 0, "out": 0})
                tot += u["in"] / 1e6 * p["in"] + u["out"] / 1e6 * p["out"]
        return tot

    def _check_budget(self):
        if self.stop_reason:
            raise (OutOfCredits if self.stop_reason == "credits" else BudgetExceeded)(self.stop_reason)
        if self.budget_usd is not None and self.session_usd() >= self.budget_usd:
            self.stop_reason = f"budget of ${self.budget_usd:.2f} reached"
            raise BudgetExceeded(self.stop_reason)

    # ---------------------------------------------------------------- chat
    def chat(self, namespace: str, messages: list[dict], model: str = C.MAIN_MODEL,
             json_mode: bool = True, logprobs: bool = False, top_logprobs: int = 0,
             temperature: float = C.TEMPERATURE, max_tokens: int = 400,
             salt: str = "") -> dict:
        request = dict(model=model, messages=messages, temperature=temperature,
                       max_tokens=max_tokens, json_mode=json_mode, logprobs=logprobs,
                       top_logprobs=top_logprobs, salt=salt)
        key = _hash(request)
        hit = self.cache.get(namespace, key)
        if hit is not None:
            return {**hit, "cached": True}
        self._check_budget()
        resp = self._call_chat(request)
        if resp.get("quota"):
            self.stop_reason = "credits"
            raise OutOfCredits("OpenAI reports no remaining credit - add credit and re-run this cell "
                               "(finished reviews are cached and will not be paid for again).")
        self.cache.put(namespace, key, request, resp)      # ignored if the call failed
        self._count(model, resp)
        return {**resp, "cached": False}

    def _call_chat(self, r: dict) -> dict:
        kwargs = dict(model=r["model"], messages=r["messages"], temperature=r["temperature"],
                      max_tokens=r["max_tokens"], seed=C.SEED)
        if r["json_mode"]:
            kwargs["response_format"] = {"type": "json_object"}
        if r["logprobs"]:
            kwargs["logprobs"] = True
            if r["top_logprobs"]:
                kwargs["top_logprobs"] = r["top_logprobs"]
        last_err = None
        for attempt in range(C.MAX_RETRIES):
            try:
                out = self.client.chat.completions.create(**kwargs)
                ch = out.choices[0]
                lp = None
                if r["logprobs"] and ch.logprobs is not None and ch.logprobs.content:
                    lp = []
                    for i, t in enumerate(ch.logprobs.content):
                        item = {"t": t.token, "lp": t.logprob}
                        if i < N_TOP_LOGPROB_TOKENS and t.top_logprobs:
                            item["top"] = [[a.token, a.logprob] for a in t.top_logprobs]
                        lp.append(item)
                return {"text": ch.message.content, "finish_reason": ch.finish_reason,
                        "logprobs": lp, "model": out.model,
                        "system_fingerprint": getattr(out, "system_fingerprint", None),
                        "usage": {"in": out.usage.prompt_tokens, "out": out.usage.completion_tokens}}
            except Exception as e:  # rate limits, timeouts, 5xx
                if _is_quota_error(e):
                    return {"text": None, "error": repr(e), "quota": True, "logprobs": None, "usage": {"in": 0, "out": 0}}
                last_err = e
                time.sleep(min(60, 2 ** attempt + random.random()))
        return {"text": None, "error": repr(last_err), "logprobs": None, "usage": {"in": 0, "out": 0}}

    # ---------------------------------------------------------- embeddings
    def embed(self, texts: list[str], name: str, model: str = C.EMBED_MODEL) -> np.ndarray:
        """Embeds texts, caching vectors in results/llm_cache/emb_<name>.npz."""
        f = self.cache.dir / f"emb_{name}.npz"
        hashes = [text_hash(t) for t in texts]
        have: dict[str, np.ndarray] = {}
        if f.exists():
            z = np.load(f, allow_pickle=True)
            have = dict(zip(z["hashes"].tolist(), z["vectors"]))
        missing = [i for i, h in enumerate(hashes) if h not in have]
        for start in range(0, len(missing), 256):
            idx = missing[start:start + 256]
            vecs = self._call_embed([texts[i] for i in idx], model)
            for i, v in zip(idx, vecs):
                have[hashes[i]] = v
        if missing:
            keys = list(have.keys())
            np.savez_compressed(f, hashes=np.array(keys), vectors=np.stack([have[k] for k in keys]))
        return np.stack([have[h] for h in hashes]).astype("float32")

    def _call_embed(self, batch: list[str], model: str) -> list[np.ndarray]:
        self._check_budget()
        for attempt in range(C.MAX_RETRIES):
            try:
                out = self.client.embeddings.create(model=model, input=[str(t)[:8000] for t in batch])
                with self.usage_lock:
                    u = self.usage.setdefault(model, {"in": 0, "out": 0}); u["in"] += out.usage.prompt_tokens
                return [np.array(d.embedding, dtype="float32") for d in out.data]
            except Exception as e:
                if _is_quota_error(e):
                    self.stop_reason = "credits"
                    raise OutOfCredits("OpenAI reports no remaining credit (embeddings).")
                time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError("Embedding failed after retries")

    def _count(self, model: str, resp: dict):
        with self.usage_lock:
            u = self.usage.setdefault(model, {"in": 0, "out": 0})
            u["in"] += resp["usage"]["in"]; u["out"] += resp["usage"]["out"]

    def cost_report(self) -> dict:
        rep = {}
        for m, u in self.usage.items():
            p = C.PRICES.get(m, {"in": 0, "out": 0})
            rep[m] = {**u, "usd_indicative": round(u["in"] / 1e6 * p["in"] + u["out"] / 1e6 * p["out"], 2)}
        return rep


# ==========================================================================
# Mock client - for code verification only. NEVER use its outputs as results.
# ==========================================================================
class MockClient(LLMClient):
    backend = "MOCK"

    def __init__(self, cache_dir: Path = C.CACHE_DIR, **_):
        self.cache = _JsonlCache(cache_dir)
        self.usage_lock = threading.Lock()
        self.usage = {}
        self.budget_usd = None
        self.stop_reason = None

    @staticmethod
    def _rng(*parts) -> random.Random:
        return random.Random(int(_hash(parts)[:12], 16))

    def _call_chat(self, r: dict) -> dict:
        prompt = "\n".join(m["content"] for m in r["messages"])
        rng = self._rng(prompt, r["model"], r["salt"])
        after = prompt.split("Review:")[-1].lstrip()
        review = after.split("\n\n")[0].lower()
        self_blame = bool(re.search(r"\b(i should|i wish|my fault|my mistake|i regret|should have)\b", review))
        if "Counterfactual Thinking" in prompt:
            base = 5 if self_blame else 2
            obj = {"counterfactual": min(7, max(1, base + rng.randint(-1, 1))),
                   "responsibility": min(7, max(1, base + rng.randint(-1, 1))),
                   "controllability": min(7, max(1, base + rng.randint(-1, 2))),
                   "affect": rng.randint(3, 7)}
            text = json.dumps(obj); lp = None
        elif "BEHAVIOR" in prompt and "SITUATION" in prompt:
            focus = "BEHAVIOR" if self_blame else rng.choice(["SITUATION", "SITUATION", "NONE"])
            text = json.dumps({"counterfactual_present": focus != "NONE", "focus": focus, "evidence": ""}); lp = None
        elif "supports the answer" in prompt:
            p = rng.uniform(0.55, 0.99)
            text = json.dumps({"verdict": "Yes"})
            lp = [{"t": '{"', "lp": 0.0}, {"t": "verdict", "lp": 0.0}, {"t": '":"', "lp": 0.0},
                  {"t": "Yes", "lp": math.log(p), "top": [["Yes", math.log(p)], ["No", math.log(1 - p)]]}]
        else:
            p_reg = rng.uniform(0.40, 0.95) if self_blame else rng.uniform(0.02, 0.58)
            lab = "REGRET" if p_reg >= 0.5 else "DISAPPOINTMENT"
            obj = {"emotion": lab, "confidence": round(max(p_reg, 1 - p_reg), 2),
                   "rationale": "mock rationale"}
            if "rationale" in prompt and prompt.find('"rationale"') < prompt.find('"emotion"'):
                obj = {"rationale": "mock rationale", "emotion": lab}
            text = json.dumps(obj)
            lp = [{"t": '{"', "lp": 0.0}, {"t": "emotion", "lp": 0.0}, {"t": '":"', "lp": 0.0},
                  {"t": "REG" if lab == "REGRET" else "DIS", "lp": math.log(max(p_reg, 1 - p_reg)),
                   "top": [["REG", math.log(p_reg)], ["DIS", math.log(1 - p_reg)]]},
                  {"t": "RET" if lab == "REGRET" else "APPOINTMENT", "lp": -0.01}]
        n_in = len(prompt) // 4
        return {"text": text, "finish_reason": "stop", "logprobs": lp, "model": r["model"] + "-MOCK",
                "system_fingerprint": "mock", "usage": {"in": n_in, "out": len(text) // 4}}

    def _call_embed(self, batch, model):
        out = []
        for t in batch:
            rs = np.random.default_rng(int(text_hash(t)[:12], 16))
            v = rs.normal(size=64).astype("float32"); out.append(v / np.linalg.norm(v))
        return out


def get_client(mock: bool = False, **kw) -> LLMClient:
    return MockClient(**kw) if mock else LLMClient(**kw)


# ==========================================================================
# Parsing helpers
# ==========================================================================
def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        s = s[4:] if s.lower().startswith("json") else s
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _norm(tok: str) -> str:
    return tok.strip().strip('":').strip()


def label_probability(logprobs: list[dict] | None, positive_prefix: str = "REG",
                      negative_prefix: str = "DIS") -> float | None:
    """P(REGRET) from the top-logprobs at the first token of the emotion value.

    Finds the first generated token that starts the label (REG.. / DIS..) and
    renormalises the probability mass over the two label prefixes."""
    if not logprobs:
        return None
    seen = ""
    for item in logprobs:
        before = seen
        seen += item["t"]
        if "emotion" not in before:          # only look at the value of the "emotion" key
            continue
        tok = item["t"].strip().strip('":').strip().upper()
        if not tok:
            continue
        if tok.startswith(positive_prefix[:2]) or tok.startswith(negative_prefix[:2]):
            alts = item.get("top") or [[item["t"], item["lp"]]]
            pos = sum(math.exp(lp) for t, lp in alts if _norm(t).upper().startswith(positive_prefix[:2]))
            neg = sum(math.exp(lp) for t, lp in alts if _norm(t).upper().startswith(negative_prefix[:2]))
            if pos + neg == 0:
                return None
            return pos / (pos + neg)
    return None


def yes_probability(logprobs: list[dict] | None) -> float | None:
    """P(Yes) at the first Yes/No token of the verifier's answer."""
    if not logprobs:
        return None
    for item in logprobs:
        tok = _norm(item["t"]).lower()
        if tok in ("yes", "no"):
            alts = item.get("top") or [[item["t"], item["lp"]]]
            y = sum(math.exp(lp) for t, lp in alts if _norm(t).lower() == "yes")
            n = sum(math.exp(lp) for t, lp in alts if _norm(t).lower() == "no")
            return y / (y + n) if y + n > 0 else None
    return None


def mean_token_prob(logprobs: list[dict] | None) -> float | None:
    """Length-normalised sequence probability exp(mean log p)."""
    if not logprobs:
        return None
    return float(math.exp(sum(i["lp"] for i in logprobs) / len(logprobs)))


# ==========================================================================
# Spend accounting from the cache (what has actually been paid for so far)
# ==========================================================================
def cache_usage(cache_dir: Path = C.CACHE_DIR) -> pd.DataFrame:
    """One row per cache namespace: valid calls, tokens and indicative USD."""
    rows = []
    for f in sorted(Path(cache_dir).glob("*.jsonl")):
        n = tin = tout = 0; model = None
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not _valid(rec["response"]):
                    continue
                n += 1; tin += rec["response"]["usage"]["in"]; tout += rec["response"]["usage"]["out"]
                model = rec["request"]["model"]
        p = C.PRICES.get(model or "", {"in": 0, "out": 0})
        rows.append({"namespace": f.stem, "model": model, "valid_calls": n, "tokens_in": tin, "tokens_out": tout,
                     "usd_indicative": round(tin / 1e6 * p["in"] + tout / 1e6 * p["out"], 2)})
    return pd.DataFrame(rows, columns=["namespace", "model", "valid_calls", "tokens_in", "tokens_out", "usd_indicative"])
