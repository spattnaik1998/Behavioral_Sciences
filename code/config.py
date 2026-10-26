"""
Central configuration for the revision experiments.

Every path is derived from REVISION_DIR, so the same code runs on a laptop
or in any other workspace. Set the environment
variable REVISION_DIR to override the auto-detected location.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
REVISION_DIR = Path(os.environ.get("REVISION_DIR", Path(__file__).resolve().parents[1]))
PROJECT_DIR = REVISION_DIR.parent                      # .../Behavioral_Economics
# Raw Datafiniti files: <repo>/dataset/ if present (GitHub layout), else ../Dataset (original layout)
DATASET_DIR = Path(os.environ.get("DATASET_DIR", REVISION_DIR / "dataset" if (REVISION_DIR / "dataset").is_dir()
                                  else PROJECT_DIR / "Dataset"))

MATERIALS_DIR = Path(os.environ.get("MATERIALS_DIR", PROJECT_DIR / "Materials"))  # literature PDFs
CODE_DIR = REVISION_DIR / "code"
DATA_DIR = REVISION_DIR / "data"
KB_DIR = REVISION_DIR / "knowledge_base"                # optional override; else Materials/ is used
ANNOT_DIR = REVISION_DIR / "annotation"
ANNOT_RETURNED_DIR = ANNOT_DIR / "returned"             # completed coder sheets go here
RESULTS_DIR = REVISION_DIR / "results"
CACHE_DIR = RESULTS_DIR / "llm_cache"                   # raw API responses (JSONL, resumable)

# The single PDF the ORIGINAL notebook indexed (only the first uploaded file was used)
ORIGINAL_KB_PATTERN = "Mental Accounting"

# Raw data. The slim file holds the same 28,332 rows as the original May19 file
# with only the columns we use; either works.
MAIN_RAW_CANDIDATES = [
    DATA_DIR / "may19_slim.csv.gz",
    DATASET_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
]
EXTERNAL_RAW = DATASET_DIR / "1429_1.csv"               # Amazon devices (Kindle/Fire/Echo)
ORIGINAL_5000_RAW = DATASET_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"

# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
SEED = 42
RATING_MAX = 3                     # keep reviews with rating <= 3 (as in the paper)
ORIGINAL_SAMPLE_N = 100            # the paper's original sample (rebuilt exactly)
ANNOTATION_N_NEW = 100             # extra reviews for the human gold set (on top of the 100)
EXTERNAL_SAMPLE_N = 500            # sample from the external dataset for generalisability

# --------------------------------------------------------------------------
# Models (OpenAI)
# --------------------------------------------------------------------------
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gpt-4o")          # classifier / verifier
DRAFTER_MODEL = os.environ.get("DRAFTER_MODEL", "gpt-4o-mini")  # Speculative RAG drafter
EMBED_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.0

# Retrieval (identical to the paper)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

# Speculative RAG (Wang et al., 2024)
SPEC_N_RETRIEVE = 10    # documents retrieved before clustering
SPEC_N_CLUSTERS = 3     # k: clusters of retrieved documents
SPEC_N_DRAFTS = 4       # m: drafts, each grounded in one subset (one doc per cluster)
SPEC_DRAFT_TEMPERATURE = 0.0

# Concurrency / robustness
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 8))
MAX_RETRIES = 6

# Repeated runs of the main system on the gold set, for test-retest reliability
N_STABILITY_RUNS = 3

# --------------------------------------------------------------------------
# Experimental conditions (ablation grid). Set "enabled": False to skip one.
#   retrieval   : retrieve literature chunks and put them in the prompt
#   definitions : include the explicit REGRET / DISAPPOINTMENT definitions
#   scope       : "pool" = all 2,019 reviews; "gold" = only the human-annotated set
# --------------------------------------------------------------------------
CONDITIONS = {
    "C0_labels_only":   dict(retrieval=False, definitions=False, speculative=False, scope="pool", enabled=True),
    "C1_definitions":   dict(retrieval=False, definitions=True,  speculative=False, scope="pool", enabled=True),
    "C2_rag_only":      dict(retrieval=True,  definitions=False, speculative=False, scope="pool", enabled=True),
    "C3_rag_full":      dict(retrieval=True,  definitions=True,  speculative=False, scope="pool", enabled=True),  # proposed system
    "C4_speculative":   dict(retrieval=True,  definitions=True,  speculative=True,  scope="pool", enabled=True),
}
MAIN_CONDITION = "C3_rag_full"

# --------------------------------------------------------------------------
# Paid work split into parts, so credit can be topped up between them.
# Each part runs only its own stages; everything is cached, so parts can be
# run in any order, interrupted, and re-run without paying twice.
# --------------------------------------------------------------------------
PARTS = {
    "part1_core": {
        "title": "Part 1 - Core reviewer responses (ablation C0-C3 + RDS dimensions)",
        "stages": ["kb", "classify", "rds"],
        "answers": "R1.1 R2.1 R2.2 R2.4 R2.5 R2.6 R3.3 R3.4 R3.5 R4.5 R4.6",
    },
    "part2_speculative": {
        "title": "Part 2 - Speculative RAG (C4)",
        "stages": ["kb", "speculative"],
        "answers": "R1.2 R3.1 R4.2 R4.7",
    },
    "part3_robustness": {
        "title": "Part 3 - Robustness extras (original replication, stability, counterfactual focus, external dataset)",
        "stages": ["kb", "original", "stability", "cf", "external"],
        "answers": "R2.8 R4.8 R4.9 + reproducibility",
    },
    "part4_decomposition": {
        "title": "Part 4 - Why did the original pipeline say 22% regret? (prompt vs knowledge base) + Speculative RAG with a GPT-4o drafter",
        "stages": ["kb", "decomp", "spec4o"],
        "answers": "R3.5 R4.6 R4.7 + explains the change from the original results",
    },
}

# The original paper's exact prompt + original single-PDF knowledge base.
# Runs only if that PDF is present in KB_DIR/original/.
RUN_ORIGINAL_REPLICATION = True

# Indicative prices (USD per 1M tokens). CHECK CURRENT PRICING before relying on these.
PRICES = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
}

LABELS = ["REGRET", "DISAPPOINTMENT"]


def ensure_dirs() -> None:
    for d in [DATA_DIR, KB_DIR, KB_DIR / "original", ANNOT_DIR, ANNOT_RETURNED_DIR, RESULTS_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def main_raw_path() -> Path:
    for p in MAIN_RAW_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the raw data files found: {MAIN_RAW_CANDIDATES}")


# --------------------------------------------------------------------------
# Knowledge-base file selection
# --------------------------------------------------------------------------
def kb_pdfs() -> list[Path]:
    """Revised knowledge base = every PDF in knowledge_base/ if any, else every PDF in Materials/."""
    own = sorted(KB_DIR.glob("*.pdf"))
    return own if own else sorted(MATERIALS_DIR.glob("*.pdf"))


def original_kb_pdfs() -> list[Path]:
    """Original (buggy) knowledge base = knowledge_base/original/*.pdf if any, else the
    Mental Accounting PDF in Materials/."""
    own = sorted((KB_DIR / "original").glob("*.pdf"))
    if own:
        return own
    for d in (KB_DIR, MATERIALS_DIR):
        hit = [p for p in sorted(d.glob("*.pdf")) if ORIGINAL_KB_PATTERN.lower() in p.name.lower()]
        if hit:
            return hit
    return []


# --------------------------------------------------------------------------
# API key from a .env file (Behavioral_Economics/.env, or Revision/.env which wins if both exist).
# The .env file OVERRIDES any OPENAI_API_KEY already set in Windows / the shell / VS Code,
# so the key in .env is always the one used. Values are never printed (only the last 4 chars).
# --------------------------------------------------------------------------
ENV_FILES = [PROJECT_DIR / ".env", REVISION_DIR / ".env"]


def load_env(override: bool = True) -> list[str]:
    loaded = []
    for f in ENV_FILES:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip().strip('"').strip("'")
            if k and v and (override or not os.environ.get(k)):
                os.environ[k] = v
                loaded.append(f"{k} (from {f.parent.name}/.env)")
    return loaded


def key_fingerprint(name: str = "OPENAI_API_KEY") -> str:
    v = os.environ.get(name, "")
    return f"...{v[-4:]} (length {len(v)})" if v else "NOT SET"


ENV_LOADED = load_env()
