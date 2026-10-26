"""
Stage 3b - transformer baselines (optional; runs locally on CPU or GPU; needs Hugging Face
model downloads, which this workspace cannot reach).

  B4  zero-shot NLI     facebook/bart-large-mnli, hypotheses phrased from the
                        same definitions the LLM receives (no training data)
  B5  RoBERTa-base CV   fine-tuned on the human gold set, stratified 5-fold CV
                        (out-of-fold predictions -> no train/test leakage)
  B6  RoBERTa silver    fine-tuned on GPT-4o (C3) labels for the 1,819 pool
                        reviews NOT in the gold set, evaluated on the gold set
                        (the "distil the LLM into a cheap classifier" scenario)

Outputs: results/05_baselines/pool_nli_zeroshot.csv, gold_roberta_cv.csv,
         gold_roberta_silver.csv, transformer_baselines_meta.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C

OUT = C.RESULTS_DIR / "05_baselines"
NLI_MODEL = "facebook/bart-large-mnli"
CLS_MODEL = "roberta-base"
HYPOTHESES = {
    "REGRET": "The reviewer blames their own decision or action for the bad outcome.",
    "DISAPPOINTMENT": "The reviewer blames the product, the seller or external circumstances for the bad outcome.",
}


def nli_zeroshot(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    import torch
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=NLI_MODEL, device=0 if torch.cuda.is_available() else -1)
    labels = list(HYPOTHESES.values())
    res = clf(df["reviews.text"].astype(str).tolist(), candidate_labels=labels, hypothesis_template="{}",
              batch_size=batch_size, multi_label=False)
    inv = {v: k for k, v in HYPOTHESES.items()}
    rows = []
    for uid, r in zip(df["review_uid"], res):
        sc = dict(zip(r["labels"], r["scores"]))
        rows.append({"review_uid": uid, "label": inv[r["labels"][0]], "p_regret_logprob": sc[HYPOTHESES["REGRET"]],
                     "condition": "B4_nli_zeroshot"})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "pool_nli_zeroshot.csv", index=False)
    return out


def _finetune_predict(train_txt, train_y, test_txt, seed=C.SEED, epochs=4, lr=2e-5, max_len=160,
                      model_name=CLS_MODEL, tokenizer=None, model_init=None):
    """Fine-tune a sequence classifier with class-weighted loss; return P(REGRET) for test."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
    model = (model_init() if model_init else AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)).to(dev)
    y = np.asarray(train_y, int)
    enc = tok(list(train_txt), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    dl = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y)), batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    w = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2).clip(min=1)), dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl); sch = get_linear_schedule_with_warmup(opt, int(.1 * steps), steps)
    model.train()
    for _ in range(epochs):
        for ids, am, yy in dl:
            opt.zero_grad()
            out = model(input_ids=ids.to(dev), attention_mask=am.to(dev))
            loss = loss_fn(out.logits, yy.to(dev)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    model.eval(); probs = []
    te = tok(list(test_txt), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        for i in range(0, len(test_txt), 64):
            lo = model(input_ids=te["input_ids"][i:i + 64].to(dev), attention_mask=te["attention_mask"][i:i + 64].to(dev)).logits
            probs.extend(torch.softmax(lo, -1)[:, 1].cpu().numpy().tolist())
    return np.array(probs)


def roberta_cv(gold: pd.DataFrame, pool: pd.DataFrame, n_splits=5, **kw) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold
    g = gold.dropna(subset=["forced_choice"]).merge(pool[["review_uid", "reviews.text"]], on="review_uid").reset_index(drop=True)
    y = (g["forced_choice"] == "REGRET").astype(int).values
    n_splits = min(n_splits, int(np.bincount(y).min()))
    p = np.full(len(g), np.nan); fold = np.zeros(len(g), int)
    for k, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=C.SEED).split(g, y)):
        p[te] = _finetune_predict(g.loc[tr, "reviews.text"], y[tr], g.loc[te, "reviews.text"], **kw); fold[te] = k
    out = pd.DataFrame({"review_uid": g["review_uid"], "label": np.where(p >= .5, "REGRET", "DISAPPOINTMENT"),
                        "p_regret_logprob": p, "fold": fold, "condition": "B5_roberta_cv"})
    OUT.mkdir(parents=True, exist_ok=True); out.to_csv(OUT / "gold_roberta_cv.csv", index=False)
    return out


def roberta_silver(gold_ids: set, pool: pd.DataFrame, llm_labels: pd.DataFrame, **kw) -> pd.DataFrame:
    tr = pool[~pool["review_uid"].isin(gold_ids)].merge(llm_labels[["review_uid", "label"]].dropna(), on="review_uid")
    te = pool[pool["review_uid"].isin(gold_ids)]
    p = _finetune_predict(tr["reviews.text"], (tr["label"] == "REGRET").astype(int), te["reviews.text"], **kw)
    out = pd.DataFrame({"review_uid": te["review_uid"], "label": np.where(p >= .5, "REGRET", "DISAPPOINTMENT"),
                        "p_regret_logprob": p, "condition": "B6_roberta_silver", "n_train_silver": len(tr)})
    OUT.mkdir(parents=True, exist_ok=True); out.to_csv(OUT / "gold_roberta_silver.csv", index=False)
    return out


def run(skip_nli=False, skip_cv=False, skip_silver=False):
    pool = pd.read_csv(C.DATA_DIR / "pool.csv")
    meta = {"nli_model": NLI_MODEL, "classifier": CLS_MODEL, "hypotheses": HYPOTHESES}
    if not skip_nli:
        nli_zeroshot(pool); meta["nli"] = "done"
    gf = C.DATA_DIR / "gold_labels.csv"
    if gf.exists():
        gold = pd.read_csv(gf)
        if not skip_cv:
            roberta_cv(gold, pool); meta["roberta_cv"] = "done"
        lf = C.RESULTS_DIR / "03_llm_outputs" / f"pool__{C.MAIN_CONDITION}__run0.csv"
        if not skip_silver and lf.exists():
            roberta_silver(set(pd.read_csv(C.DATA_DIR / "annotation_set.csv")["review_uid"]), pool, pd.read_csv(lf))
            meta["roberta_silver"] = "done"
    else:
        meta["roberta"] = "skipped - gold labels not available yet"
    (OUT / "transformer_baselines_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(run())
