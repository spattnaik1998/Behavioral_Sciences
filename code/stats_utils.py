"""Statistical helpers used across the analysis (no LLM needed)."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config as C

N_BOOT = 10_000
N_PERM = 10_000


def rng(seed=C.SEED):
    return np.random.default_rng(seed)


def bootstrap_ci(stat_fn, n: int, n_boot: int = N_BOOT, alpha: float = 0.05, seed=C.SEED):
    """Percentile CI for stat_fn(indices) under case resampling."""
    r = rng(seed); vals = []
    for _ in range(n_boot):
        try:
            v = stat_fn(r.integers(0, n, n))
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def wilson_ci(k: int, n: int, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2); p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / sp
    j = 1 - 3 / (4 * (nx + ny) - 9)
    return d, d * j


def d_ci(d, nx, ny, alpha=0.05):
    """Approximate CI for Cohen's d (Hedges & Olkin large-sample SE)."""
    se = math.sqrt((nx + ny) / (nx * ny) + d ** 2 / (2 * (nx + ny)))
    z = stats.norm.ppf(1 - alpha / 2)
    return d - z * se, d + z * se


def two_group_report(x, y, label_x="REGRET", label_y="DISAPPOINTMENT", seed=C.SEED) -> dict:
    """Full robust comparison of two independent groups (R2.5)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    student = stats.ttest_ind(x, y, equal_var=True)
    welch = stats.ttest_ind(x, y, equal_var=False)
    # Welch-Satterthwaite df and CI for the mean difference
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df_w = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    diff = x.mean() - y.mean(); se = math.sqrt(vx + vy)
    tcrit = stats.t.ppf(0.975, df_w)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = rng(seed)
    # bootstrap CI for the mean difference (stratified: resample within groups)
    boots = np.array([r.choice(x, len(x)).mean() - r.choice(y, len(y)).mean() for _ in range(N_BOOT)])
    # permutation test for the mean difference
    pooled = np.concatenate([x, y]); obs = abs(diff); cnt = 0
    for _ in range(N_PERM):
        p = r.permutation(pooled)
        cnt += abs(p[:len(x)].mean() - p[len(x):].mean()) >= obs
    d, g = hedges_g(x, y)
    dlo, dhi = d_ci(d, len(x), len(y))
    # rank-biserial correlation from Mann-Whitney U
    rbc = 1 - 2 * mw.statistic / (len(x) * len(y))
    lev = stats.levene(x, y, center="median")
    return {
        f"n_{label_x}": len(x), f"n_{label_y}": len(y),
        f"mean_{label_x}": x.mean(), f"sd_{label_x}": x.std(ddof=1),
        f"mean_{label_y}": y.mean(), f"sd_{label_y}": y.std(ddof=1),
        "mean_diff": diff,
        "student_t": student.statistic, "student_df": len(x) + len(y) - 2, "student_p": student.pvalue,
        "welch_t": welch.statistic, "welch_df": df_w, "welch_p": welch.pvalue,
        "welch_ci95_low": diff - tcrit * se, "welch_ci95_high": diff + tcrit * se,
        "bootstrap_ci95_low": np.percentile(boots, 2.5), "bootstrap_ci95_high": np.percentile(boots, 97.5),
        "permutation_p": (cnt + 1) / (N_PERM + 1),
        "mann_whitney_U": mw.statistic, "mann_whitney_p": mw.pvalue, "rank_biserial_r": -rbc,
        "levene_brown_forsythe_p": lev.pvalue,
        "cohens_d": d, "cohens_d_ci95_low": dlo, "cohens_d_ci95_high": dhi, "hedges_g": g,
    }


def power_two_sample(d, n1, n2, alpha=0.05):
    """Power of a two-sided two-sample t-test (noncentral t)."""
    df = n1 + n2 - 2
    nc = d * math.sqrt(n1 * n2 / (n1 + n2))
    tc = stats.t.ppf(1 - alpha / 2, df)
    p = 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    if not np.isfinite(p):                      # numerical underflow for very large nc
        zc = stats.norm.ppf(1 - alpha / 2)
        p = 1 - stats.norm.cdf(zc - nc) + stats.norm.cdf(-zc - nc)
    return float(min(1.0, p))


def n_for_power(d, ratio, power=0.8, alpha=0.05):
    """Smallest n1 (n2 = ratio*n1) achieving the requested power."""
    for n1 in range(2, 100000):
        n2 = max(2, int(round(ratio * n1)))
        if power_two_sample(d, n1, n2, alpha) >= power:
            return n1, n2
    return None


def _boot_idx(n, n_boot, seed):
    return rng(seed).integers(0, n, size=(n_boot, n))


def _rowwise_pearson(X, Y):
    X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
    den = np.sqrt((X ** 2).sum(1) * (Y ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (X * Y).sum(1) / den


def corr_report(x, y, seed=C.SEED, n_boot=4000) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    pr = stats.pearsonr(x, y); sr = stats.spearmanr(x, y); kt = stats.kendalltau(x, y)
    n = len(x)
    idx = _boot_idx(n, n_boot, seed)
    bx, by = x[idx], y[idx]
    pb = _rowwise_pearson(bx, by)
    sb = _rowwise_pearson(stats.rankdata(bx, axis=1), stats.rankdata(by, axis=1))
    pb, sb = pb[~np.isnan(pb)], sb[~np.isnan(sb)]
    z = np.arctanh(pr.statistic); se = 1 / math.sqrt(n - 3)
    ci = lambda a, b: f"[{float(a):.3f}, {float(b):.3f}]"
    return {"n": n, "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "pearson_ci95_fisher": ci(math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)),
            "pearson_ci95_boot": ci(*np.percentile(pb, [2.5, 97.5])),
            "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "spearman_ci95_boot": ci(*np.percentile(sb, [2.5, 97.5])),
            "kendall_tau_b": kt.statistic, "kendall_p": kt.pvalue}


def icc2_1(a, b) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater."""
    X = np.column_stack([a, b]).astype(float)
    X = X[~np.isnan(X).any(axis=1)]
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar test on paired correctness vectors (system A vs B)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b01 = int((a & ~b).sum()); b10 = int((~a & b).sum())
    n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5))
    return {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "exact_p": p}


def classification_metrics(y_true, y_pred, pos="REGRET", sample_weight=None) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, matthews_corrcoef
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    labs = ["REGRET", "DISAPPOINTMENT"]
    w = sample_weight
    out = {"n": len(yt),
           "accuracy": accuracy_score(yt, yp, sample_weight=w),
           "balanced_accuracy": balanced_accuracy_score(yt, yp, sample_weight=w),
           "macro_f1": f1_score(yt, yp, labels=labs, average="macro", zero_division=0, sample_weight=w),
           "kappa_vs_human": cohen_kappa_score(yt, yp, sample_weight=w), "mcc": matthews_corrcoef(yt, yp, sample_weight=w)}
    for lab in labs:
        out[f"precision_{lab}"] = precision_score(yt, yp, labels=labs, average=None, zero_division=0, sample_weight=w)[labs.index(lab)]
        out[f"recall_{lab}"] = recall_score(yt, yp, labels=labs, average=None, zero_division=0, sample_weight=w)[labs.index(lab)]
        out[f"f1_{lab}"] = f1_score(yt, yp, labels=labs, average=None, zero_division=0, sample_weight=w)[labs.index(lab)]
    return out


def _fast_metrics(yt_pos, yp_pos, yp_valid):
    """Vectorised accuracy / macro-F1 / kappa / F1(REGRET) for rows of boolean arrays.
    yt_pos: human == REGRET; yp_pos: predicted == REGRET; yp_valid: prediction is a valid label."""
    tp = (yt_pos & yp_pos).sum(1); fp = (~yt_pos & yp_pos).sum(1)
    fn = (yt_pos & ~yp_pos).sum(1); tn = (~yt_pos & ~yp_pos & yp_valid).sum(1)
    n = yt_pos.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = (tp + tn) / n
        f1_r = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        # DISAPPOINTMENT as the positive class (invalid predictions count as wrong)
        tp_d = tn; fp_d = (yt_pos & ~yp_pos & yp_valid).sum(1); fn_d = fp + ((~yt_pos) & ~yp_valid).sum(1)
        f1_d = np.where(2 * tp_d + fp_d + fn_d > 0, 2 * tp_d / (2 * tp_d + fp_d + fn_d), 0.0)
        po = acc
        p_h = yt_pos.mean(1); p_p = yp_pos.mean(1); p_pd = (yp_valid & ~yp_pos).mean(1)
        pe = p_h * p_p + (1 - p_h) * p_pd
        kap = (po - pe) / (1 - pe)
        prec_r = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
        rec_r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
        n_neg = (~yt_pos).sum(1)
        spec = np.where(n_neg > 0, tn / np.maximum(n_neg, 1), np.nan)
    return {"accuracy": acc, "macro_f1": (f1_r + f1_d) / 2, "kappa_vs_human": kap, "f1_REGRET": f1_r,
            "precision_REGRET": prec_r, "recall_REGRET": rec_r, "balanced_accuracy": (rec_r + spec) / 2}


def metrics_with_ci(y_true, y_pred, seed=C.SEED, n_boot=4000) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    base = classification_metrics(yt, yp)
    idx = _boot_idx(len(yt), n_boot, seed)
    t = (yt == "REGRET")[idx]; p = (yp == "REGRET")[idx]; v = np.isin(yp, ["REGRET", "DISAPPOINTMENT"])[idx]
    fm = _fast_metrics(t, p, v)
    for key, vals in fm.items():
        vals = vals[~np.isnan(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
        base[f"{key}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
    return base


def ece(prob_pos, y_true_pos, n_bins=10) -> float:
    """Expected calibration error for P(REGRET)."""
    p = np.asarray(prob_pos, float); y = np.asarray(y_true_pos, float)
    ok = ~np.isnan(p); p, y = p[ok], y[ok]
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def fmt(x, nd=3):
    if isinstance(x, tuple):
        return "[" + ", ".join(str(fmt(v, nd)) for v in x) + "]"
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    return x


def to_table(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"statistic": list(d.keys()), "value": [fmt(v) for v in d.values()]})
