"""
Publication figures (PNG, 300 dpi) -> results/07_figures/

Palette: validated reference categorical slots (blue / orange / aqua), used
consistently: DISAPPOINTMENT = blue, REGRET = orange, third series = aqua.
Every figure function checks its inputs and is skipped if they are missing.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

import config as C

OUT = C.RESULTS_DIR / "07_figures"
A = C.RESULTS_DIR / "06_analysis"
IN = C.RESULTS_DIR / "03_llm_outputs"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COL = {"DISAPPOINTMENT": BLUE, "REGRET": ORANGE}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figure:", name)


def _ci(s):
    s = str(s).strip("[]").split(",")
    return float(s[0]), float(s[1])


# --------------------------------------------------------------------------
# Fig. R1 - architecture: standard RAG vs Speculative RAG (R1.2, R3.1, R4.2, R4.7)
# --------------------------------------------------------------------------
def fig_architecture():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
    for ax in axes:
        ax.set_xlim(-2, 102); ax.set_ylim(0, 40); ax.axis("off")

    def box(ax, x, y, w, h, text, fc="#f3f2ee", ec=MUTED, bold=False, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, color=color,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=INK2, lw=0.9))

    # (a) standard RAG (conditions C2/C3)
    ax = axes[0]
    ax.set_title("(a) Standard RAG classifier (proposed system, C3)", loc="left", color=INK, fontweight="bold")
    box(ax, 0, 12, 12, 9, "Customer\nreview")
    box(ax, 17, 27, 20, 9, "Knowledge base:\nliterature PDFs,\n500/50-char chunks")
    box(ax, 17, 12, 20, 9, "Embed review\n(OpenAI 3-small)\n+ FAISS search")
    box(ax, 42, 12, 13, 9, "Top-k = 5\nchunks")
    box(ax, 60, 12, 19, 9, "GPT-4o, T = 0\ncontext + review\n+ definitions", fc="#e8f0fb", ec=BLUE)
    box(ax, 84, 12, 15, 9, "1 label +\nconfidence")
    arrow(ax, 12, 16.5, 17, 16.5); arrow(ax, 27, 27, 27, 21); arrow(ax, 37, 16.5, 42, 16.5)
    arrow(ax, 55, 16.5, 60, 16.5); arrow(ax, 79, 16.5, 84, 16.5)
    ax.plot([6, 6, 69.5], [12, 4, 4], color=INK2, lw=0.9)
    arrow(ax, 69.5, 4, 69.5, 12)
    ax.text(30, 1.2, "review text also inserted into the prompt", fontsize=6.5, color=MUTED)

    # (b) speculative RAG (condition C4)
    ax = axes[1]
    ax.set_title("(b) Speculative RAG (C4): parallel drafts by a small model, selection by a large verifier",
                 loc="left", color=INK, fontweight="bold")
    box(ax, 0, 15, 11, 9, "Customer\nreview")
    box(ax, 15, 15, 13, 9, "Top-n = 10\nchunks")
    box(ax, 32, 15, 13, 9, "k-means\nk = 3 clusters")
    ys = [31, 22, 13, 4]
    for i, y in enumerate(ys):
        box(ax, 48, y, 16, 6.5, f"Subset {i+1}\n(1 chunk/cluster)", fc="#f3f2ee")
        box(ax, 67, y, 13, 6.5, f"Drafter\nGPT-4o-mini", fc="#fdeee7", ec=ORANGE)
        arrow(ax, 45, 19.5, 48, y + 3.2); arrow(ax, 64, y + 3.2, 67, y + 3.2); arrow(ax, 80, y + 3.2, 84, 19.5)
    box(ax, 84, 13, 16, 13, "Verifier GPT-4o\nscore = ρdraft ×\nρself-consist. ×\nρself-reflect.\n→ best draft", fc="#e8f0fb", ec=BLUE)
    arrow(ax, 11, 19.5, 15, 19.5); arrow(ax, 28, 19.5, 32, 19.5)
    ax.text(49, 38.6, "each draft = rationale + label, from a different evidence subset (the 'speculative' step)",
            fontsize=6.5, color=MUTED)
    fig.tight_layout()
    _save(fig, "R1_architecture_rag_vs_speculative.png")


# --------------------------------------------------------------------------
# Fig. R2 - power of the RDS group comparison by sample size (R3.4)
# --------------------------------------------------------------------------
def fig_power():
    f = C.RESULTS_DIR / "00_reported_stats" / "power_table.csv"
    if not f.exists():
        return
    import stats_utils as S
    ns = np.arange(50, 2101, 25)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for d, col, lab in [(0.2, AQUA, "d = 0.2 (small)"), (0.52, ORANGE, "d = 0.52 (paper's estimate)")]:
        p = [S.power_two_sample(d, round(n * .16), n - round(n * .16)) for n in ns]
        ax.plot(ns, p, color=col, lw=2)
        ax.text(ns[-1] + 20, p[-1], lab, color=INK2, va="center", fontsize=8)
    ax.axhline(0.8, color=MUTED, lw=1, ls="--"); ax.text(60, 0.82, "80% power", color=MUTED, fontsize=7.5)
    for n, lab in [(100, "paper\nn = 100"), (2019, "full pool\nn = 2,019")]:
        ax.axvline(n, color=GRID, lw=1.2, zorder=0); ax.text(n + 15, 0.05, lab, fontsize=7.5, color=INK2)
    ax.set_xlim(0, 2700); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Reviews classified (16% regret, as in the paper)"); ax.set_ylabel("Power (two-sided, α = 0.05)")
    ax.set_title("Power to detect an RDS difference between regret and disappointment", loc="left", color=INK)
    _save(fig, "R2_power_curve.png")


# --------------------------------------------------------------------------
# Fig. R3 - regret prevalence by system with 95% CI (R3.5, R4.5-7)
# --------------------------------------------------------------------------
def fig_prevalence():
    f = A / "A_prevalence_by_system.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).iloc[::-1]
    gold = A / "B_human_prevalence.json"
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(d) + 1.2))
    y = np.arange(len(d))
    lo = [_ci(c)[0] for c in d["pct_regret_ci95"]]; hi = [_ci(c)[1] for c in d["pct_regret_ci95"]]
    ax.hlines(y, lo, hi, color=ORANGE, lw=2)
    ax.plot(d["pct_regret"], y, "o", color=ORANGE, ms=6, mec="white", mew=1.5)
    for yi, v in zip(y, d["pct_regret"]):
        ax.text(v, yi + 0.28, f"{v:.1f}%", ha="center", fontsize=7.5, color=INK2)
    if gold.exists():
        g = json.loads(gold.read_text(encoding="utf-8"))
        ax.axvspan(g["ci95"][0], g["ci95"][1], color=GRID, zorder=0)
        ax.axvline(g["pct"], color=INK2, lw=1, ls="--")
        ax.text(g["pct"], -0.85, f"human gold standard {g['pct']:.1f}% (band = 95% CI)", fontsize=7.5, color=INK2, ha="center")
        ax.set_ylim(-1.2, len(d) - 0.4)
    ax.set_ylim(ax.get_ylim()[0], len(d) - 0.35)
    ax.set_yticks(y); ax.set_yticklabels(d["description"]); ax.set_xlabel("Reviews classified as REGRET (%), 95% Wilson CI")
    ax.set_title("Regret prevalence in the 2,019-review pool, by system", loc="left", color=INK)
    _save(fig, "R3_prevalence_by_system.png")


# --------------------------------------------------------------------------
# Fig. R4 - agreement with human gold (macro-F1, kappa) with bootstrap CI
# --------------------------------------------------------------------------
def fig_accuracy():
    f = A / "B_accuracy_vs_human.csv"
    if not f.exists():
        return
    d = pd.read_csv(f); d = d[d["subset"].str.startswith("all")].iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 0.42 * len(d) + 1.3), sharey=True)
    y = np.arange(len(d))
    for ax, m, title in [(axes[0], "macro_f1", "Macro-F1"), (axes[1], "kappa_vs_human", "Cohen's κ vs human")]:
        lo = [_ci(c)[0] for c in d[f"{m}_ci95"]]; hi = [_ci(c)[1] for c in d[f"{m}_ci95"]]
        ax.hlines(y, lo, hi, color=BLUE, lw=2); ax.plot(d[m], y, "o", color=BLUE, ms=6, mec="white", mew=1.5)
        for yi, v in zip(y, d[m]):
            ax.text(v, yi + 0.28, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left", color=INK)
        if m == "kappa_vs_human":
            ax.axvline(0, color=MUTED, lw=1)
    axes[0].set_yticks(y); axes[0].set_yticklabels(d["description"])
    fig.supxlabel("Point estimate and 95% bootstrap CI on the human-annotated gold set", fontsize=8, color=INK2)
    _save(fig, "R4_accuracy_vs_human.png")


# --------------------------------------------------------------------------
# Fig. R5 - RDS by label: LLM label vs HUMAN label (R2.3, R3.2)
# --------------------------------------------------------------------------
def fig_rds_by_label():
    mf = IN / "pool_master.csv"; gf = C.DATA_DIR / "gold_labels.csv"
    if not mf.exists():
        return
    m = pd.read_csv(mf)
    if "rds_paper" not in m:
        return
    lab = f"{C.MAIN_CONDITION}__label"
    panels = [("LLM label (C3), full pool", m, lab)]
    if gf.exists():
        g = m.merge(pd.read_csv(gf)[["review_uid", "forced_choice"]], on="review_uid")
        panels.append(("Human label, gold set", g, "forced_choice"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), sharey=True, squeeze=False)
    r = np.random.default_rng(0)
    for ax, (title, df, col) in zip(axes[0], panels):
        for i, L in enumerate(["REGRET", "DISAPPOINTMENT"]):
            v = df.loc[df[col] == L, "rds_paper"].dropna()
            if not len(v):
                continue
            ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
                       boxprops=dict(fc="white", ec=COL[L], lw=1.2), medianprops=dict(color=COL[L], lw=2),
                       whiskerprops=dict(color=COL[L]), capprops=dict(color=COL[L]))
            ax.scatter(i + r.uniform(-.17, .17, len(v)), v, s=5 if len(v) > 300 else 9, color=COL[L], alpha=.35, lw=0)
            ax.text(i, ax.get_ylim()[1] if False else 6.4, f"n = {len(v)}\nmean {v.mean():.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["REGRET", "DISAPPOINTMENT"]); ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(-6.5, 7.8)
    axes[0][0].set_ylabel("RDS differential (paper formula)")
    _save(fig, "R5_rds_by_llm_and_human_label.png")


# --------------------------------------------------------------------------
# Fig. R6 - RDS by star rating (range restriction made visible) (R2.6, R3.2)
# --------------------------------------------------------------------------
def fig_rds_rating():
    mf = IN / "pool_master.csv"
    if not mf.exists():
        return
    m = pd.read_csv(mf)
    if "rds_paper" not in m:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    r = np.random.default_rng(1)
    for i, rt in enumerate([1, 2, 3]):
        v = m.loc[m["reviews.rating"] == rt, "rds_paper"].dropna()
        ax.boxplot(v, positions=[i], widths=.5, showfliers=False, patch_artist=True,
                   boxprops=dict(fc="white", ec=BLUE, lw=1.2), medianprops=dict(color=BLUE, lw=2),
                   whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
        ax.scatter(i + r.uniform(-.2, .2, len(v)), v + r.uniform(-.05, .05, len(v)), s=4, color=BLUE, alpha=.25, lw=0)
        ax.text(i, 6.4, f"n = {len(v)}", ha="center", fontsize=7.5, color=INK2)
    f = A / "D_rating_correlations.csv"
    if f.exists():
        c = pd.read_csv(f).iloc[0]
        ax.text(0.02, 0.02, f"Spearman ρ = {c['spearman_rho']:.2f} {c['spearman_ci95_boot']}", transform=ax.transAxes,
                fontsize=7.5, color=INK2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["1 star", "2 stars", "3 stars"]); ax.set_ylim(-6.5, 7.6)
    ax.axhline(0, color=MUTED, lw=1, ls="--"); ax.set_ylabel("RDS differential")
    ax.set_title("RDS by star rating (ratings 4-5 excluded by design)", loc="left", color=INK)
    _save(fig, "R6_rds_by_rating.png")


# --------------------------------------------------------------------------
# Fig. R7 - weight sensitivity (R2.4, R3.3)
# --------------------------------------------------------------------------
def fig_weights():
    f = A / "D_weight_sensitivity_random_draws.csv"; s = A / "D_weight_sensitivity.csv"
    if not f.exists() or not s.exists():
        return
    d = pd.read_csv(f); sp = pd.read_csv(s)
    col = "hedges_g_llm_label"
    if col not in d:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.hist(d[col].dropna(), bins=40, color=BLUE, alpha=.85, ec="white", lw=0.5)
    paper = sp.loc[sp.weights.str.startswith("paper"), col].iloc[0]
    ax.axvline(paper, color=ORANGE, lw=2); ax.text(paper, ax.get_ylim()[1] * .92, "  paper's weights", color=INK2, fontsize=7.5)
    q = d[col].quantile([.025, .975]).values
    ax.set_xlabel("Hedges' g, REGRET vs DISAPPOINTMENT (LLM label)")
    ax.set_ylabel("Random weight sets (of 2,000)")
    ax.set_title(f"RDS group separation under random weights (95% of draws in [{q[0]:.2f}, {q[1]:.2f}])", loc="left", color=INK, fontsize=9)
    _save(fig, "R7_weight_sensitivity.png")


# --------------------------------------------------------------------------
# Fig. R8 - LLM vs human dimension ratings (R2.3, R3.2, R3.3)
# --------------------------------------------------------------------------
def fig_dims():
    mf = IN / "pool_master.csv"; gf = C.DATA_DIR / "gold_labels.csv"
    if not (mf.exists() and gf.exists()):
        return
    m = pd.read_csv(mf).merge(pd.read_csv(gf), on="review_uid")
    dims = ["counterfactual", "responsibility", "controllability", "affect"]
    if not all(f"rds_{x}" in m for x in dims):
        return
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), sharex=True, sharey=True)
    r = np.random.default_rng(2)
    from scipy.stats import spearmanr
    for ax, dd in zip(axes, dims):
        x, y = m[f"human_{dd}"], m[f"rds_{dd}"]; ok = x.notna() & y.notna()
        ax.scatter(x[ok] + r.uniform(-.15, .15, ok.sum()), y[ok] + r.uniform(-.15, .15, ok.sum()), s=9, color=BLUE, alpha=.5, lw=0)
        ax.plot([1, 7], [1, 7], color=MUTED, lw=1, ls="--")
        rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 3 else np.nan
        ax.set_title(f"{dd}\nρ = {rho:.2f}", loc="left", color=INK, fontsize=8.5)
        ax.set_xlim(.5, 7.5); ax.set_ylim(.5, 7.5); ax.set_xticks(range(1, 8)); ax.set_yticks(range(1, 8))
    axes[0].set_ylabel("GPT-4o rating"); fig.supxlabel("Human rating (mean of two coders)", fontsize=8.5, color=INK2)
    _save(fig, "R8_llm_vs_human_dimensions.png")


def run():
    for fn in [fig_architecture, fig_power, fig_prevalence, fig_accuracy, fig_rds_by_label, fig_rds_rating, fig_weights, fig_dims]:
        try:
            fn()
        except Exception as e:
            print(f"  [figures] {fn.__name__} failed: {e!r}")


if __name__ == "__main__":
    run()
