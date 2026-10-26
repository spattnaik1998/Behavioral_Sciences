"""
Stage 0 - re-analysis from the summary statistics reported in the paper
(Section 4.4 / Figure 4 / Figure 5). Needs no raw data and no API.

Answers (immediately): R2.5 (Welch test, CIs), R3.4 (power / inconclusive result),
R2.4 + R3.3 (algebraic structure of the RDS composite).

Outputs: results/00_reported_stats/
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import stats_utils as S

OUT = C.RESULTS_DIR / "00_reported_stats"

# Values exactly as printed in the manuscript
REPORTED = dict(n_regret=16, mean_regret=-1.50, sd_regret=1.69,
                n_disap=84, mean_disap=-2.36, sd_disap=1.62,
                t_reported=1.790, p_reported=0.0765, d_reported=0.52,
                r_rating=0.637, n_total=100)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    R = REPORTED
    n1, m1, s1, n2, m2, s2 = R["n_regret"], R["mean_regret"], R["sd_regret"], R["n_disap"], R["mean_disap"], R["sd_disap"]
    stud = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=True)
    welch = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    df_w = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    diff = m1 - m2; se_w = math.sqrt(v1 + v2)
    ci_w = (diff - stats.t.ppf(.975, df_w) * se_w, diff + stats.t.ppf(.975, df_w) * se_w)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    se_s = sp * math.sqrt(1 / n1 + 1 / n2)
    ci_s = (diff - stats.t.ppf(.975, n1 + n2 - 2) * se_s, diff + stats.t.ppf(.975, n1 + n2 - 2) * se_s)
    d = diff / sp; g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    dlo, dhi = S.d_ci(d, n1, n2)

    power_obs = S.power_two_sample(d, n1, n2)
    need_same_ratio = S.n_for_power(d, n2 / n1, 0.80)
    need_balanced = S.n_for_power(d, 1.0, 0.80)
    # power if the whole pool (2,019) is classified with the same 16 % regret share
    n_pool = 2019; n1p = round(n_pool * .16); n2p = n_pool - n1p
    power_pool = S.power_two_sample(d, n1p, n2p)
    power_pool_small = S.power_two_sample(0.2, n1p, n2p)
    # minimum detectable effect at 80 % power with the pool
    mde = next(dd for dd in np.arange(0.01, 2, 0.005) if S.power_two_sample(dd, n1p, n2p) >= .8)

    # Correlation with star rating: Fisher CI
    r, n = R["r_rating"], R["n_total"]
    z, se = np.arctanh(r), 1 / math.sqrt(n - 3)
    r_ci = (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))

    # Internal-consistency check of the reported numbers
    overall_mean_implied = (n1 * m1 + n2 * m2) / (n1 + n2)
    total_ss = (n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2 + n1 * (m1 - overall_mean_implied) ** 2 + n2 * (m2 - overall_mean_implied) ** 2
    overall_sd_implied = math.sqrt(total_ss / (n1 + n2 - 1))
    # pooled SD that WOULD reproduce the reported t = 1.790
    sp_for_reported_t = diff / (R["t_reported"] * math.sqrt(1 / n1 + 1 / n2))
    consistency = {
        "overall_mean_reported": -2.22, "overall_mean_implied_by_group_stats": overall_mean_implied,
        "overall_sd_reported": 1.78, "overall_sd_implied_by_group_stats": overall_sd_implied,
        "student_t_reported": R["t_reported"], "student_t_implied_by_group_stats": stud.statistic,
        "pooled_sd_needed_to_get_reported_t": sp_for_reported_t,
        "verdict": ("Group means are consistent with the overall mean, but the reported group SDs (1.69, 1.62) "
                    "imply an overall SD of ~1.65 and t = 1.93 (p = 0.056), not the reported SD 1.78 and t = 1.790 "
                    "(p = 0.0765). The reported t and overall SD are mutually consistent with a pooled SD of ~1.76, "
                    "so the group SDs printed in Section 4.4 / Figure 4 are probably mis-transcribed. "
                    "Resolve by recomputing from the raw results file."),
    }
    (OUT / "internal_consistency_check.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")

    res = {
        "reported": R,
        "internal_consistency": consistency,
        "student_t_recomputed": stud.statistic, "student_p_recomputed": stud.pvalue,
        "student_ci95_mean_diff": ci_s,
        "welch_t": welch.statistic, "welch_df": df_w, "welch_p": welch.pvalue, "welch_ci95_mean_diff": ci_w,
        "mean_diff": diff, "cohens_d": d, "cohens_d_ci95": (dlo, dhi), "hedges_g": g,
        "power_observed_design_at_d": power_obs,
        "n_needed_80pct_power_same_16_84_ratio": need_same_ratio,
        "n_needed_80pct_power_balanced": need_balanced,
        "power_full_pool_2019_at_observed_d": power_pool,
        "power_full_pool_2019_at_d_0.2": power_pool_small,
        "min_detectable_d_full_pool_80pct": round(float(mde), 3),
        "rating_corr_r": r, "rating_corr_fisher_ci95": r_ci,
        "notes": [
            "The paper's t = 1.790 is Student's pooled-variance t; Welch's t is reported here as requested by R2.5.",
            "The 95% CI for the mean difference includes 0 under both tests, so the result is inconclusive (R3.4).",
            "The CI for Cohen's d spans from a negligible-to-negative effect to a large one; d = 0.52 is not a precise estimate.",
            "Power figures assume the observed d is the true effect; they illustrate why n = 100 (16 vs 84) was underpowered.",
        ],
    }
    (OUT / "reanalysis_from_reported_stats.json").write_text(json.dumps(res, indent=2, default=lambda o: S.fmt(o) if not isinstance(o, tuple) else list(o)), encoding="utf-8")
    S.to_table({k: v for k, v in res.items() if k not in ("reported", "notes")}).to_csv(OUT / "reanalysis_from_reported_stats.csv", index=False)

    # Power curve for the paper
    rows = []
    for n_total in [100, 200, 300, 500, 1000, 2019]:
        a = round(n_total * .16); b = n_total - a
        rows.append({"n_total": n_total, "n_regret": a, "n_disappointment": b,
                     **{f"power_d_{dd}": round(S.power_two_sample(dd, a, b), 3) for dd in (0.2, 0.3, 0.5, 0.52)}})
    pd.DataFrame(rows).to_csv(OUT / "power_table.csv", index=False)

    # RDS algebra (R2.4, R3.3): verify the closed form on every combination of the 1-7 inputs
    import itertools
    grid = pd.DataFrame(list(itertools.product(range(1, 8), repeat=4)),
                        columns=["rds_counterfactual", "rds_responsibility", "rds_controllability", "rds_affect"])
    from tasks import rds_composite
    paper = rds_composite(grid)
    closed = 0.4 * grid.rds_counterfactual + 0.6 * grid.rds_responsibility + 0.6 * grid.rds_controllability - 0.4 * grid.rds_affect - 4.8
    algebra = {
        "paper_formula": "Regret=0.4CF+0.3R+0.3C ; Disap=0.4A+0.3(8-R)+0.3(8-C) ; RDS=Regret-Disap",
        "closed_form": "RDS = 0.4*CF + 0.6*R + 0.6*C - 0.4*A - 4.8",
        "max_abs_difference_over_all_2401_inputs": float((paper - closed).abs().max()),
        "range_of_RDS": [float(paper.min()), float(paper.max())],
        "value_when_all_dimensions_at_midpoint_4": float(rds_composite(pd.DataFrame([[4, 4, 4, 4]], columns=grid.columns)).iloc[0]),
        "implication": ("The reverse-scoring (8 - x) does not add information: responsibility and controllability enter "
                        "with total weight 0.6 each, counterfactual +0.4, affect -0.4. The score is a fixed linear index "
                        "whose weights the authors chose; it is zero at the scale midpoint."),
    }
    (OUT / "rds_formula_algebra.json").write_text(json.dumps(algebra, indent=2), encoding="utf-8")
    return {"reanalysis": res, "algebra": algebra}


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))
