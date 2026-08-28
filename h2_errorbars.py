"""Closed-form error bars for replay lift under eval-table estimation noise.

The lift statistic is dV = sum_cs (n^planned_cs - n^greedy_cs) * p_eval_cs.
Each p_eval_cs is a smoothed binomial estimate with
Var(p_hat) ~= n * p(1-p) / (n + alpha)^2 (cells below min_count share the
campaign base whose own variance is negligible at campaign-level counts; we
treat those cells as fixed, which UNDERSTATES se slightly).

se(dV) = sqrt( sum_cs dn_cs^2 * Var(p_hat_cs) )   [cells independent]
Reported: dV, se, z = dV/se for planned-vs-greedy, per run.
Usage: python h2_errorbars.py tag1 tag2 ...
"""
import sys, json, os
import numpy as np

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research\results"
ALPHA = 20.0

for tag in sys.argv[1:]:
    z = np.load(os.path.join(BASE, f"h2_{tag}_assign.npz"))
    P_eval, N, ga, pa = z["P_eval"], z["N_eval"], z["g_assign"], z["p_assign"]
    dn = pa - ga
    dv = float((dn * P_eval).sum())
    var_cell = N * P_eval * (1 - P_eval) / (N + ALPHA) ** 2
    se = float(np.sqrt((dn ** 2 * var_cell).sum()))
    g_val = float((ga * P_eval).sum())
    print(f"{tag}: dV={dv:+9.1f}  se={se:7.1f}  z={dv/se:+.2f}   "
          f"lift={(dv/g_val)*100:+.2f}%  se_lift={(se/g_val)*100:.2f}pp")
