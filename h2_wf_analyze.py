"""Combine walk-forward folds: granularity table + paired planned-vs-dual.

For each config prefix, loads h2_<prefix>_d{9..13}_assign.npz, combines
dV and closed-form se across folds for planned-vs-greedy, dual-vs-greedy,
and planned-vs-dual.
Usage: python h2_wf_analyze.py wf2_td wf2_tier_only wf2_demo_only wf2_t500
"""
import sys, os
import numpy as np

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research\results"
ALPHA = 20.0
DAYS = [9, 10, 11, 12, 13]

for prefix in sys.argv[1:]:
    stats = {}
    for pair in [("p_assign", "g_assign", "planned_vs_greedy"),
                 ("d_assign", "g_assign", "dual_vs_greedy"),
                 ("p_assign", "d_assign", "planned_vs_dual")]:
        a, b, name = pair
        dv = se2 = base = 0.0
        folds = []
        ok = True
        for d in DAYS:
            f = os.path.join(BASE, f"h2_{prefix}_d{d}_assign.npz")
            if not os.path.exists(f):
                ok = False
                break
            z = np.load(f)
            if a not in z or b not in z:
                ok = False
                break
            Pe, N = z["P_eval"], z["N_eval"]
            dn = z[a] - z[b]
            v = float((dn * Pe).sum())
            s2 = float((dn ** 2 * N * Pe * (1 - Pe) / (N + ALPHA) ** 2).sum())
            gb = float((z[b] * Pe).sum())
            dv += v; se2 += s2; base += gb
            folds.append(v / gb * 100)
        if not ok:
            print(f"{prefix} {name}: missing files/keys, skipped")
            continue
        se = se2 ** 0.5
        stats[name] = (dv / base * 100, se / base * 100, dv / se, folds)
    for name, (lift, se, zz, folds) in stats.items():
        print(f"{prefix:>14} {name:>18}: lift={lift:+.2f}%  se={se:.2f}pp  z={zz:+.2f}  "
              f"folds=[{', '.join(f'{x:+.1f}' for x in folds)}]")
