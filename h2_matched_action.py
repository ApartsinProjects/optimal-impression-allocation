"""E1: matched-action (replay-style) OUTCOME bracket for the Taobao replay.

For each walk-forward fold of the pre-committed config (tier8_only, top-100),
re-execute the deterministic serving passes (greedy and static plan) recording
the per-event chosen campaign, and compare it to the campaign the production
system actually served (the logged action). On the matched subset the policy's
value is observable from REALIZED clicks, with no click model.

Reported per policy: match rate; realized CTR on matched events; model-scored
CTR on the same matched events (calibration link); and the matched-outcome
planned-vs-greedy comparison with user-agnostic binomial errors.

Caveat stated in the paper: the matched subset is selected by agreement with
the logging policy, so it is biased toward the logging policy's preferences;
it is a bracket and a calibration check, not an unbiased policy value.
"""
import os
import numpy as np
import duckdb
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research"
DB = os.path.join(BASE, "data", "taobao", "prep", "taobao.duckdb")


def solve_lp(nC, nS, supply, quota, P):
    idx = [(c, s) for c in range(nC) for s in range(nS)]
    cost = np.array([-P[c, s] for c, s in idx])
    A = lil_matrix((nS + nC, len(idx)))
    for j, (c, s) in enumerate(idx):
        A[s, j] = 1.0
        A[nS + c, j] = 1.0
    b = np.concatenate([supply, quota]).astype(float)
    r = linprog(cost, A_ub=A.tocsr(), b_ub=b, bounds=(0, None), method="highs")
    assert r.status == 0
    x = np.zeros((nC, nS))
    for j, (c, s) in enumerate(idx):
        x[c, s] = r.x[j]
    return x


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute("""CREATE OR REPLACE TEMP VIEW ev AS
      SELECT raw.user AS user_id, raw.clk, raw.time_stamp,
             date_part('day', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
             ad.campaign_id FROM raw JOIN ad USING (adgroup_id)""")
    agg = {"g": [0, 0, 0.0], "p": [0, 0, 0.0]}  # matches, clicks, model-sum
    n_events_total = 0
    for d in [9, 10, 11, 12, 13]:
        z = np.load(os.path.join(BASE, "results", f"h2_wf3_t8_d{d}_assign.npz"))
        P, quota, forecast = z["P"], z["quota"], z["forecast"]
        nC, nS = P.shape
        con.execute(f"""CREATE OR REPLACE TEMP TABLE t8 AS
          SELECT user_id, CASE WHEN count(*)>=200 THEN 7 WHEN count(*)>=100 THEN 6
            WHEN count(*)>=50 THEN 5 WHEN count(*)>=30 THEN 4 WHEN count(*)>=20 THEN 3
            WHEN count(*)>=10 THEN 2 WHEN count(*)>=5 THEN 1 ELSE 0 END AS vt
          FROM ev WHERE day BETWEEN 6 AND {d-1} GROUP BY 1""")
        top = [r[0] for r in con.execute(f"""SELECT campaign_id FROM ev
          WHERE day BETWEEN 6 AND {d-1} GROUP BY 1 ORDER BY count(*) DESC LIMIT 100""").fetchall()]
        camp_ix = {c: i for i, c in enumerate(top)}
        df = con.execute("""SELECT COALESCE(t.vt,0) AS seg, ev.campaign_id, ev.clk
          FROM ev LEFT JOIN t8 t USING (user_id)
          WHERE day = ? AND campaign_id IN (SELECT UNNEST(?)) ORDER BY time_stamp""",
          [d, top]).df()
        segs = df["seg"].to_numpy()
        logged = df["campaign_id"].map(camp_ix).to_numpy()
        clk = df["clk"].to_numpy()
        n_events_total += len(segs)
        plan = solve_lp(nC, nS, forecast, quota, P)
        order_p = np.argsort(-P, axis=0)

        def run(use_plan):
            served = np.zeros(nC)
            remaining = plan.copy() if use_plan else None
            choice = np.empty(len(segs), dtype=int)
            for i, s in enumerate(segs):
                c = None
                if use_plan:
                    cand = np.where(remaining[:, s] > 0.5)[0]
                    cand = cand[served[cand] < quota[cand]]
                    if len(cand):
                        c = cand[np.argmax(remaining[cand, s])]
                        remaining[c, s] -= 1.0
                if c is None:
                    for cc in order_p[:, s]:
                        if served[cc] < quota[cc]:
                            c = cc
                            break
                served[c] += 1
                choice[i] = c
            return choice

        for key, use_plan in [("g", False), ("p", True)]:
            ch = run(use_plan)
            m = ch == logged
            agg[key][0] += int(m.sum())
            agg[key][1] += int(clk[m].sum())
            agg[key][2] += float(z["P_eval"][ch[m], segs[m]].sum())

    print(f"total replay events: {n_events_total:,}")
    res = {}
    for key, name in [("g", "greedy"), ("p", "planned")]:
        n, k, msum = agg[key]
        ctr = k / n
        se = (ctr * (1 - ctr) / n) ** 0.5
        res[key] = (n, ctr, se, msum / n)
        print(f"{name}: match rate {n/n_events_total*100:.1f}%  matched realized CTR "
              f"{ctr*100:.3f}% (se {se*100:.3f})  model-scored CTR on same subset {msum/n*100:.3f}%")
    d_ctr = res["p"][1] - res["g"][1]
    d_se = (res["p"][2] ** 2 + res["g"][2] ** 2) ** 0.5
    print(f"matched-outcome planned-vs-greedy: {d_ctr/res['g'][1]*100:+.2f}% "
          f"(z = {d_ctr/d_se:+.2f}) [subsets differ; bracket, not unbiased value]")


if __name__ == "__main__":
    main()
