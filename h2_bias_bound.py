"""Bias-bound analysis for dual-model replay scoring (reviewer weakness #1).

The eval table p_hat(c, s) is fitted on logged exposures chosen by the
production policy; within a cell, campaign c's logged users may differ from
the segment average on click-relevant covariates. We correct along the
strongest observable covariate NOT used in the segmentation: the user's
personal train-window CTR propensity q(u) (smoothed toward global, B=20).

Adjusted scorer: P_adj(c,s) = P_eval(c,s) - [E(q | logged c,s) - E(q | s)],
a first-order standardization that removes the part of each cell's rate
attributable to observable within-cell selection. All saved policy
assignments are re-scored under P_adj; the raw-vs-adjusted lift difference
bounds the observable-selection bias of the headline.

Runs on the pre-committed config folds (tier8_only, top-100, walk-forward).
Usage: python h2_bias_bound.py wf3_t8
"""
import sys, os, json
import numpy as np
import duckdb

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research"
DB = os.path.join(BASE, "data", "taobao", "prep", "taobao.duckdb")
prefix = sys.argv[1] if len(sys.argv) > 1 else "wf3_t8"

con = duckdb.connect(DB, read_only=True)
con.execute("""CREATE OR REPLACE TEMP VIEW ev AS
  SELECT raw.user AS user_id, raw.clk,
         date_part('day', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
         ad.campaign_id
  FROM raw JOIN ad USING (adgroup_id)""")

tot_raw = {"g": 0.0, "p": 0.0, "d": 0.0}
tot_adj = {"g": 0.0, "p": 0.0, "d": 0.0}
shifts = []
for d in [9, 10, 11, 12, 13]:
    z = np.load(os.path.join(BASE, "results", f"h2_{prefix}_d{d}_assign.npz"))
    P_eval, quota = z["P_eval"], z["quota"]
    nC, nS = P_eval.shape
    TD = f"day BETWEEN 6 AND {d-1}"
    # per-user train propensity q(u), smoothed; and 8-tier from train volume
    con.execute(f"""CREATE OR REPLACE TEMP TABLE uq AS
      SELECT user_id,
             (sum(clk) + 20.0 * (SELECT avg(clk) FROM ev WHERE {TD}))
               / (count(*) + 20.0) AS q,
             CASE WHEN count(*)>=200 THEN 7 WHEN count(*)>=100 THEN 6
                  WHEN count(*)>=50 THEN 5 WHEN count(*)>=30 THEN 4
                  WHEN count(*)>=20 THEN 3 WHEN count(*)>=10 THEN 2
                  WHEN count(*)>=5 THEN 1 ELSE 0 END AS vt
      FROM ev WHERE {TD} GROUP BY 1""")
    g_ctr = con.execute(f"SELECT avg(clk) FROM ev WHERE {TD}").fetchone()[0]
    # replay events of the fold's top-100 campaigns, with seg (=tier) and q
    top = con.execute(f"""SELECT campaign_id FROM ev WHERE {TD}
      GROUP BY 1 ORDER BY count(*) DESC LIMIT 100""").df()["campaign_id"]
    camp_ix = {c: i for i, c in enumerate(top)}
    rows = con.execute(f"""
      SELECT ev.campaign_id, COALESCE(uq.vt, 0) AS seg,
             avg(COALESCE(uq.q, {g_ctr})) AS mq, count(*) AS n
      FROM ev LEFT JOIN uq USING (user_id)
      WHERE day = {d} AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1, 2""",
      [list(map(int, top))]).df()
    seg_q = con.execute(f"""
      SELECT COALESCE(uq.vt, 0) AS seg, avg(COALESCE(uq.q, {g_ctr})) AS mq
      FROM ev LEFT JOIN uq USING (user_id) WHERE day = {d}
      AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1""",
      [list(map(int, top))]).df().set_index("seg")["mq"]
    delta = np.zeros((nC, nS))
    for cid, sg, mq, n in rows.itertuples(index=False):
        if cid in camp_ix and n >= 30:
            delta[camp_ix[cid], int(sg)] = mq - seg_q.get(int(sg), g_ctr)
            shifts.append(mq - seg_q.get(int(sg), g_ctr))
    P_adj = np.clip(P_eval - delta, 1e-5, 1.0)
    for key, arr in [("g", "g_assign"), ("p", "p_assign"), ("d", "d_assign")]:
        tot_raw[key] += float((z[arr] * P_eval).sum())
        tot_adj[key] += float((z[arr] * P_adj).sum())

shifts = np.array(shifts)
print(f"within-cell covariate shift delta (pp): mean {shifts.mean()*100:+.3f}, "
      f"|mean| {np.abs(shifts).mean()*100:.3f}, p95 |.| {np.percentile(np.abs(shifts),95)*100:.3f}")
for name, a, b in [("planned_vs_greedy", "p", "g"), ("planned_vs_dual", "p", "d")]:
    lr = (tot_raw[a] / tot_raw[b] - 1) * 100
    la = (tot_adj[a] / tot_adj[b] - 1) * 100
    print(f"{name}: raw lift {lr:+.2f}%  adjusted lift {la:+.2f}%  bias-bound delta {la-lr:+.2f}pp")
