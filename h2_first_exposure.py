"""(d) Fatigue sensitivity: rescore saved assignments under a FIRST-EXPOSURE
eval table (each user's first impression of each campaign only), removing
repeat-exposure composition from the scorer. If the headline holds, frequency
composition does not drive it.

Usage: python h2_first_exposure.py wf3_t8
"""
import os, sys
import numpy as np
import duckdb

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research"
DB = os.path.join(BASE, "data", "taobao", "prep", "taobao.duckdb")
ALPHA, MINC = 20.0, 30
prefix = sys.argv[1] if len(sys.argv) > 1 else "wf3_t8"

con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA memory_limit='3GB'")
con.execute("""CREATE OR REPLACE TEMP VIEW ev AS
  SELECT raw.user AS user_id, raw.clk, raw.time_stamp,
         date_part('day', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
         ad.campaign_id FROM raw JOIN ad USING (adgroup_id)""")

tot = {"g": [0.0, 0.0], "p": [0.0, 0.0]}  # [raw, first-exposure] scored values
for d in [9, 10, 11, 12, 13]:
    z = np.load(os.path.join(BASE, "results", f"h2_{prefix}_d{d}_assign.npz"))
    Pe = z["P_eval"]
    nC, nS = Pe.shape
    con.execute(f"""CREATE OR REPLACE TEMP TABLE t8 AS
      SELECT user_id, CASE WHEN count(*)>=200 THEN 7 WHEN count(*)>=100 THEN 6
        WHEN count(*)>=50 THEN 5 WHEN count(*)>=30 THEN 4 WHEN count(*)>=20 THEN 3
        WHEN count(*)>=10 THEN 2 WHEN count(*)>=5 THEN 1 ELSE 0 END AS vt
      FROM ev WHERE day BETWEEN 6 AND {d-1} GROUP BY 1""")
    top = [r[0] for r in con.execute(f"""SELECT campaign_id FROM ev
      WHERE day BETWEEN 6 AND {d-1} GROUP BY 1 ORDER BY count(*) DESC LIMIT 100""").fetchall()]
    camp_ix = {c: i for i, c in enumerate(top)}
    # first exposure per (user, campaign) on the serving day only
    fe = con.execute("""
      WITH r AS (SELECT ev.user_id, ev.campaign_id, ev.clk, COALESCE(t.vt,0) AS seg,
                 row_number() OVER (PARTITION BY ev.user_id, ev.campaign_id
                                    ORDER BY ev.time_stamp) rn
                 FROM ev LEFT JOIN t8 t USING (user_id)
                 WHERE day = ? AND campaign_id IN (SELECT UNNEST(?)))
      SELECT campaign_id, seg, sum(clk) s, count(*) c FROM r WHERE rn = 1
      GROUP BY 1,2""", [d, top]).df()
    g = float(fe["s"].sum() / fe["c"].sum())
    P_fe = np.full((nC, nS), g)
    camp = fe.groupby("campaign_id").agg(s=("s", "sum"), c=("c", "sum"))
    base = np.full(nC, g)
    for cid, row in camp.iterrows():
        if row["c"] >= MINC:
            base[camp_ix[cid]] = (row["s"] + ALPHA * g) / (row["c"] + ALPHA)
    P_fe[:] = base[:, None]
    for cid, sg, s_, c_ in fe.itertuples(index=False):
        if c_ >= MINC:
            ci = camp_ix[cid]
            P_fe[ci, int(sg)] = (s_ + ALPHA * base[ci]) / (c_ + ALPHA)
    for key, arr in [("g", "g_assign"), ("p", "p_assign")]:
        tot[key][0] += float((z[arr] * Pe).sum())
        tot[key][1] += float((z[arr] * P_fe).sum())

for i, name in [(0, "raw eval table"), (1, "first-exposure eval table")]:
    lift = (tot["p"][i] / tot["g"][i] - 1) * 100
    print(f"planned vs greedy under {name}: {lift:+.2f}%")
