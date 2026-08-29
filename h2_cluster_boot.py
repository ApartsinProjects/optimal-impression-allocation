"""E5: user-clustered inference for replay lifts (Poisson bootstrap over users).

The closed-form binomial se treats impressions as independent; users contribute
many impressions each. Here the serving-window eval table is re-fitted under
B Poisson(1) user weights per fold; assignment matrices stay fixed; each
comparison's dv is recomputed under the resampled table. Reported: bootstrap
se and z combined across folds, next to the closed form and the across-fold
t on 4 df.

Usage: python h2_cluster_boot.py wf3_t8   (pre-committed config folds)
"""
import os, sys
import numpy as np
import duckdb
from scipy.sparse import csr_matrix

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research"
DB = os.path.join(BASE, "data", "taobao", "prep", "taobao.duckdb")
ALPHA, MINC, B = 20.0, 30, 200
prefix = sys.argv[1] if len(sys.argv) > 1 else "wf3_t8"

rng = np.random.default_rng(0)
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA memory_limit='3GB'")
con.execute("""CREATE OR REPLACE TEMP VIEW ev AS
  SELECT raw.user AS user_id, raw.clk,
         date_part('day', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
         ad.campaign_id FROM raw JOIN ad USING (adgroup_id)""")

comps = [("p_assign", "g_assign", "planned_vs_greedy"),
         ("p_assign", "d_assign", "planned_vs_dual")]
boot = {n: np.zeros(B) for _, _, n in comps}
point = {n: 0.0 for _, _, n in comps}
base = {n: 0.0 for _, _, n in comps}
fold_lifts = {n: [] for _, _, n in comps}

for d in [9, 10, 11, 12, 13]:
    z = np.load(os.path.join(BASE, "results", f"h2_{prefix}_d{d}_assign.npz"))
    nC, nS = z["P_eval"].shape
    con.execute(f"""CREATE OR REPLACE TEMP TABLE t8 AS
      SELECT user_id, CASE WHEN count(*)>=200 THEN 7 WHEN count(*)>=100 THEN 6
        WHEN count(*)>=50 THEN 5 WHEN count(*)>=30 THEN 4 WHEN count(*)>=20 THEN 3
        WHEN count(*)>=10 THEN 2 WHEN count(*)>=5 THEN 1 ELSE 0 END AS vt
      FROM ev WHERE day BETWEEN 6 AND {d-1} GROUP BY 1""")
    top = [r[0] for r in con.execute(f"""SELECT campaign_id FROM ev
      WHERE day BETWEEN 6 AND {d-1} GROUP BY 1 ORDER BY count(*) DESC LIMIT 100""").fetchall()]
    camp_ix = {c: i for i, c in enumerate(top)}
    rows = con.execute("""SELECT ev.user_id, ev.campaign_id, COALESCE(t.vt,0) AS seg,
        sum(ev.clk) s, count(*) c
      FROM ev LEFT JOIN t8 t USING (user_id)
      WHERE day = ? AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1,2,3""", [d, top]).df()
    uids, uinv = np.unique(rows["user_id"].to_numpy(), return_inverse=True)
    cell = rows["campaign_id"].map(camp_ix).to_numpy() * nS + rows["seg"].to_numpy()
    Mc = csr_matrix((rows["s"].to_numpy(float), (uinv, cell)), shape=(len(uids), nC * nS))
    Mn = csr_matrix((rows["c"].to_numpy(float), (uinv, cell)), shape=(len(uids), nC * nS))
    g = float(rows["s"].sum() / rows["c"].sum())

    def table(S, N):
        camp_s = S.reshape(nC, nS).sum(1); camp_n = N.reshape(nC, nS).sum(1)
        bse = np.where(camp_n >= MINC, (camp_s + ALPHA * g) / (camp_n + ALPHA), g)
        P = np.tile(bse[:, None], (1, nS))
        Ss, Nn = S.reshape(nC, nS), N.reshape(nC, nS)
        m = Nn >= MINC
        P[m] = (Ss[m] + ALPHA * np.tile(bse[:, None], (1, nS))[m]) / (Nn[m] + ALPHA)
        return P

    W = rng.poisson(1.0, size=(B, len(uids))).astype(float)
    Sb = W @ Mc; Nb = W @ Mn
    S0 = np.asarray(Mc.sum(0)).ravel(); N0 = np.asarray(Mn.sum(0)).ravel()
    P0 = table(S0, N0)
    for a, b_, name in comps:
        dn = z[a] - z[b_]
        point[name] += float((dn * P0).sum())
        base[name] += float((z[b_] * P0).sum())
        fold_lifts[name].append(float((dn * P0).sum()) / float((z[b_] * P0).sum()) * 100)
        for i in range(B):
            boot[name][i] += float((dn * table(Sb[i], Nb[i])).sum())
    print(f"d{d}: users={len(uids):,} done")

for _, _, name in comps:
    se = boot[name].std()
    lift = point[name] / base[name] * 100
    se_l = se / base[name] * 100
    fl = np.array(fold_lifts[name])
    tstat = fl.mean() / (fl.std(ddof=1) / np.sqrt(len(fl)))
    print(f"{name}: lift={lift:+.2f}%  user-bootstrap se={se_l:.2f}pp  z={point[name]/se:+.2f}"
          f"  | across-fold t(4)={tstat:+.2f} (fold mean {fl.mean():+.2f}%)")
