"""Change-3: user-clustered Poisson bootstrap on the load-bearing regime-map
cells (feasibility-aware fx_* runs). Confirms the conclusions that rest on the
parametric z-values hold under user clustering.
"""
import os
import numpy as np
import duckdb
from scipy.sparse import csr_matrix

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research"
DB = os.path.join(BASE, "data", "taobao", "prep", "taobao.duckdb")
ALPHA, MINC, B = 20.0, 30, 150
DAYS = [9, 10, 11, 12, 13]
rng = np.random.default_rng(0)

con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA memory_limit='6GB'")
con.execute("PRAGMA threads=2")
con.execute("""CREATE OR REPLACE TEMP VIEW ev AS
  SELECT raw.user AS user_id, raw.clk,
         date_part('day', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
         ad.campaign_id FROM raw JOIN ad USING (adgroup_id)""")

# cells to test: (prefix, policyA, policyB, label)
CELLS = [("fx_q10", "r_assign", "g_assign", "replan_vs_greedy @1.0"),
         ("fx_q125", "r_assign", "g_assign", "replan_vs_greedy @1.25"),
         ("fx_q15", "r_assign", "g_assign", "replan_vs_greedy @1.5"),
         ("fx_q15", "d_assign", "g_assign", "dual_plain_vs_greedy @1.5"),
         ("fx_q125", "dw_assign", "g_assign", "dual_warm_vs_greedy @1.25")]

# precompute per-fold resampled tables once, reuse across cells sharing a prefix
tables = {}  # (prefix, day) -> (P0, [Pb...]) built lazily
def get_tables(prefix, d):
    key = d  # resampled eval tables depend only on the serving day, not quota scale
    if key in tables:
        return tables[key]
    nC, nS = 100, 8
    con.execute(f"""CREATE OR REPLACE TEMP TABLE t8 AS
      SELECT user_id, CASE WHEN count(*)>=200 THEN 7 WHEN count(*)>=100 THEN 6
        WHEN count(*)>=50 THEN 5 WHEN count(*)>=30 THEN 4 WHEN count(*)>=20 THEN 3
        WHEN count(*)>=10 THEN 2 WHEN count(*)>=5 THEN 1 ELSE 0 END AS vt
      FROM ev WHERE day BETWEEN 6 AND {d-1} GROUP BY 1""")
    top = [r[0] for r in con.execute(f"""SELECT campaign_id FROM ev
      WHERE day BETWEEN 6 AND {d-1} GROUP BY 1 ORDER BY count(*) DESC LIMIT 100""").fetchall()]
    camp_ix = {c: i for i, c in enumerate(top)}
    rows = con.execute("""SELECT ev.user_id, ev.campaign_id, COALESCE(t.vt,0) AS seg,
        sum(ev.clk) s, count(*) c FROM ev LEFT JOIN t8 t USING (user_id)
        WHERE day = ? AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1,2,3""", [d, top]).df()
    uids, uinv = np.unique(rows["user_id"].to_numpy(), return_inverse=True)
    cell = rows["campaign_id"].map(camp_ix).to_numpy() * nS + rows["seg"].to_numpy()
    Mc = csr_matrix((rows["s"].to_numpy(float), (uinv, cell)), shape=(len(uids), nC * nS))
    Mn = csr_matrix((rows["c"].to_numpy(float), (uinv, cell)), shape=(len(uids), nC * nS))
    g = float(rows["s"].sum() / rows["c"].sum())

    def table(S, N):
        Ss, Nn = S.reshape(nC, nS), N.reshape(nC, nS)
        camp_s, camp_n = Ss.sum(1), Nn.sum(1)
        base = np.where(camp_n >= MINC, (camp_s + ALPHA * g) / (camp_n + ALPHA), g)
        P = np.tile(base[:, None], (1, nS))
        m = Nn >= MINC
        P[m] = (Ss[m] + ALPHA * np.tile(base[:, None], (1, nS))[m]) / (Nn[m] + ALPHA)
        return P

    W = rng.poisson(1.0, size=(B, len(uids))).astype(float)
    Sb, Nb = W @ Mc, W @ Mn
    P0 = table(np.asarray(Mc.sum(0)).ravel(), np.asarray(Mn.sum(0)).ravel())
    Pb = np.stack([table(Sb[i], Nb[i]) for i in range(B)])
    tables[key] = (P0, Pb)
    return tables[key]

for prefix, a, b, label in CELLS:
    dv0 = 0.0; base = 0.0; bootv = np.zeros(B)
    for d in DAYS:
        z = np.load(os.path.join(BASE, "results", f"h2_{prefix}_d{d}_assign.npz"))
        dn = z[a] - z[b]
        P0, Pb = get_tables(prefix, d)
        dv0 += float((dn * P0).sum()); base += float((z[b] * P0).sum())
        for i in range(B):
            bootv[i] += float((dn * Pb[i]).sum())
    se = bootv.std()
    print(f"{label}: lift={dv0/base*100:+.2f}%  clustered-boot z={dv0/se:+.2f}")
