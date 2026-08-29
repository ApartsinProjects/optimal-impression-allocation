"""Third-dataset pre-committed validation on Outbrain.

Two modes enforce the pre-registration discipline:
  --mode persist : measure double-centered interaction persistence per candidate
                   axis on the PLANNING window only; print the ranking and the
                   implied configuration. NO replay is run. Commit this output
                   BEFORE running --mode replay.
  --mode replay  : run the walk-forward dual-model replay on the frozen config
                   (--seg-mode), reporting planned/re-solved/greedy lifts.

Candidate axes: activity tier (train-window impression count, 8 bins),
platform (3), geo region (US state prefix). Segments for replay: activity tier
only (mirrors the Taobao pre-committed choice) unless --seg-mode overrides.

Usage:
  python h2_outbrain.py --mode persist --train-days 0,9 --replay-days 10
  python h2_outbrain.py --mode replay  --seg-mode tier8 --train-days 0,d-1 --replay-days d
"""
import argparse, os, json
import numpy as np
import duckdb
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research"
DB = os.path.join(BASE, "data", "outbrain", "prep", "outbrain.duckdb")

TIER8 = ("""CASE WHEN count(*)>=200 THEN 7 WHEN count(*)>=100 THEN 6
  WHEN count(*)>=50 THEN 5 WHEN count(*)>=30 THEN 4 WHEN count(*)>=20 THEN 3
  WHEN count(*)>=10 THEN 2 WHEN count(*)>=5 THEN 1 ELSE 0 END""")


def con_ro():
    c = duckdb.connect(DB, read_only=True)
    c.execute("PRAGMA memory_limit='6GB'")
    c.execute("PRAGMA threads=2")
    return c


def dc_persist(con, TD, RD, axis_sql, join="", topk=100):
    top = [r[0] for r in con.execute(
        f"SELECT campaign_id FROM ev WHERE {TD} GROUP BY 1 ORDER BY count(*) DESC LIMIT {topk}").fetchall()]
    q = lambda dfilt: con.execute(f"""
      SELECT ev.campaign_id, {axis_sql} AS grp, sum(ev.clk) s, count(*) c
      FROM ev {join} WHERE {dfilt} AND ev.campaign_id IN (SELECT UNNEST(?))
      GROUP BY 1,2""", [top]).df()
    # split the planning window in two halves for the persistence measurement
    lo, hi = [int(x) for x in TD.replace("day BETWEEN ", "").split(" AND ")]
    mid = (lo + hi) // 2
    a = q(f"day BETWEEN {lo} AND {mid}")
    b = q(f"day BETWEEN {mid+1} AND {hi}") if hi > mid else q(f"day BETWEEN {lo} AND {hi}")
    m = a.merge(b, on=["campaign_id", "grp"], suffixes=("_a", "_b"))
    m = m[(m.c_a >= 300) & (m.c_b >= 150)].copy()
    if len(m) < 30:
        return None, len(m)
    for w in ("a", "b"):
        camp = m.groupby("campaign_id").apply(lambda x: x[f"s_{w}"].sum()/x[f"c_{w}"].sum(), include_groups=False)
        grp = m.groupby("grp").apply(lambda x: x[f"s_{w}"].sum()/x[f"c_{w}"].sum(), include_groups=False)
        gl = m[f"s_{w}"].sum()/m[f"c_{w}"].sum()
        m[f"r_{w}"] = m[f"s_{w}"]/m[f"c_{w}"] - camp.reindex(m.campaign_id).to_numpy() - grp.reindex(m.grp).to_numpy() + gl
    return float(np.corrcoef(m.r_a, m.r_b)[0, 1]), len(m)


def mode_persist(args):
    con = con_ro()
    TD = f"day BETWEEN {args.train_days.replace(',', ' AND ')}"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tier AS
      SELECT user_id, {TIER8} AS vt FROM ev WHERE {TD} GROUP BY 1""")
    axes = {
        "activity_tier8": ("COALESCE(t.vt,0)", "LEFT JOIN tier t USING (user_id)"),
        "platform": ("ev.platform", ""),
        "geo_region": ("split_part(ev.geo_location,'>',1)", ""),
    }
    out = {}
    for name, (sql, join) in axes.items():
        r, n = dc_persist(con, TD, args.replay_days, sql, join)
        out[name] = {"persistence": None if r is None else round(r, 3), "cells": n}
        print(f"{name}: persistence r={out[name]['persistence']} (cells={n})")
    ranked = sorted([k for k in out if out[k]["persistence"] is not None],
                    key=lambda k: -out[k]["persistence"])
    implied = ranked[0] if ranked else None
    out["_implied_config"] = {"segment_axis": implied,
                              "granularity": "8 bins (activity) / native (categorical)",
                              "rule": "segment on the most persistent axis"}
    p = os.path.join(BASE, "results", "outbrain_persist_FROZEN.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print("\nIMPLIED CONFIG:", json.dumps(out["_implied_config"]))
    print("Frozen to", p, "- COMMIT before running --mode replay")


def solve_lp(nC, nS, supply, quota, P, want_duals=False):
    idx = [(c, s) for c in range(nC) for s in range(nS)]
    cost = np.array([-P[c, s] for c, s in idx])
    A = lil_matrix((nS + nC, len(idx)))
    for j, (c, s) in enumerate(idx):
        A[s, j] = 1.0; A[nS + c, j] = 1.0
    r = linprog(cost, A_ub=A.tocsr(), b_ub=np.concatenate([supply, quota]).astype(float),
                bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    x = np.zeros((nC, nS))
    for j, (c, s) in enumerate(idx):
        x[c, s] = r.x[j]
    if want_duals:
        return x, np.maximum(-np.asarray(r.ineqlin.marginals)[nS:], 0.0)
    return x


def mode_replay(args):
    con = con_ro()
    tlo, thi = args.train_days.split(",")
    TD = f"day BETWEEN {tlo} AND {thi}"
    n_td = int(thi) - int(tlo) + 1
    rd = args.replay_days
    n_rd = len(rd.split(","))
    if args.seg_mode == "tier8":
        con.execute(f"""CREATE OR REPLACE TEMP TABLE tier AS
          SELECT user_id, {TIER8} AS vt FROM ev WHERE {TD} GROUP BY 1""")
        nS = 8
        con.execute("""CREATE OR REPLACE TEMP VIEW evs AS
          SELECT ev.*, COALESCE(t.vt,0) AS seg FROM ev LEFT JOIN tier t USING (user_id)""")
    elif args.seg_mode == "platform":
        # platform values 1/2/3 (+ null) -> seg 0..3
        nS = 4
        con.execute("""CREATE OR REPLACE TEMP VIEW evs AS
          SELECT ev.*, CAST(COALESCE(TRY_CAST(ev.platform AS INT), 0) AS INT) AS seg FROM ev""")
    elif args.seg_mode == "geo":
        gmap = {v: i for i, (v,) in enumerate(con.execute(
            "SELECT DISTINCT split_part(geo_location,'>',1) FROM ev ORDER BY 1").fetchall())}
        nS = len(gmap)
        con.execute("CREATE TEMP TABLE gmap (v VARCHAR, i INT)")
        con.executemany("INSERT INTO gmap VALUES (?,?)", list(gmap.items()))
        con.execute("""CREATE OR REPLACE TEMP VIEW evs AS
          SELECT ev.*, COALESCE(gmap.i, 0) AS seg
          FROM ev LEFT JOIN gmap ON split_part(ev.geo_location,'>',1)=gmap.v""")
    else:
        raise SystemExit(f"unknown seg-mode {args.seg_mode}")
    top = [r[0] for r in con.execute(
        f"SELECT campaign_id FROM evs WHERE {TD} GROUP BY 1 ORDER BY count(*) DESC LIMIT {args.topk}").fetchall()]
    cix = {c: i for i, c in enumerate(top)}
    nC = len(cix)
    con.execute("CREATE TEMP TABLE topc AS SELECT UNNEST(?) AS campaign_id", [top])
    g = con.execute(f"SELECT avg(clk) FROM evs WHERE {TD}").fetchone()[0]

    def build_P(dfilt, a=20.0, mc=30):
        camp = con.execute(f"SELECT campaign_id, sum(clk) s, count(*) c FROM evs JOIN topc USING (campaign_id) WHERE {dfilt} GROUP BY 1").df()
        cs = con.execute(f"SELECT campaign_id, seg, sum(clk) s, count(*) c FROM evs JOIN topc USING (campaign_id) WHERE {dfilt} GROUP BY 1,2 HAVING count(*)>={mc}").df()
        M = np.zeros((nC, nS)); base = np.full(nC, g)
        for cid, s_, c_ in camp.itertuples(index=False):
            if c_ >= mc: base[cix[cid]] = (s_ + a*g)/(c_ + a)
        M[:] = base[:, None]
        for cid, sg, s_, c_ in cs.itertuples(index=False):
            M[cix[cid], int(sg)] = (s_ + a*base[cix[cid]])/(c_ + a)
        N = np.zeros((nC, nS))
        allc = con.execute(f"SELECT campaign_id, seg, count(*) c FROM evs JOIN topc USING (campaign_id) WHERE {dfilt} GROUP BY 1,2").df()
        for cid, sg, c_ in allc.itertuples(index=False):
            N[cix[cid], int(sg)] = c_
        return M, N

    P, _ = build_P(TD)
    P_eval, N_eval = build_P(f"day IN ({rd})")
    replay = con.execute(f"SELECT seg, campaign_id FROM evs JOIN topc USING (campaign_id) WHERE day IN ({rd}) ORDER BY ts").df()
    fc = con.execute(f"SELECT seg, count(*) c FROM evs JOIN topc USING (campaign_id) WHERE {TD} GROUP BY 1").df()
    forecast = np.zeros(nS); forecast[fc.seg.to_numpy()] = fc.c.to_numpy()/n_td*n_rd
    quota = np.zeros(nC)
    for cid, c in replay.groupby("campaign_id").size().items():
        quota[cix[cid]] = c
    quota = np.ceil(quota * args.quota_scale)
    segs = replay.seg.to_numpy()
    plan, plan_duals = solve_lp(nC, nS, forecast, quota, P, want_duals=True)
    _, = (None,)
    bound = float((solve_lp(nC, nS, np.bincount(segs, minlength=nS).astype(float), quota, P_eval) * P_eval).sum())
    order = np.argsort(-P, axis=0)

    def serve(use_plan, every=0):
        served = np.zeros(nC); rem = plan.copy() if use_plan else None; val = 0.0
        assign = np.zeros((nC, nS)); T = float(len(segs))
        for t, s in enumerate(segs):
            if every and t % every == 0:
                rq = np.maximum(quota - served, 0); rs = forecast*max(1-t/T, 1e-9)
                rem = solve_lp(nC, nS, rs, rq, P)
            c = None
            if rem is not None:
                cand = np.where(rem[:, s] > 0.5)[0]; cand = cand[served[cand] < quota[cand]]
                if len(cand): c = cand[np.argmax(rem[cand, s])]; rem[c, s] -= 1
            if c is None:
                for cc in order[:, s]:
                    if served[cc] < quota[cc]: c = cc; break
            served[c] += 1; assign[c, s] += 1; val += P_eval[c, s]
        return val, assign

    def serve_fixed_dual(mu):
        served = np.zeros(nC); assign = np.zeros((nC, nS)); sc = P - mu[:, None]; val = 0.0
        for i, s in enumerate(segs):
            ok = np.where(served < quota)[0]
            if not len(ok): continue
            v = sc[ok, s]; tied = ok[v >= v.max()-1e-12]
            c = tied[(i*2654435761) % len(tied)] if len(tied) > 1 else tied[0]
            served[c] += 1; assign[c, s] += 1; val += P_eval[c, s]
        return val, assign

    def serve_dual(eta):
        served = np.zeros(nC); assign = np.zeros((nC, nS)); mu = np.zeros(nC)
        D = np.maximum(quota * min(1.0, forecast.sum()/max(quota.sum(),1)), 1.0)
        T = float(len(segs)); val = 0.0
        for t, s in enumerate(segs, 1):
            ok = np.where(served < quota)[0]
            if not len(ok): continue
            c = ok[np.argmax(P[ok, s] - mu[ok])]
            served[c] += 1; assign[c, s] += 1; val += P_eval[c, s]
            mu[c] += eta*g*(served[c]/D[c] - t/T)
        return val, assign

    gv, ga = serve(False)
    pv, pa = serve(True)
    rv, ra = serve(True, every=20000)
    fdv, fda = serve_fixed_dual(plan_duals)
    adv, ada = serve_dual(0.1)
    for v in (gv, pv, rv, fdv, adv):
        assert v <= bound*(1+1e-9), "INVARIANT FAIL"
    res = {"seg_mode": args.seg_mode, "quota_scale": args.quota_scale,
           "n_replay": int(len(replay)), "nC": nC, "nS": nS,
           "lift_planned_vs_greedy_pct": (pv-gv)/gv*100,
           "lift_replan_vs_greedy_pct": (rv-gv)/gv*100,
           "lift_fixeddual_vs_greedy_pct": (fdv-gv)/gv*100,
           "lift_adaptivedual_vs_greedy_pct": (adv-gv)/gv*100,
           "planned_pct_bound": pv/bound*100, "greedy_pct_bound": gv/bound*100}
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    np.savez_compressed(os.path.join(BASE, "results", f"outbrain_{args.tag}_assign.npz"),
                        P_eval=P_eval, N_eval=N_eval, g_assign=ga, p_assign=pa, r_assign=ra)
    json.dump(res, open(os.path.join(BASE, "results", f"outbrain_{args.tag}.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["persist", "replay"], required=True)
    ap.add_argument("--tag", default="fold")
    ap.add_argument("--seg-mode", default="tier8")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--train-days", default="0,9")
    ap.add_argument("--replay-days", default="10")
    ap.add_argument("--quota-scale", type=float, default=1.0)
    a = ap.parse_args()
    (mode_persist if a.mode == "persist" else mode_replay)(a)
