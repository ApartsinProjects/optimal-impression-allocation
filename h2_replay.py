"""H2: allocation replay on the top-K-campaign sub-market (Taobao).

Market: replay-window impressions (China days 12-13 of May 2017) that were
historically served by one of the top-K campaigns (ranked by train-window
volume, days 6-11; no lookahead in the ranking). Quota per campaign = its
realized impression count in the replay window, so the counterfactual is
"the same inventory, re-paired". Every campaign is eligible on every event
(Taobao display slots are not context-gated), subject to quota.

Value model: expected clicks at the (campaign, segment) level, where a
segment = (user cluster, hour-bucket x pid slot). p_cs comes from train-window
empirical rates with Laplace smoothing and backoff (cluster,campaign,slot) ->
(cluster,campaign) -> (campaign) -> global. Scoring policies at the same
(c,s) granularity keeps the hindsight LP a TRUE upper bound on any feasible
assignment of the realized stream.

Policies:
  greedy   : time order; serve the quota-remaining campaign with max p_cs.
  planned  : LP on FORECAST supply (train-window per-day mean x 2), then
             serve by drawing down remaining plan x[c,s]; fall back to greedy
             where the local plan is exhausted.
  hindsight: LP on the REALIZED segment counts (upper bound, not a policy).

Invariants (assert): greedy <= hindsight; planned <= hindsight; with
--uniform-p the greedy-vs-planned lift is ~0; quota fill reported.

Output: research/results/h2_<tag>.json (refuses to overwrite).
Usage: python h2_replay.py --tag smoke --user-frac 0.05 --topk 100 --k 50
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import duckdb
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from sklearn.cluster import MiniBatchKMeans

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research"
DB = os.path.join(BASE, "data", "taobao", "prep", "taobao.duckdb")
PROFILE_COLS = ["cms_segid", "cms_group_id", "final_gender_code", "age_level",
                "pvalue_level", "shopping_level", "occupation", "new_user_class_level"]


def solve_lp(nC, nS, supply, quota, P, elig_mask=None):
    """max sum x_cs P_cs  s.t.  sum_c x_cs <= supply_s, sum_s x_cs <= quota_c."""
    idx = [(c, s) for c in range(nC) for s in range(nS)
           if (elig_mask is None or elig_mask[c, s])]
    cost = np.array([-P[c, s] for c, s in idx])
    A = lil_matrix((nS + nC, len(idx)))
    for j, (c, s) in enumerate(idx):
        A[s, j] = 1.0
        A[nS + c, j] = 1.0
    b = np.concatenate([supply, quota]).astype(float)
    r = linprog(cost, A_ub=A.tocsr(), b_ub=b, bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    x = np.zeros((nC, nS))
    for j, (c, s) in enumerate(idx):
        x[c, s] = r.x[j]
    return x, -r.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--user-frac", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uniform-p", action="store_true")
    ap.add_argument("--seg-mode",
                    choices=["cluster_slot", "tier_demo", "tier_only", "demo_only", "tier8_only"],
                    default="cluster_slot")
    ap.add_argument("--replay-days", default="12,13", help="comma list of replay days")
    ap.add_argument("--train-days", default="6,11", help="lo,hi inclusive train-day range")
    ap.add_argument("--user-half", type=int, default=-1, choices=[-1, 0, 1],
                    help="restrict to a user-hash half (for split-half CIs)")
    ap.add_argument("--plan-alpha", type=float, default=None,
                    help="EB shrinkage for the PLANNING table only (default: --alpha)")
    ap.add_argument("--dual-eta", type=float, default=0.5,
                    help="dual-price baseline learning rate, in units of global CTR")
    args = ap.parse_args()

    tlo, thi = args.train_days.split(",")
    TD = f"day BETWEEN {tlo} AND {thi}"
    n_tdays = int(thi) - int(tlo) + 1
    n_rdays = len(args.replay_days.split(","))
    plan_alpha = args.plan_alpha if args.plan_alpha is not None else args.alpha

    out_path = os.path.join(BASE, "results", f"h2_{args.tag}.json")
    if os.path.exists(out_path):
        sys.exit(f"refusing to overwrite {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()

    con = duckdb.connect(DB, read_only=True)
    con.execute(f"""CREATE OR REPLACE TEMP VIEW ev AS
      SELECT raw.user AS user_id, raw.clk, raw.pid, raw.time_stamp,
             date_part('day',  to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
             CAST(date_part('hour', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') // 4 AS INT) AS hb,
             ad.campaign_id
      FROM raw JOIN ad USING (adgroup_id)
      WHERE hash(raw.user) % 1000 < {int(args.user_frac * 1000)}
        {'' if args.user_half < 0 else f'AND hash(raw.user + 7) % 2 = {args.user_half}'}""")

    top = con.execute(f"""SELECT campaign_id FROM ev WHERE {TD}
      GROUP BY 1 ORDER BY count(*) DESC LIMIT {args.topk}""").df()["campaign_id"]
    camp_ix = {c: i for i, c in enumerate(top)}
    nC = len(camp_ix)
    con.execute("CREATE TEMP TABLE topc AS SELECT UNNEST(?) AS campaign_id", [list(map(int, top))])

    if args.seg_mode == "cluster_slot":
        # cluster users on profile (sparse one-hot; profile-less -> extra cluster)
        prof = con.execute(f"""SELECT userid, {', '.join('COALESCE(' + c + ", -1) AS " + c for c in PROFILE_COLS)}
          FROM up WHERE hash(userid) % 1000 < {int(args.user_frac * 1000)}""").df()
        from sklearn.preprocessing import OneHotEncoder
        X = OneHotEncoder(handle_unknown="ignore", dtype=np.float32).fit_transform(prof[PROFILE_COLS])
        km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
        lab = km.fit_predict(X)
        NOPROF = args.k
        nS = (args.k + 1) * 12
        con.execute("CREATE TEMP TABLE ucl (userid BIGINT, cl INT)")
        con.register("ucl_df", pd.DataFrame({"userid": prof["userid"].to_numpy(), "cl": lab}))
        con.execute("INSERT INTO ucl SELECT * FROM ucl_df")
        con.execute(f"""CREATE OR REPLACE TEMP VIEW evs AS
          SELECT ev.*, COALESCE(ucl.cl, {NOPROF}) * 12
                 + ev.hb * 2 + CASE WHEN ev.pid = '430548_1007' THEN 1 ELSE 0 END AS seg
          FROM ev LEFT JOIN ucl ON ev.user_id = ucl.userid""")
    else:
        # tier-based modes: activity tier from the train window only
        tier_case = ("""CASE WHEN count(*)>=200 THEN 7 WHEN count(*)>=100 THEN 6
                        WHEN count(*)>=50 THEN 5 WHEN count(*)>=30 THEN 4
                        WHEN count(*)>=20 THEN 3 WHEN count(*)>=10 THEN 2
                        WHEN count(*)>=5 THEN 1 ELSE 0 END"""
                     if args.seg_mode == "tier8_only" else
                     """CASE WHEN count(*)>=100 THEN 3 WHEN count(*)>=30 THEN 2
                        WHEN count(*)>=10 THEN 1 ELSE 0 END""")
        con.execute(f"""CREATE TEMP TABLE tier AS
          SELECT user_id, {tier_case} AS vt
          FROM ev WHERE {TD} GROUP BY 1""")
        seg_expr = {
            "tier8_only": "COALESCE(t.vt, 0)",
            "tier_demo": """COALESCE(t.vt, 0) * 40
                 + (COALESCE(up.final_gender_code, -1) + 2) * 10
                 + (COALESCE(up.age_level, -1) + 1)""",
            "tier_only": "COALESCE(t.vt, 0)",
            "demo_only": """(COALESCE(up.final_gender_code, -1) + 2) * 10
                 + (COALESCE(up.age_level, -1) + 1)""",
        }[args.seg_mode]
        nS = {"tier_demo": 168, "tier_only": 4, "demo_only": 48, "tier8_only": 8}[args.seg_mode]
        con.execute(f"""CREATE OR REPLACE TEMP VIEW evs AS
          SELECT ev.*, {seg_expr} AS seg
          FROM ev LEFT JOIN tier t USING (user_id)
                  LEFT JOIN up ON ev.user_id = up.userid""")

    g_ctr = con.execute(f"SELECT avg(clk) FROM evs WHERE {TD}").fetchone()[0]

    def build_P(day_filter, alpha, min_count, g, with_counts=False):
        camp = con.execute(f"""SELECT campaign_id, sum(clk) s, count(*) c FROM evs
          JOIN topc USING (campaign_id) WHERE {day_filter} GROUP BY 1""").df()
        mid = {"cluster_slot": 12, "tier_demo": 40,
               "tier_only": nS, "demo_only": nS, "tier8_only": nS}[args.seg_mode]
        cs_t = con.execute(f"""SELECT campaign_id, seg, sum(clk) s, count(*) c FROM evs
          JOIN topc USING (campaign_id) WHERE {day_filter} GROUP BY 1,2 HAVING count(*) >= {min_count}""").df()
        ccl_t = con.execute(f"""SELECT campaign_id, seg // {mid} AS cl, sum(clk) s, count(*) c FROM evs
          JOIN topc USING (campaign_id) WHERE {day_filter} GROUP BY 1,2 HAVING count(*) >= {min_count}""").df()
        M = np.zeros((nC, nS))
        base = np.full(nC, g)
        for cid, s_, c_ in camp.itertuples(index=False):
            if c_ >= min_count:
                base[camp_ix[cid]] = (s_ + alpha * g) / (c_ + alpha)
        M[:] = base[:, None]
        for cid, cl, s_, c_ in ccl_t.itertuples(index=False):
            ci = camp_ix[cid]
            M[ci, int(cl) * mid:(int(cl) + 1) * mid] = (s_ + alpha * base[ci]) / (c_ + alpha)
        for cid, sg, s_, c_ in cs_t.itertuples(index=False):
            ci = camp_ix[cid]
            M[ci, int(sg)] = (s_ + alpha * base[ci]) / (c_ + alpha)
        if with_counts:
            N = np.zeros((nC, nS))
            allc = con.execute(f"""SELECT campaign_id, seg, count(*) c FROM evs
              JOIN topc USING (campaign_id) WHERE {day_filter} GROUP BY 1,2""").df()
            for cid, sg, c_ in allc.itertuples(index=False):
                N[camp_ix[cid], int(sg)] = c_
            return M, N
        return M

    P = build_P(TD, plan_alpha, args.min_count, g_ctr)
    P_eval, N_eval = build_P(f"day IN ({args.replay_days})", args.alpha, args.min_count, g_ctr, with_counts=True)
    if args.uniform_p:
        P = np.full_like(P, g_ctr)
        P_eval = np.full_like(P_eval, g_ctr)

    # replay stream (top-K only), supply forecast, realized counts, quotas
    rd = args.replay_days
    replay = con.execute(f"""SELECT evs.seg, evs.campaign_id FROM evs JOIN topc USING (campaign_id)
      WHERE day IN ({rd}) ORDER BY time_stamp""").df()
    fc = con.execute(f"""SELECT seg, count(*) c FROM evs JOIN topc USING (campaign_id)
      WHERE {TD} GROUP BY 1""").df()
    forecast = np.zeros(nS)
    forecast[fc["seg"].to_numpy()] = fc["c"].to_numpy() / n_tdays * n_rdays
    realized = np.bincount(replay["seg"].to_numpy(), minlength=nS).astype(float)
    quota = np.zeros(nC)
    for c_id, c in replay.groupby("campaign_id").size().items():
        quota[camp_ix[c_id]] = c

    plan, plan_obj = solve_lp(nC, nS, forecast, quota, P)      # planner sees train only
    _, bound = solve_lp(nC, nS, realized, quota, P_eval)       # bound under EVAL model
    _, bound_plan_model = solve_lp(nC, nS, realized, quota, P)  # diagnostic only

    # ---- replay serving (decisions under P; scoring under P_eval) ----
    segs = replay["seg"].to_numpy()
    order_p = np.argsort(-P, axis=0)  # per segment, campaigns by planning value desc

    def serve(use_plan):
        served = np.zeros(nC)
        assign = np.zeros((nC, nS))
        remaining = plan.copy() if use_plan else None
        val = val_plan_model = 0.0
        for s in segs:
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
            assign[c, s] += 1
            val += P_eval[c, s]
            val_plan_model += P[c, s]
        return val, val_plan_model, served, assign

    def serve_dual(eta_ctr):
        """Adaptive plan-free baseline: greedy on P - mu_c with pacing duals.
        mu_c rises when campaign c runs ahead of its quota pace, falls behind."""
        eta = eta_ctr * g_ctr
        served = np.zeros(nC)
        assign = np.zeros((nC, nS))
        mu = np.zeros(nC)
        T = float(len(segs))
        val = 0.0
        for t, s in enumerate(segs, 1):
            ok = np.where(served < quota)[0]
            c = ok[np.argmax(P[ok, s] - mu[ok])]
            served[c] += 1
            assign[c, s] += 1
            val += P_eval[c, s]
            mu[c] += eta * (served[c] / quota[c] - t / T)
        return val, served, assign

    g_val, g_val_pm, g_served, g_assign = serve(False)
    p_val, p_val_pm, p_served, p_assign = serve(True)
    d_val, d_served, d_assign = serve_dual(args.dual_eta)
    np.savez_compressed(os.path.join(BASE, "results", f"h2_{args.tag}_assign.npz"),
                        P=P, P_eval=P_eval, N_eval=N_eval, g_assign=g_assign,
                        p_assign=p_assign, d_assign=d_assign, quota=quota,
                        forecast=forecast, realized=realized)

    assert g_val <= bound * (1 + 1e-9), "INVARIANT FAIL: greedy > hindsight"
    assert p_val <= bound * (1 + 1e-9), "INVARIANT FAIL: planned > hindsight"
    assert d_val <= bound * (1 + 1e-9), "INVARIANT FAIL: dual > hindsight"

    res = {"config": vars(args), "n_replay_events": int(len(replay)),
           "n_campaigns": nC, "n_segments": nS,
           "quota_total": float(quota.sum()),
           "greedy_exp_clicks": g_val, "planned_exp_clicks": p_val,
           "dual_exp_clicks": d_val,
           "hindsight_bound": bound,
           "lift_planned_vs_greedy_pct": (p_val - g_val) / g_val * 100 if g_val else None,
           "lift_planned_vs_dual_pct": (p_val - d_val) / d_val * 100 if d_val else None,
           "greedy_pct_of_bound": g_val / bound * 100,
           "planned_pct_of_bound": p_val / bound * 100,
           "dual_pct_of_bound": d_val / bound * 100,
           "quota_fill_dual": float(d_served.sum() / quota.sum()),
           "quota_fill_greedy": float(g_served.sum() / quota.sum()),
           "quota_fill_planned": float(p_served.sum() / quota.sum()),
           "diag_plan_model": {"greedy": g_val_pm, "planned": p_val_pm,
                               "bound": bound_plan_model,
                               "plan_obj_on_forecast": plan_obj},
           "runtime_s": round(time.time() - t0, 1)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
