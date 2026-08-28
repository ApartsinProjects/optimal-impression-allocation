"""Monte-Carlo comparison: greedy myopic serving vs LP-planned allocation.

Instance model
--------------
- S audience segments (cluster x context-slot), each with expected traffic H_s
  (lognormal, so a few heavy segments and a long tail).
- C campaigns, each with impression quota Q_c (total quota = SELL_THROUGH of
  expected traffic), per-impression price r_c and per-click price q_c.
- Click probabilities p[c,s] from a low-rank affinity model:
  p = base_c * exp(a_c . f_s), clipped to [0.001, 0.08] -- realistic mobile CTRs
  with genuine campaign-segment interaction (some campaigns fit some audiences).
- Targeting eligibility mask: each campaign eligible on a random ~ELIG share of
  segments (always including its best-affinity segments' subset by construction
  of the mask draw; feasibility is checked by the LP).

Policies
--------
1. GREEDY  : events stream in; each event from segment s is served the eligible,
             quota-remaining campaign maximizing immediate expected revenue
             r_c + q_c * p[c,s].  (The "conventional" per-event argmax.)
2. PLANNED : LP solved offline on H_s (the forecast); at serving time an event
             from s is served the campaign with the largest remaining planned
             allocation x[c,s] (quota-remaining, eligible). Plan faces forecast
             noise because actual traffic is a multinomial draw, not H_s itself.
3. LP BOUND: offline optimum on the *realized* traffic -- an upper bound no
             online policy can beat.

Both policies see identical event streams and identical p, r, q, masks.

Invariants checked each replication
-----------------------------------
I1: greedy <= LP bound and planned <= LP bound (tolerance 1e-9 relative).
I2: with p uniform across campaigns (degenerate instance), |lift| < 1%.
I3: planned quota fill >= greedy quota fill - 1e-9 is NOT asserted (not a
    theorem); both fills are reported instead.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

S, C = 60, 25
SELL_THROUGH = 0.80
ELIG = 0.60
REPS = 20


def make_instance(rng, uniform_p=False):
    H = rng.lognormal(mean=8.0, sigma=0.7, size=S)
    H = H / H.sum() * 200_000
    T = int(H.sum())

    shares = rng.dirichlet(np.ones(C) * 2.0)
    Q = np.floor(shares * SELL_THROUGH * T).astype(int)

    r = rng.uniform(0.05, 0.30, size=C)          # $ per impression
    q = rng.uniform(0.50, 3.00, size=C)          # $ per click

    if uniform_p:
        p = np.tile(rng.uniform(0.005, 0.03, size=(C, 1)), (1, S))
        p = np.tile(p.mean(axis=0), (C, 1))      # identical across campaigns
    else:
        K = 5
        a = rng.normal(0, 1.0, size=(C, K))
        f = rng.normal(0, 1.0, size=(K, S)) / np.sqrt(K)
        base = rng.uniform(0.004, 0.02, size=(C, 1))
        p = np.clip(base * np.exp(a @ f), 0.001, 0.08)

    elig = rng.random((C, S)) < ELIG
    for c in range(C):                            # every campaign somewhere
        if not elig[c].any():
            elig[c, rng.integers(S)] = True
    return H, T, Q, r, q, p, elig


def solve_lp(supply, Q, r, q, p, elig):
    """max sum x[c,s]*(r_c+q_c*p[c,s])  s.t. sum_c x<=supply_s, sum_s x<=Q_c."""
    idx = [(c, s) for c in range(C) for s in range(S) if elig[c, s]]
    n = len(idx)
    cost = np.array([-(r[c] + q[c] * p[c, s]) for c, s in idx])
    A = lil_matrix((S + C, n))
    for j, (c, s) in enumerate(idx):
        A[s, j] = 1.0
        A[S + c, j] = 1.0
    b = np.concatenate([supply, Q]).astype(float)
    res = linprog(cost, A_ub=A.tocsr(), b_ub=b, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    x = np.zeros((C, S))
    for j, (c, s) in enumerate(idx):
        x[c, s] = res.x[j]
    return x, -res.fun


def serve(events, Q, r, q, p, elig, plan=None):
    """Stream events; return expected revenue, expected clicks, quota served."""
    served = np.zeros(C)
    rev = clicks = 0.0
    remaining = None if plan is None else plan.copy()
    score = r[:, None] + q[:, None] * p          # (C,S) immediate value
    for s in events:
        ok = elig[:, s] & (served < Q)
        if not ok.any():
            continue
        if remaining is None:                    # greedy
            cands = np.where(ok)[0]
            c = cands[np.argmax(score[cands, s])]
        else:                                    # follow the plan
            cands = np.where(ok & (remaining[:, s] > 0.5))[0]
            if len(cands) == 0:                  # plan exhausted here: fall back
                cands = np.where(ok)[0]
                c = cands[np.argmax(score[cands, s])]
            else:
                c = cands[np.argmax(remaining[cands, s])]
                remaining[c, s] -= 1.0
        served[c] += 1
        rev += score[c, s]
        clicks += p[c, s]
    return rev, clicks, served.sum() / Q.sum()


def replicate(seed, uniform_p=False):
    rng = np.random.default_rng(seed)
    H, T, Q, r, q, p, elig = make_instance(rng, uniform_p)

    # realized traffic: multinomial draw around the forecast
    counts = rng.multinomial(T, H / H.sum())
    events = np.repeat(np.arange(S), counts)
    rng.shuffle(events)

    plan, _ = solve_lp(H, Q, r, q, p, elig)              # plan on the FORECAST
    _, bound = solve_lp(counts.astype(float), Q, r, q, p, elig)  # bound on ACTUAL

    g_rev, g_clk, g_fill = serve(events, Q, r, q, p, elig)
    p_rev, p_clk, p_fill = serve(events, Q, r, q, p, elig, plan=plan)

    assert g_rev <= bound * (1 + 1e-9), "I1 violated: greedy beats LP bound"
    assert p_rev <= bound * (1 + 1e-9), "I1 violated: planned beats LP bound"
    return g_rev, p_rev, bound, g_clk, p_clk, g_fill, p_fill


def main():
    rows = [replicate(seed) for seed in range(REPS)]
    g_rev, p_rev, bound, g_clk, p_clk, g_fill, p_fill = map(np.array, zip(*rows))

    lift = (p_rev - g_rev) / g_rev * 100
    clift = (p_clk - g_clk) / g_clk * 100
    gap = p_rev / bound * 100
    ggap = g_rev / bound * 100

    print(f"replications: {REPS}   segments: {S}  campaigns: {C}  "
          f"events/rep: ~200k  sell-through: {SELL_THROUGH:.0%}")
    print(f"revenue lift planned vs greedy : mean {lift.mean():6.2f}%   "
          f"min {lift.min():6.2f}%  max {lift.max():6.2f}%")
    print(f"click lift   planned vs greedy : mean {clift.mean():6.2f}%   "
          f"min {clift.min():6.2f}%  max {clift.max():6.2f}%")
    print(f"planned as % of LP upper bound : mean {gap.mean():6.2f}%   "
          f"min {gap.min():6.2f}%")
    print(f"greedy  as % of LP upper bound : mean {ggap.mean():6.2f}%   "
          f"min {ggap.min():6.2f}%")
    print(f"quota fill: greedy {g_fill.mean():.4f}   planned {p_fill.mean():.4f}")

    # I2: degenerate uniform-p instances -> lift must vanish
    urows = [replicate(1000 + s, uniform_p=True) for s in range(5)]
    ug, up = np.array([x[0] for x in urows]), np.array([x[1] for x in urows])
    ulift = (up - ug) / ug * 100
    print(f"I2 uniform-p sanity: lift mean {ulift.mean():.3f}% "
          f"(each: {np.round(ulift, 3)})  -- expect ~0")
    assert np.all(np.abs(ulift) < 1.0), "I2 violated"
    print("all invariants passed")


if __name__ == "__main__":
    main()
