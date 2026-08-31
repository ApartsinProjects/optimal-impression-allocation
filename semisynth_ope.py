"""Semi-synthetic potential-outcome benchmark (review W4 / section 9.4).

Purpose: in a world where the counterfactual truth is KNOWN, check whether the
paper's model-scored replay recovers the correct policy ranking and value, even
though the logging policy is selective. If it does, the evaluation protocol is
validated where real logs cannot verify it.

Design:
  1. Freeze a ground-truth response theta_cs on a DISJOINT slice (Taobao days
     6-8): shrunk empirical (campaign, tier8) click rates, interaction amplified
     by kappa so policies genuinely differ in truth. This is the known world.
  2. The logging policy = the real (campaign, segment) exposures. Sample clicks
     ~ Bernoulli(theta) on days 9-11 to fit the PLANNING table P (with realistic
     estimation noise), and on days 12-13 to fit the EVAL table P_eval. theta is
     never used to fit tables, only to define truth.
  3. Replay the day 12-13 segment stream under each policy; score each assignment
     BOTH under P_eval (what the paper reports) and under theta (the truth).
  4. Report per policy: true lift vs greedy, model-scored lift, bias; and whether
     the model-scored ranking matches the true ranking (Spearman) and picks the
     true-best policy.

No real clicks are used for scoring; only sampled outcomes for table estimation.
"""
import numpy as np
import duckdb
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.stats import spearmanr

DB = r"E:\Projects\Submitted\MobileAd_Comverse\research\data\taobao\prep\taobao.duckdb"
import os
KAPPA = float(os.environ.get("KAPPA", "1.5"))
ALPHA, MINC, TOPK, NS = 20.0, 30, 100, 8
rng = np.random.default_rng(0)
TIER8 = ("CASE WHEN c>=200 THEN 7 WHEN c>=100 THEN 6 WHEN c>=50 THEN 5 "
         "WHEN c>=30 THEN 4 WHEN c>=20 THEN 3 WHEN c>=10 THEN 2 WHEN c>=5 THEN 1 ELSE 0 END")

con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA memory_limit='4GB'"); con.execute("PRAGMA threads=2")
con.execute("""CREATE OR REPLACE TEMP VIEW ev AS
  SELECT raw.user AS user_id, raw.clk, raw.time_stamp,
         date_part('day', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
         ad.campaign_id FROM raw JOIN ad USING (adgroup_id)""")
con.execute(f"CREATE TEMP TABLE tier AS SELECT user_id, {TIER8.replace('c','count(*)')} AS vt FROM ev WHERE day BETWEEN 6 AND 11 GROUP BY 1")
con.execute("""CREATE OR REPLACE TEMP VIEW evs AS
  SELECT ev.*, COALESCE(t.vt,0) AS seg FROM ev LEFT JOIN tier t USING (user_id)""")
top=[int(r[0]) for r in con.execute(f"SELECT campaign_id FROM evs WHERE day BETWEEN 6 AND 11 GROUP BY 1 ORDER BY count(*) DESC LIMIT {TOPK}").fetchall()]
cix={c:i for i,c in enumerate(top)}; nC=len(top)
con.execute("CREATE TEMP TABLE topc AS SELECT UNNEST(?) AS campaign_id",[top])

def agg(dfilt):
    """(campaign, seg) -> (exposures n, clicks s) on the real log."""
    d=con.execute(f"SELECT campaign_id, seg, count(*) n, sum(clk) s FROM evs JOIN topc USING (campaign_id) WHERE {dfilt} GROUP BY 1,2").df()
    N=np.zeros((nC,NS)); S=np.zeros((nC,NS))
    for cid,sg,n,s in d.itertuples(index=False):
        N[cix[cid],int(sg)]=n; S[cix[cid],int(sg)]=s
    return N,S

# 1. ground truth theta on days 6-8 (disjoint), interaction amplified
N0,S0=agg("day BETWEEN 6 AND 8")
g=S0.sum()/N0.sum()
camp=np.where(N0.sum(1)>=MINC,(S0.sum(1)+ALPHA*g)/(N0.sum(1)+ALPHA),g)
rate=np.where(N0>=MINC,(S0+ALPHA*camp[:,None])/(N0+ALPHA),camp[:,None])
seg_mean=(rate*N0).sum(0)/np.maximum(N0.sum(0),1)
inter=rate-camp[:,None]-seg_mean[None,:]+g
theta=np.clip(camp[:,None]+seg_mean[None,:]-g+KAPPA*inter,0.001,0.5)

def sample_table(dfilt):
    """Fit a click table from Bernoulli(theta) outcomes on the real exposures.
    ADV=1: the logging policy selects on OUTCOME -- exposures reweighted by theta,
    so high-response cells are over-observed and low-response cells starve. This is
    the worst case for a scorer fit on logged data (maximally selected logging)."""
    N,_=agg(dfilt)
    if os.environ.get("ADV"):
        N=np.round(N*(theta/theta.mean())).astype(float)   # outcome-correlated logging
    Ssim=rng.binomial(N.astype(int),theta)
    base=np.where(N.sum(1)>=MINC,(Ssim.sum(1)+ALPHA*g)/(N.sum(1)+ALPHA),g)
    M=np.tile(base[:,None],(1,NS))
    m=N>=MINC
    M[m]=(Ssim[m]+ALPHA*np.tile(base[:,None],(1,NS))[m])/(N[m]+ALPHA)
    return M,N

P,Ntr=sample_table("day BETWEEN 9 AND 11")
P_eval,Nev=sample_table("day IN (12,13)")

# forecast + quota + stream
stream=con.execute("SELECT seg FROM evs JOIN topc USING (campaign_id) WHERE day IN (12,13) ORDER BY time_stamp").df()["seg"].to_numpy()
forecast=np.bincount(stream,minlength=NS).astype(float)  # perfect-ish supply (fair to all)
quota=np.zeros(nC)
for cid,c in con.execute("SELECT campaign_id,count(*) c FROM evs JOIN topc USING (campaign_id) WHERE day IN (12,13) GROUP BY 1").df().itertuples(index=False):
    quota[cix[cid]]=c

def solve_lp(supply,q,M):
    idx=[(c,s) for c in range(nC) for s in range(NS)]
    A=lil_matrix((NS+nC,len(idx)))
    for j,(c,s) in enumerate(idx): A[s,j]=1; A[NS+c,j]=1
    r=linprog([-M[c,s] for c,s in idx],A_ub=A.tocsr(),b_ub=np.concatenate([supply,q]).astype(float),bounds=(0,None),method='highs')
    x=np.zeros((nC,NS))
    for j,(c,s) in enumerate(idx): x[c,s]=r.x[j]
    return x
def solve_duals(supply,q,M):
    idx=[(c,s) for c in range(nC) for s in range(NS)]
    A=lil_matrix((NS+nC,len(idx)))
    for j,(c,s) in enumerate(idx): A[s,j]=1; A[NS+c,j]=1
    r=linprog([-M[c,s] for c,s in idx],A_ub=A.tocsr(),b_ub=np.concatenate([supply,q]).astype(float),bounds=(0,None),method='highs')
    return np.maximum(-np.asarray(r.ineqlin.marginals)[NS:],0.0)
plan=solve_lp(forecast,quota,P)
mu_star=solve_duals(forecast,quota,P)
order=np.argsort(-P,axis=0)

def serve(mode,every=0,mu=None,eta=0.0,geom=None):
    served=np.zeros(nC); rem=plan.copy() if mode=='plan' else None
    mud=mu.copy() if mu is not None else np.zeros(nC)
    rho=np.maximum(quota,1.0)/len(stream)
    vt=vm=0.0
    for t,s in enumerate(stream):
        if mode=='replan' and every and t%every==0:
            rem=solve_lp(forecast*max(1-t/len(stream),1e-9),np.maximum(quota-served,0),P)
        c=None
        if rem is not None:
            cand=np.where(rem[:,s]>0.5)[0]; cand=cand[served[cand]<quota[cand]]
            if len(cand): c=cand[np.argmax(rem[cand,s])]; rem[c,s]-=1
        if c is None and mode in ('fixeddual','dmd'):
            ok=np.where(served<quota)[0]
            if len(ok): c=ok[np.argmax(P[ok,s]-mud[ok])]
        if c is None:
            for cc in order[:,s]:
                if served[cc]<quota[cc]: c=cc; break
        served[c]+=1; vt+=theta[c,s]; vm+=P_eval[c,s]
        if mode=='dmd':
            gr=rho.copy(); gr[c]-=1.0
            mud=np.maximum(0.0,mud-eta*gr) if geom=='eucl' else np.clip(mud*np.exp(-eta*gr),1e-9,10.0)
    return vt,vm

pols={'greedy':dict(mode='greedy'),'planned':dict(mode='plan'),
      're-solved':dict(mode='replan',every=20000),
      'fixed-dual':dict(mode='fixeddual',mu=mu_star),
      'DMD-entropic':dict(mode='dmd',mu=(np.full(nC,g*0.1)),eta=0.5*g,geom='entr')}
res={name:serve(**kw) for name,kw in pols.items()}
gt,gm=res['greedy']
print(f"kappa={KAPPA}  (true theta range {theta.min():.3f}-{theta.max():.3f}, mean {theta.mean():.3f})")
print(f"{'policy':>10} | {'TRUE lift':>10} | {'MODEL lift':>11} | {'bias(pp of greedy)':>18}")
rows=[]
for name in pols:
    vt,vm=res[name]
    tl=(vt/gt-1)*100; ml=(vm/gm-1)*100
    rows.append((name,tl,ml)); print(f"{name:>10} | {tl:>+9.2f}% | {vm/gm*100-100:>+10.2f}% | {ml-tl:>+17.2f}")
names=[r[0] for r in rows]; tl=[r[1] for r in rows]; ml=[r[2] for r in rows]
rho=spearmanr(tl,ml).correlation if len(set(tl))>1 else 1.0
print(f"\nSpearman(model ranking, true ranking) = {rho:.3f}")
print(f"model-best = {names[int(np.argmax(ml))]}   true-best = {names[int(np.argmax(tl))]}   MATCH={names[int(np.argmax(ml))]==names[int(np.argmax(tl))]}")
