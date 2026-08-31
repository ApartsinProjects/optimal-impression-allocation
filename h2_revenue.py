"""Scarce-inventory revenue objective (IJOC review P0 #5).

The paper's core experiments maximize expected CLICKS under hard campaign
quotas. A referee asked whether the regime map (adaptation collapses under
heavy contention; re-solve and the fixed optimal dual survive) is an artifact
of that pure-click objective, or whether it holds under a realistic
scarce-inventory REVENUE objective with heterogeneous campaign values and
underdelivery penalties.

Reformulation (guaranteed delivery, e.g. Balseiro-Feldman-Mirrokni-Muthukrishnan
2014). Each campaign c has a booked quota Q_c and a payment type:
  - CPC campaigns pay per click: per-impression value on segment s is
    cpc_c * p_cs, so the click INTERACTION drives their revenue (segment matters).
  - CPM campaigns pay per impression: per-impression value is a flat cpm_c,
    independent of segment (reach matters, not interaction).
Values are calibrated so both types have mean value-per-impression ~1 (lognormal
heterogeneity), so neither type trivially dominates; the economically meaningful
difference is that CPC value is segment-dependent and CPM value is flat, which
breaks the alignment between "high click probability" and "high revenue" that
the click-max objective assumes.

Delivery is capped at the booked quota (advertisers do not want overdelivery);
underdelivery of a guaranteed campaign costs a makegood penalty beta_c per
undelivered impression, beta_c = penalty_mult * value-scale_c. Net revenue of a
policy = realized gross value of served impressions - underdelivery penalty.

Dual-model scoring is preserved: policies plan and serve on the planning-window
table P; all realized values are scored on an independent serving-window table
P_eval. CPM value is table-independent, so no winner's-curse there; the CPC
component still obeys the dual-model guard.

Contention is swept by scaling all quotas (over-subscription), the same lever as
the click-max regime map. We report, per contention level, each policy's net
revenue lift over revenue-greedy and the fill/underdelivery, to see whether the
same policies survive.
"""
import os, sys
import numpy as np
import duckdb
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

DB = r"E:\Projects\Submitted\MobileAd_Comverse\research\data\taobao\prep\taobao.duckdb"
ALPHA, MINC, TOPK, NS = 20.0, 30, 100, 8
PENALTY_MULT = float(os.environ.get("PENALTY", "1.0"))   # makegood cost per undelivered imp, in value-scale units
SCALES = [float(x) for x in os.environ.get("SCALES", "1.0,1.1,1.25,1.5").split(",")]
rng = np.random.default_rng(int(os.environ.get("SEED","0")))
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
# tiebreak on campaign_id: count(*) ties at the LIMIT boundary otherwise leave the
# top-K index order unstable run to run, which reshuffles the per-campaign value draw.
top=[int(r[0]) for r in con.execute(f"SELECT campaign_id FROM evs WHERE day BETWEEN 6 AND 11 GROUP BY 1 ORDER BY count(*) DESC, campaign_id LIMIT {TOPK}").fetchall()]
cix={c:i for i,c in enumerate(top)}; nC=len(top)
con.execute("CREATE TEMP TABLE topc AS SELECT UNNEST(?) AS campaign_id",[top])
_gs,_gn=con.execute("SELECT sum(clk), count(*) FROM evs WHERE day BETWEEN 6 AND 11").fetchone()
g=_gs/_gn   # exact integer ratio: avoids DuckDB parallel-float avg() varying in low bits run to run

def build_P(dfilt):
    """shrunk (campaign, seg) click table; campaign fallback for thin cells."""
    camp=con.execute(f"SELECT campaign_id, sum(clk) s, count(*) n FROM evs JOIN topc USING (campaign_id) WHERE {dfilt} GROUP BY 1").df()
    cs=con.execute(f"SELECT campaign_id, seg, sum(clk) s, count(*) n FROM evs JOIN topc USING (campaign_id) WHERE {dfilt} GROUP BY 1,2 HAVING count(*)>={MINC}").df()
    base=np.full(nC,g)
    for cid,s,n in camp.itertuples(index=False):
        if n>=MINC: base[cix[cid]]=(s+ALPHA*g)/(n+ALPHA)
    M=np.tile(base[:,None],(1,NS))
    for cid,sg,s,n in cs.itertuples(index=False):
        M[cix[cid],int(sg)]=(s+ALPHA*base[cix[cid]])/(n+ALPHA)
    return M

P=build_P("day BETWEEN 6 AND 11")           # planning table
P_eval=build_P("day IN (12,13)")            # independent scoring table
pbar=np.maximum(P.mean(1),1e-4)             # campaign avg click prob (planning)

# campaign payment types + calibrated values (mean value-per-imp ~1 both types)
# type is drawn at random per seed so payment type does not correlate with volume rank
is_cpc=np.zeros(nC,bool); is_cpc[rng.permutation(nC)[:nC//2]]=True
w=np.exp(rng.normal(0.0,0.5,nC))            # lognormal willingness, mean ~1.13
cpc=np.where(is_cpc, w/pbar, 0.0)           # CPC: value = cpc*p ; at avg p -> w
cpm=np.where(is_cpc, 0.0, w)               # CPM: flat value w
beta=PENALTY_MULT*w                          # makegood per undelivered imp

def gross_plan(): return np.where(is_cpc[:,None], cpc[:,None]*P,      cpm[:,None]*np.ones((1,NS)))
def gross_eval(): return np.where(is_cpc[:,None], cpc[:,None]*P_eval, cpm[:,None]*np.ones((1,NS)))
Vp=gross_plan()                              # planning gross value coeff
Ve=gross_eval()                              # realized gross value coeff
Veff=Vp+beta[:,None]                         # effective marginal delivery value (plan)
order=np.argsort(-Veff,axis=0,kind='stable') # per-segment campaign preference by value (stable ties)

# stream, supply, base quota (realized day 12-13 delivery)
# total order on the stream (time_stamp ties broken deterministically) so the
# arrival sequence -- and every online policy -- is reproducible run to run.
stream=con.execute("SELECT seg FROM evs JOIN topc USING (campaign_id) WHERE day IN (12,13) ORDER BY time_stamp, user_id, campaign_id, clk, seg").df()["seg"].to_numpy()
forecast=np.bincount(stream,minlength=NS).astype(float)
q0=np.zeros(nC)
for cid,c in con.execute("SELECT campaign_id,count(*) c FROM evs JOIN topc USING (campaign_id) WHERE day IN (12,13) GROUP BY 1").df().itertuples(index=False):
    q0[cix[cid]]=c

def solve_lp(supply,q,val):
    idx=[(c,s) for c in range(nC) for s in range(NS)]
    A=lil_matrix((NS+nC,len(idx)))
    for j,(c,s) in enumerate(idx): A[s,j]=1; A[NS+c,j]=1
    r=linprog([-val[c,s] for c,s in idx],A_ub=A.tocsr(),
              b_ub=np.concatenate([supply,q]).astype(float),bounds=(0,None),method='highs')
    x=np.zeros((nC,NS))
    for j,(c,s) in enumerate(idx): x[c,s]=r.x[j]
    duals=np.maximum(-np.asarray(r.ineqlin.marginals)[NS:],0.0)
    return x,duals

def net(served, gross):
    """net revenue = realized gross - underdelivery penalty, given served counts."""
    return gross - float((beta*np.maximum(quota-served,0)).sum())

def serve(mode,quota,plan=None,mu=None,eta=0.0,every=0,geom=None):
    served=np.zeros(nC); rem=plan.copy() if plan is not None else None
    mud=mu.copy() if mu is not None else np.zeros(nC)
    rho=np.maximum(quota,1.0)/len(stream)
    gross=0.0
    for t,s in enumerate(stream):
        if mode=='replan' and every and t%every==0:
            rem,_=solve_lp(forecast*max(1-t/len(stream),1e-9),np.maximum(quota-served,0),Veff)
        c=None
        if rem is not None:
            cand=np.where(rem[:,s]>0.5)[0]; cand=cand[served[cand]<quota[cand]]
            if len(cand): c=cand[np.argmax(rem[cand,s])]; rem[c,s]-=1
        if c is None and mode in ('fixeddual','dmd'):
            ok=np.where(served<quota)[0]
            if len(ok): c=ok[np.argmax(Veff[ok,s]-mud[ok])]  # serve argmax value-minus-price among quota-remaining
        if c is None:                                    # greedy and any fallback: fill by value order
            for cc in order[:,s]:
                if served[cc]<quota[cc]: c=cc; break
        if c is None: continue
        served[c]+=1; gross+=Ve[c,s]
        if mode=='dmd':
            gr=rho.copy(); gr[c]-=1.0
            mud=np.maximum(0.0,mud-eta*gr) if geom=='eucl' else np.clip(mud*np.exp(-eta*gr),1e-9,10.0)
    return net(served,gross), served

print(f"CPC campaigns {is_cpc.sum()}, CPM {(~is_cpc).sum()}; penalty_mult={PENALTY_MULT}")
print(f"value-per-imp: CPC mean {(cpc*pbar)[is_cpc].mean():.3f}, CPM mean {cpm[~is_cpc].mean():.3f}")
print(f"{'scale':>6} | {'greedy$':>9} | " + " | ".join(f"{p:>10}" for p in ['planned','re-solved','fixed-dual','scalar','DMD-entr']) + " | fill%")
for sc in SCALES:
    quota=np.ceil(q0*sc)
    plan,mu_star=solve_lp(forecast,quota,Veff)
    pols={
      'greedy':dict(mode='greedy',quota=quota),
      'planned':dict(mode='plan',quota=quota,plan=plan),
      're-solved':dict(mode='replan',quota=quota,every=20000),
      'fixed-dual':dict(mode='fixeddual',quota=quota,mu=mu_star),
      'scalar':dict(mode='dmd',quota=quota,mu=np.full(nC,g),eta=0.0,geom='eucl'),  # eta=0 -> static scalar? no; use small
      'DMD-entr':dict(mode='dmd',quota=quota,mu=np.full(nC,w.mean()*0.1),eta=0.5*w.mean(),geom='entr'),
    }
    # scalar pacing: euclidean DMD with a single learning rate on all duals
    pols['scalar']=dict(mode='dmd',quota=quota,mu=np.zeros(nC),eta=0.5*w.mean(),geom='eucl')
    res={n:serve(**kw) for n,kw in pols.items()}
    gnet,gserved=res['greedy']
    fill=gserved.sum()/quota.sum()*100
    lifts=[(res[n][0]/gnet-1)*100 for n in ['planned','re-solved','fixed-dual','scalar','DMD-entr']]
    print(f"{sc:>6.2f} | {gnet:>9.0f} | " + " | ".join(f"{x:>+9.2f}%" for x in lifts) + f" | {fill:>4.0f}")
    if os.environ.get("DEBUG"):
        # decompose greedy vs fixed-dual: gross, penalty, served mass, dual summary
        def decomp(name,kw):
            served=np.zeros(nC)
            netv,served=serve(**kw)  # re-run to get served
            # recompute gross and penalty from a fresh pass
            return netv,served
        for name in ['greedy','fixed-dual','re-solved']:
            netv,served=res[name]
            pen=float((beta*np.maximum(quota-served,0)).sum())
            gross=netv+pen
            print(f"      [{name:>10}] gross {gross:>9.0f}  penalty {pen:>8.0f}  served {served.sum():>7.0f}/{quota.sum():.0f}")
        nz=mu_star>1e-9
        print(f"      mu*: {nz.sum()} nonzero of {nC}, max {mu_star.max():.3f}, mean(nz) {mu_star[nz].mean() if nz.any() else 0:.3f}; quota slack campaigns (served<Q under greedy): {(gserved<quota).sum()}")
