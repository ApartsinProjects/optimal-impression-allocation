"""Verify Algorithm 1 (PGSS) selects the claimed depth blind, per market.
For Taobao top-100 and top-500: measure activity-tier interaction persistence
at depths 4/8/16 on the planning window (days 6-11 split 6-9 vs 10-11), and the
fraction of planned-LP mass landing in cells that meet the serving-window min
count. Apply the rule: admissible = support>=1-tau AND persistence>rho0; pick
finest admissible depth. Reports selected depth vs the ex-post best-lift depth.
"""
import numpy as np
import duckdb
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

DB = r"E:\Projects\Submitted\MobileAd_Comverse\research\data\taobao\prep\taobao.duckdb"
TAU, MINC, RHO0 = 0.20, 30, 0.15
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA memory_limit='3GB'"); con.execute("PRAGMA threads=2")
con.execute("""CREATE OR REPLACE TEMP VIEW ev AS
  SELECT raw.user AS user_id, raw.clk,
         date_part('day', to_timestamp(raw.time_stamp + 8*3600) AT TIME ZONE 'UTC') AS day,
         ad.campaign_id FROM raw JOIN ad USING (adgroup_id)""")

TIERS = {
 4:  "CASE WHEN c>=100 THEN 3 WHEN c>=30 THEN 2 WHEN c>=10 THEN 1 ELSE 0 END",
 8:  "CASE WHEN c>=200 THEN 7 WHEN c>=100 THEN 6 WHEN c>=50 THEN 5 WHEN c>=30 THEN 4 WHEN c>=20 THEN 3 WHEN c>=10 THEN 2 WHEN c>=5 THEN 1 ELSE 0 END",
 16: "CASE WHEN c>=800 THEN 15 WHEN c>=500 THEN 14 WHEN c>=300 THEN 13 WHEN c>=200 THEN 12 WHEN c>=150 THEN 11 WHEN c>=100 THEN 10 WHEN c>=75 THEN 9 WHEN c>=50 THEN 8 WHEN c>=40 THEN 7 WHEN c>=30 THEN 6 WHEN c>=20 THEN 5 WHEN c>=15 THEN 4 WHEN c>=10 THEN 3 WHEN c>=7 THEN 2 WHEN c>=5 THEN 1 ELSE 0 END",
}

def tier_table(dlo, dhi):
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tc AS
      SELECT user_id, count(*) c FROM ev WHERE day BETWEEN {dlo} AND {dhi} GROUP BY 1""")

def persistence(depth, top):
    q = lambda lo, hi: con.execute(f"""
      SELECT campaign_id, {TIERS[depth]} AS g, sum(clk) s, count(*) n
      FROM ev JOIN tc USING (user_id)
      WHERE day BETWEEN {lo} AND {hi} AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1,2""",[top]).df()
    a, b = q(6, 9), q(10, 11)
    m = a.merge(b, on=['campaign_id','g'], suffixes=('_a','_b'))
    m = m[(m.n_a>=300)&(m.n_b>=150)].copy()
    if len(m) < 20: return None
    for w in ('a','b'):
        camp=m.groupby('campaign_id').apply(lambda x:x[f's_{w}'].sum()/x[f'n_{w}'].sum(),include_groups=False)
        grp=m.groupby('g').apply(lambda x:x[f's_{w}'].sum()/x[f'n_{w}'].sum(),include_groups=False)
        gl=m[f's_{w}'].sum()/m[f'n_{w}'].sum()
        m[f'r_{w}']=m[f's_{w}']/m[f'n_{w}']-camp.reindex(m.campaign_id).to_numpy()-grp.reindex(m.g).to_numpy()+gl
    # bootstrap LCB over cells
    r=np.corrcoef(m.r_a,m.r_b)[0,1]; n=len(m); rng=np.random.default_rng(0)
    bs=[np.corrcoef(m.r_a.to_numpy()[i],m.r_b.to_numpy()[i])[0,1] for i in (rng.integers(0,n,n) for _ in range(200))]
    return r, float(np.percentile(bs,5))

def solve_lp(nC,nS,supply,quota,P):
    idx=[(c,s) for c in range(nC) for s in range(nS)]
    A=lil_matrix((nS+nC,len(idx)))
    for j,(c,s) in enumerate(idx): A[s,j]=1; A[nS+c,j]=1
    r=linprog([-P[c,s] for c,s in idx],A_ub=A.tocsr(),b_ub=np.concatenate([supply,quota]).astype(float),bounds=(0,None),method='highs')
    x=np.zeros((nC,nS))
    for j,(c,s) in enumerate(idx): x[c,s]=r.x[j]
    return x

def support_frac(depth, top):
    # planned mass in cells whose serving-window (day 12-13) count >= MINC
    con.execute(f"""CREATE OR REPLACE TEMP VIEW evs AS
      SELECT ev.*, {TIERS[depth]} AS seg FROM ev JOIN tc USING (user_id)""")
    nS=depth; cix={c:i for i,c in enumerate(top)}; nC=len(top)
    g=con.execute("SELECT avg(clk) FROM evs WHERE day BETWEEN 6 AND 11").fetchone()[0]
    def tbl(df):
        camp=df.groupby('campaign_id').agg(s=('clk','sum'),n=('clk','count'))
        cs=df.groupby(['campaign_id','seg']).agg(s=('clk','sum'),n=('clk','count'))
        M=np.zeros((nC,nS)); base=np.full(nC,g)
        for cid,row in camp.iterrows():
            if cid in cix and row.n>=MINC: base[cix[cid]]=(row.s+20*g)/(row.n+20)
        M[:]=base[:,None]
        for (cid,sg),row in cs.iterrows():
            if cid in cix and row.n>=MINC: M[cix[cid],int(sg)]=(row.s+20*base[cix[cid]])/(row.n+20)
        return M
    tr=con.execute("SELECT campaign_id,seg,clk FROM evs WHERE day BETWEEN 6 AND 11 AND campaign_id IN (SELECT UNNEST(?))",[top]).df()
    P=tbl(tr)
    fc=con.execute("SELECT seg,count(*) c FROM evs WHERE day BETWEEN 6 AND 11 AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1",[top]).df()
    forecast=np.zeros(nS); forecast[fc.seg.to_numpy()]=fc.c.to_numpy()/6.0*2.0
    quota=np.zeros(nC)
    rq=con.execute("SELECT campaign_id,count(*) c FROM evs WHERE day IN (12,13) AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1",[top]).df()
    for cid,c in zip(rq.campaign_id,rq.c):
        if cid in cix: quota[cix[cid]]=c
    plan=solve_lp(nC,nS,forecast,quota,P)
    Nserv=np.zeros((nC,nS))
    sc=con.execute("SELECT campaign_id,seg,count(*) c FROM evs WHERE day IN (12,13) AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1,2",[top]).df()
    for cid,sg,c in zip(sc.campaign_id,sc.seg,sc.c):
        if cid in cix: Nserv[cix[cid],int(sg)]=c
    # training-window count per cell, and median over cells carrying planned mass
    Ntr=np.zeros((nC,nS))
    tc2=con.execute("SELECT campaign_id,seg,count(*) c FROM evs WHERE day BETWEEN 6 AND 11 AND campaign_id IN (SELECT UNNEST(?)) GROUP BY 1,2",[top]).df()
    for cid,sg,c in zip(tc2.campaign_id,tc2.seg,tc2.c):
        if cid in cix: Ntr[cix[cid],int(sg)]=c
    active = plan > (0.001*plan.sum()/max((plan>0).sum(),1))
    med_train = float(np.median(Ntr[active])) if active.any() else 0.0
    supported=plan*(Nserv>=MINC)
    return supported.sum()/max(plan.sum(),1), med_train

for topk in (100,500):
    top=[int(r[0]) for r in con.execute(f"SELECT campaign_id FROM ev WHERE day BETWEEN 6 AND 11 GROUP BY 1 ORDER BY count(*) DESC LIMIT {topk}").fetchall()]
    tier_table(6,11)
    print(f"== top-{topk}")
    admissible=[]
    for d in (4,8,16):
        pr=persistence(d,top); sf,medtr=support_frac(d,top)
        ok = sf>=1-TAU and pr and pr[1]>RHO0
        print(f"  {d} tiers: persistence r={pr[0]:.3f} LCB={pr[1]:.3f}  support_frac={sf:.3f}  median_train_per_cell={medtr:.0f}  admissible={ok}")
        if ok: admissible.append(d)
    print(f"  finest-admissible depth = {max(admissible) if admissible else 'ABSTAIN'}")
