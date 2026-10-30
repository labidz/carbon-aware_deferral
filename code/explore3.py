import numpy as np, pandas as pd
d = np.load("data/vintage_tensor.npz"); F=d["F"]; Av=d["A_v"]; A=d["A"]; t0=int(d["t0"]); LMIN=int(d["LMIN"])
T=len(A); ts = pd.to_datetime(t0+1800*np.arange(T), unit="s")
ok=np.isfinite(A)
print("zeros in actual:", (A==0).sum(), " zeros in forecasts:", (F==0).sum(), " min nonzero actual:", np.nanmin(np.where(A>0,A,np.nan)))
zi = np.where(A==0)[0]; print("zero-actual dates:", pd.Series(ts[zi]).dt.date.value_counts().head(8).to_dict())
# revision analysis
def col(L): return F[:, L-LMIN]
print("\nRevision efficiency tests (2023-24 only), target=A-F(L) ; r1=F(L)-F(L+1) (latest revision), r4=F(L)-F(L+4)")
m23 = ok & (ts<pd.Timestamp("2025-01-01"))
for L in [2,6,12,24,40,60]:
    f=col(L); r1=f-col(L+1); r4=f-col(L+4); r12=f-col(L+12)
    e=A-f
    k=m23&np.isfinite(e)&np.isfinite(r1)&np.isfinite(r4)&np.isfinite(r12)
    c1=np.corrcoef(e[k],r1[k])[0,1]; c4=np.corrcoef(e[k],r4[k])[0,1]; c12=np.corrcoef(e[k],r12[k])[0,1]
    # slope of e on r4
    sl=np.polyfit(r4[k],e[k],1)[0]
    # revision autocorrelation: r1(L) vs r1(L+1)
    r1b=col(L+1)-col(L+2); kk=k&np.isfinite(r1b); ac=np.corrcoef(r1[kk],r1b[kk])[0,1]
    print(f"L={L:2d} n={k.sum():6d} corr(e,r1)={c1:+.3f} corr(e,r4)={c4:+.3f} corr(e,r12)={c12:+.3f} slope(e~r4)={sl:+.3f}  rev-autocorr={ac:+.3f}  sd(r1)={np.std(r1[k]):.2f}")
# persistence of error: same-lead error of windows just realised
print("\nSame-lead recent error persistence")
for L in [4,12,24,48]:
    f=col(L); e=A-f
    # at capture time c = w - L; windows realised by c: w' <= c-2 = w-L-2 ; recent error of those windows at same lead L
    idx=np.arange(T)
    prev = np.full(T,np.nan)
    for lag in range(L+2, L+2+4):   # windows w-lag, lag>=L+2 ; error at same lead L
        sh = np.full(T,np.nan); sh[lag:] = e[:T-lag]; 
        prev = np.where(np.isnan(prev), sh, prev) if lag==L+2 else np.nanmean(np.vstack([prev,sh]),axis=0)
    k=m23&np.isfinite(e)&np.isfinite(prev)
    print(f"L={L:2d} corr(e, mean recent same-lead err)={np.corrcoef(e[k],prev[k])[0,1]:+.3f}  n={k.sum()}")
# hour-of-day bias
print("\nbias(A-F) by hour of day at L=24, 2023-24 and 2026")
h = ts.hour.values
for name,mm in [("2023-24",m23),("2026",ok&(ts>=pd.Timestamp("2026-01-01")))]:
    f=col(24); e=A-f; k=mm&np.isfinite(e)
    print(name, pd.Series(e[k]).groupby(h[k]).mean().round(1).values.tolist())
