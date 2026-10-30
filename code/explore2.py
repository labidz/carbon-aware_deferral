import numpy as np, pandas as pd
d = np.load("data/vintage_tensor.npz"); F=d["F"]; A=d["A"]; t0=int(d["t0"]); LMIN=int(d["LMIN"])
T=len(A); ts = pd.to_datetime(t0+1800*np.arange(T), unit="s")
per = np.where(ts<pd.Timestamp("2025-01-01"), "2023-24", "2026")
ok = np.isfinite(A)
leads = np.arange(LMIN, 96)
rows=[]
for p in ["2023-24","2026"]:
    m = ok & (per==p)
    for L in [0,1,2,4,6,8,12,16,24,32,40,48,64,80,95]:
        f = F[m, L-LMIN]; a = A[m]; k=np.isfinite(f)
        e = (a[k]-f[k])
        rows.append((p,L,L/2,k.sum(),np.mean(np.abs(e)),np.mean(e),np.sqrt(np.mean(e**2)), np.mean(np.abs(e)/np.maximum(a[k],1))*100))
r=pd.DataFrame(rows,columns=["period","lead_hh","lead_h","n","MAE","bias(A-F)","RMSE","MAPE%"])
print(r.round(2).to_string(index=False))
# mean CI level by period
for p in ["2023-24","2026"]:
    m=ok&(per==p); print(p,"mean actual",A[m].mean().round(1),"std",A[m].std().round(1), "min/max",A[m].min(),A[m].max())
# negative lead behavior (after window): is the forecast converging to actual?
for L in [-1,-2,-4,-8,-24,-48]:
    f=F[ok,L-LMIN]; a=A[ok]; k=np.isfinite(f)
    print("lead",L,"n",k.sum(),"MAE vs final actual",np.mean(np.abs(a[k]-f[k])).round(2))
