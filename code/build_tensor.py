import pandas as pd, numpy as np
df = pd.read_pickle("data/national_raw.pkl")
df = df.drop(columns=["file"])
# duplicates?
dup = df.duplicated(["window_utc","capture_utc"]).sum()
print("duplicate (window,capture) rows:", dup)
df = df.drop_duplicates(["window_utc","capture_utc"], keep="last")
t0 = df.window_utc.min(); t1 = df.window_utc.max()
T = int((t1-t0)//1800)+1
LMIN, LMAX = -48, 95
NL = LMAX-LMIN+1
print("slots", T, "leads", NL)
widx = ((df.window_utc - t0)//1800).astype(int).values
lead = ((df.window_utc - df.capture_utc)//1800).astype(int).values
F = np.full((T, NL), np.nan, dtype=np.float32)
F[widx, lead-LMIN] = df.forecast.values.astype(np.float32)
# actual per capture too (so we can reconstruct what was known at issue time)
A_v = np.full((T, NL), np.nan, dtype=np.float32)
A_v[widx, lead-LMIN] = df.actual.values.astype(np.float32)
# final actual = last non-null along capture order = smallest lead index (most negative lead = latest capture)
A_final = np.full(T, np.nan, dtype=np.float32)
for j in range(NL):  # lead ascending = capture time descending; we want latest => iterate from largest lead down to smallest
    pass
# latest capture = smallest lead; take first non-nan scanning lead from LMIN upward
for j in range(NL):
    col = A_v[:, j]
    m = np.isnan(A_final) & ~np.isnan(col)
    A_final[m] = col[m]
print("windows with final actual:", np.isfinite(A_final).sum())
np.savez_compressed("data/vintage_tensor.npz", F=F, A_v=A_v, A=A_final, t0=t0, LMIN=LMIN)
# coverage
cov = np.isfinite(F).mean(0)
print("coverage by lead (selected):", {int(l):round(float(cov[l-LMIN]),2) for l in [-48,-24,-1,0,1,2,4,8,12,24,48,72,95]})
ts = pd.to_datetime(t0 + 1800*np.arange(T), unit="s")
ok = np.isfinite(A_final)
print("actual coverage by month:"); print(pd.Series(ok, index=ts).groupby(ts.to_period("M")).mean().round(2).to_string())
