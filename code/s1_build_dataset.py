"""
Step 1: Curate the UK-CIV dataset (UK Carbon-Intensity Vintages) from the scraped NESO archive.
Input : SQLite archive produced by the nmpowell/carbon-intensity-forecast-tracking git-scraper (MIT; NESO data CC BY 4.0)
Output: data/ukciv/{vintages.parquet, targets.parquet, captures.parquet, tensor.npz, qc.json}
"""
import sqlite3, glob, json, os, numpy as np, pandas as pd
R = "/home/claude/repos/carbon-intensity-forecast-tracking/data/db"
OUT = "/home/claude/data/ukciv"; os.makedirs(OUT, exist_ok=True)
LMIN, LMAX = -48, 95

# ---------- national vintages ----------
parts, capparts = [], []
for f in sorted(glob.glob(f"{R}/20*/national_*.sqlite")):
    c = sqlite3.connect(f)
    parts.append(pd.read_sql("select window_utc, capture_utc, forecast, actual from national_intensity", c))
    capparts.append(pd.read_sql("select distinct capture_utc, source from captures", c))
df = pd.concat(parts, ignore_index=True).drop_duplicates(["window_utc", "capture_utc"], keep="last")
caps = pd.concat(capparts).drop_duplicates("capture_utc").sort_values("capture_utc").reset_index(drop=True)
t0 = int(df.window_utc.min()); t1 = int(df.window_utc.max()); T = (t1 - t0)//1800 + 1
df["w"] = ((df.window_utc - t0)//1800).astype(int)
df["lead"] = ((df.window_utc - df.capture_utc)//1800).astype(int)
df["s"] = df.w - df.lead                      # issue (capture) slot
assert df.lead.between(LMIN, LMAX).all()

# ---------- tensors ----------
NL = LMAX - LMIN + 1
F  = np.full((T, NL), np.nan, np.float32); F[df.w.values, df.lead.values - LMIN] = df.forecast.values
Av = np.full((T, NL), np.nan, np.float32); Av[df.w.values, df.lead.values - LMIN] = df.actual.values
A  = np.full(T, np.nan, np.float32)
for j in range(NL):                            # smallest lead = latest capture => final actual
    col = Av[:, j]; m = np.isnan(A) & ~np.isnan(col); A[m] = col[m]

# ---------- generation mix (per-mille shares), final value per window ----------
mix = pd.concat([pd.read_sql("select * from generation_mix", sqlite3.connect(f)) for f in sorted(glob.glob(f"{R}/20*/generation_*.sqlite"))])
mix = mix.sort_values("capture_utc").groupby("window_utc").last().drop(columns="capture_utc")
MIXC = ["biomass","coal","gas","hydro","imports","nuclear","other","solar","wind"]
G = np.full((T, len(MIXC)), np.nan, np.float32)
gi = ((mix.index.values - t0)//1800).astype(int); ok = (gi >= 0) & (gi < T)
G[gi[ok]] = mix[MIXC].values[ok].astype(np.float32)

# ---------- QC ----------
qc = {}
qc["n_rows_vintage"] = int(len(df)); qc["n_captures"] = int(df.s.nunique()); qc["n_windows_with_forecast"] = int(df.w.nunique())
qc["n_windows_final_actual"] = int(np.isfinite(A).sum())
qc["span_first_window"] = str(pd.to_datetime(t0, unit="s")); qc["span_last_window"] = str(pd.to_datetime(t1, unit="s"))
cs = np.sort(df.s.unique()); gaps = np.diff(cs)
qc["capture_gap_share_gt1slot"] = float((gaps > 1).mean()); qc["capture_max_gap_slots_within_period"] = int(gaps[gaps < 400].max())
qc["zero_actual_windows"] = int((A == 0).sum()); qc["zero_forecast_cells"] = int((F == 0).sum())
a = df.dropna(subset=["actual"]).groupby("w").actual.agg(["nunique", "min", "max"])
qc["share_windows_actual_revised"] = float((a["nunique"] > 1).mean()); qc["mean_actual_revision_range"] = float((a["max"] - a["min"]).mean())
qc["forecast_range"] = [float(np.nanmin(F)), float(np.nanmax(F))]; qc["actual_range"] = [float(np.nanmin(A)), float(np.nanmax(A))]
# suspicious windows: actual==0 -> excluded from targets
bad = np.isfinite(A) & (A <= 0); A_clean = A.copy(); A_clean[bad] = np.nan
qc["windows_excluded_for_zero_actual"] = int(bad.sum())
# sources of captures
qc["capture_sources"] = caps.source.value_counts().to_dict()

# ---------- splits (by ISSUE time, 2-day embargo) ----------
ts = pd.to_datetime(t0 + 1800*np.arange(T), unit="s")
def sl(a, b): return (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b))
splits = {
 "train": sl("2023-03-01","2023-10-29"), "val": sl("2023-11-01","2023-11-29"),
 "cal":   sl("2023-12-01","2023-12-30"), "testA": sl("2024-01-01","2024-02-02"),
 "testB": sl("2026-07-15","2026-10-25"),
}
split_arr = np.full(T, "", dtype=object)
for k, m in splits.items(): split_arr[m] = k
qc["splits_issue_windows"] = {k: [str(ts[m][0]), str(ts[m][-1])] for k, m in splits.items()}

# ---------- save ----------
np.savez_compressed(f"{OUT}/tensor.npz", F=F, Av=Av, A=A_clean, A_raw=A, G=G, t0=t0, LMIN=LMIN, split=split_arr.astype(str))
cap_slots = np.unique(df.s.values)
pd.DataFrame({"capture_utc": t0 + 1800*cap_slots, "issue_slot": cap_slots}).to_parquet(f"{OUT}/captures.parquet")
vint = df[["window_utc", "capture_utc", "lead", "forecast", "actual"]].rename(columns={"lead": "lead_halfhours", "actual": "actual_visible_at_capture"})
vint.to_parquet(f"{OUT}/vintages.parquet", index=False)
tgt = pd.DataFrame({"window_utc": t0 + 1800*np.arange(T), "actual_final": A_clean})
for i, c in enumerate(MIXC): tgt[f"mix_{c}_permille"] = G[:, i]
tgt = tgt[np.isfinite(tgt.actual_final) | np.isfinite(tgt.mix_gas_permille)]
tgt.to_parquet(f"{OUT}/targets.parquet", index=False)
json.dump(qc, open("/home/claude/results/qc.json", "w"), indent=1)
json.dump(qc, open(f"{OUT}/qc.json", "w"), indent=1)
print(json.dumps(qc, indent=1)); print("T slots", T, "t0", t0)
