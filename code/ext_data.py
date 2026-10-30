"""Loader for the 13-region CarbonCast Jul-Dec 2021 data: hourly actual direct carbon intensity and 96-h forecasts re-issued daily."""
import numpy as np, pandas as pd, glob, os
ROOT = "/home/claude/repos/CarbonCast/data"
REGIONS = ["AUS_QLD", "BPAT", "CISO", "DE", "ERCO", "ES", "FPL", "ISNE", "NL", "NYISO", "PJM", "PL", "SE"]
def load_region(r):
    f = glob.glob(f"{ROOT}/{r}/{r}_carbon_from_src*forecasts_direct_Jul_Dec_2021.csv")[0]
    d = pd.read_csv(f); d["t"] = pd.to_datetime(d["UTC time"]); d = d.rename(columns={"actual_carbon_intensity_direct": "A", "forecasted_carbon_intensity_direct": "F"})[["t", "A", "F"]]
    t0 = pd.Timestamp("2021-07-01 00:00:00"); T = 184 * 24
    A = np.full(T, np.nan); h = ((d.t - t0).dt.total_seconds() // 3600).astype(int).values; A[h] = d.A.values        # actual per hour (identical across issues)
    # split into issue blocks at negative time jumps
    brk = np.where(np.diff(h) < 0)[0] + 1; starts = np.r_[0, brk]; ends = np.r_[brk, len(d)]
    blocks = {}
    for a, b in zip(starts, ends):
        hs = h[a:b]; st = hs[0]; lead = hs - st
        if len(hs) < 90 or not np.all(np.diff(hs) == 1): continue
        P = np.full(96, np.nan); P[lead] = d.F.values[a:b]; blocks[st] = P          # later duplicate overwrites earlier
    bs = np.array(sorted(blocks)); F = np.stack([blocks[k] for k in bs])
    return dict(A=A, bstart=bs, F=F, T=T, region=r)
if __name__ == "__main__":
    for r in REGIONS:
        R = load_region(r); ok = np.isfinite(R["A"])
        mae = np.nanmean(np.abs(R["A"][R["bstart"][:, None] + np.arange(96)[None, :].clip(0, R["T"] - 1)] - R["F"]), 0) if True else 0
        print(f"{r:8s} hours {R['T']} actual finite {ok.mean():.3f} blocks {len(R['bstart'])} mean CI {np.nanmean(R['A']):6.1f} sd {np.nanstd(R['A']):5.1f}  MAE lead0-23 {np.nanmean(mae[:24]):5.1f} 24-47 {np.nanmean(mae[24:48]):5.1f} 72-95 {np.nanmean(mae[72:]):5.1f}  start gaps {np.unique(np.diff(R['bstart']), return_counts=True)}")
