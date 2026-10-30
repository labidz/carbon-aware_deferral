import sqlite3, glob, pandas as pd, numpy as np, os
R = "/home/claude/repos/carbon-intensity-forecast-tracking/data/db"
files = sorted(glob.glob(f"{R}/20*/national_*.sqlite"))
parts = []
for f in files:
    c = sqlite3.connect(f)
    df = pd.read_sql("select window_utc, capture_utc, forecast, actual from national_intensity", c)
    caps = pd.read_sql("select capture_utc, endpoint, window_first_utc, window_last_utc, source from captures", c)
    df["file"] = os.path.basename(f)
    parts.append(df)
    print(os.path.basename(f), len(df), "caps:", caps.capture_utc.nunique(), "endpoints:", sorted(caps.endpoint.unique())[:6], "sources:", caps.source.unique()[:3])
df = pd.concat(parts, ignore_index=True)
df.to_parquet("/home/claude/data/national_raw.parquet") if False else df.to_pickle("/home/claude/data/national_raw.pkl")
print(df.shape)
print(df.head())
print(df.describe().T)
