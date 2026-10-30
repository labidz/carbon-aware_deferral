import pandas as pd, numpy as np
df = pd.read_pickle("data/national_raw.pkl")
df["lead"] = (df.window_utc - df.capture_utc)/1800.0   # in half-hours; >0 future
print("lead range", df.lead.min(), df.lead.max(), "non-integer share:", (df.lead%1!=0).mean())
df["cap_dt"] = pd.to_datetime(df.capture_utc, unit="s")
df["win_dt"] = pd.to_datetime(df.window_utc, unit="s")
caps = df.drop_duplicates("capture_utc")[["capture_utc","cap_dt"]].sort_values("capture_utc")
caps["gap_min"] = caps.capture_utc.diff()/60
print("n captures", len(caps))
print(caps.gap_min.describe())
print("gap>35min share:", (caps.gap_min>35).mean())
# captures per month
m = caps.groupby(caps.cap_dt.dt.to_period("M")).size()
print(m.to_string())
# capture minute-of-hour distribution
print(caps.cap_dt.dt.minute.value_counts().head(10))
# windows
W = df.groupby("window_utc").agg(n=("capture_utc","size"), nf=("forecast","count"), na=("actual","count"), lead_max=("lead","max"), lead_min=("lead","min"))
print(W.describe())
# actual availability: min lead (most negative) where actual first seen
a = df[df.actual.notna()].groupby("window_utc").lead.max()
print("max lead at which actual already present (half-hours):"); print(a.describe())
# does actual change between captures for the same window?
g = df[df.actual.notna()].groupby("window_utc").actual.agg(["nunique","min","max"])
print("share windows with >1 distinct actual:", (g["nunique"]>1).mean(), " mean range:", (g["max"]-g["min"]).mean())
print(g[g["nunique"]>1].head())
# windows with 'final' actual
fin = df[df.actual.notna()].sort_values("capture_utc").groupby("window_utc").actual.last()
print("windows with final actual:", len(fin), " of ", df.window_utc.nunique())
# range of dates covered by windows with actuals
fd = pd.to_datetime(fin.index, unit="s")
print(pd.Series(fd).groupby(pd.Series(fd).dt.to_period("M")).size().to_string())
