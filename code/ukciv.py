"""
ukciv.py - shared utilities for the UK-CIV (UK Carbon-Intensity Vintages) study.

Conventions
-----------
* Time is discretised in half-hour slots (NESO settlement periods); slot index w = (window_utc - t0)/1800.
* An *issue* (a.k.a. capture, vintage) at slot s is one scrape of the NESO API. At that moment the
  published forecast for window w = s + L (L = lead in slots, 0..95) is  F[w, L - LMIN].
* The *final actual* A[w] is the last published estimate of realised carbon intensity of window w.
  Av[w, L - LMIN] is the "actual" as visible at the capture made at slot w - L (L <= -2 in practice).
* LEAKAGE RULE: any feature for sample (s, L) may only use information visible at slot s:
    - forecasts F[., lead] issued at or before s (captures at slots <= s),
    - actuals of windows w' <= s - 2, taken from Av as seen at s (revisions after s are NOT used),
    - generation mix of windows w' <= s - 2.
"""
import numpy as np, pandas as pd

D = np.load("/home/claude/data/ukciv/tensor.npz", allow_pickle=True)
F, Av, A, G = D["F"], D["Av"], D["A"], D["G"]
T = len(A); t0 = int(D["t0"]); LMIN = int(D["LMIN"]); SPLIT = D["split"]
NL = F.shape[1]; H = 96                       # forecast horizon (slots) = 48 h
TS = pd.to_datetime(t0 + 1800 * np.arange(T), unit="s")
TOD = (TS.hour.values * 2 + TS.minute.values // 30).astype(np.int16)
DOW = TS.dayofweek.values.astype(np.int16)
MIXC = ["biomass", "coal", "gas", "hydro", "imports", "nuclear", "other", "solar", "wind"]


# ---- cumulative same-lead error tables (settled actuals only: used for windows <= s-49) ----
_E = (A[:, None] - F[:, 0 - LMIN: 0 - LMIN + H]).astype(np.float64)       # (T, H): error of the forecast issued L slots before w
_fin = np.isfinite(_E)
ECUM = np.cumsum(np.where(_fin, _E, 0.0), 0).astype(np.float64)
CCUM = np.cumsum(_fin, 0).astype(np.float64)
EWIN = np.where(_fin, _E, np.nan).astype(np.float32)

LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
REVS = [1, 2, 4, 8, 12, 24, 48]


def getF(w, L):
    """Vectorised F[w, L] with bounds -> NaN."""
    w = np.asarray(w); L = np.asarray(L)
    ok = (w >= 0) & (w < T) & (L >= LMIN) & (L < LMIN + NL)
    out = np.full(np.broadcast(w, L).shape, np.nan, np.float32)
    ww, LL = np.broadcast_arrays(w, L)
    out[ok] = F[ww[ok], LL[ok] - LMIN]
    return out


def vis_actual(wp, s):
    """Actual of window wp as visible at issue slot s (NaN if not yet published)."""
    wp, s = np.broadcast_arrays(np.asarray(wp), np.asarray(s))
    dl = wp - s
    out = np.full(wp.shape, np.nan, np.float32)
    recent = (dl <= -2) & (dl >= LMIN) & (wp >= 0) & (wp < T)
    out[recent] = Av[wp[recent], dl[recent] - LMIN]
    old = (dl < LMIN) & (wp >= 0)
    out[old] = A[wp[old]]
    return out


def all_captures():
    c = pd.read_parquet("/home/claude/data/ukciv/captures.parquet").issue_slot.values
    return c


def complete_view(s):
    s = np.asarray(s); idx = np.arange(H)
    n = np.isfinite(F[np.clip(s[:, None] + idx, 0, T - 1), idx - LMIN]).sum(1)
    return n >= 90


def split_slots(name, stride=1, require_complete=True):
    caps = all_captures()
    m = SPLIT[caps] == name
    c = caps[m]
    if require_complete:
        c = c[complete_view(c)]
    if stride > 1:                              # keep captures on the hour (even slots) etc.
        c = c[(c % stride) == 0]
    return c


def make_samples(S, leads=None, with_seq=False):
    """Build the (issue, lead) sample table. Returns dict of numpy arrays."""
    S = np.asarray(S); leads = np.arange(H) if leads is None else np.asarray(leads)
    s = np.repeat(S, len(leads)); L = np.tile(leads, len(S)); w = s + L
    feats = {}
    f0 = getF(w, L)
    feats["f0"] = f0
    feats["lead"] = L.astype(np.float32)
    # ---- vintage trajectory of the same target ----
    lag_arr = []
    for k in LAGS:
        v = getF(w, L + k); lag_arr.append(v)
    LA = np.stack(lag_arr, 1)
    for k, v in zip(LAGS, lag_arr):
        feats[f"lag{k}"] = v
    for k in REVS:
        feats[f"rev{k}"] = f0 - getF(w, L + k)
    feats["vint_mean"] = np.nanmean(LA, 1) if LA.shape[1] else np.nan
    feats["vint_std"] = np.nanstd(LA, 1)
    feats["vint_n"] = np.isfinite(LA).sum(1).astype(np.float32)
    first = getF(w, np.full_like(L, H - 1))
    feats["rev_total"] = np.where(L < H - 1, f0 - first, np.nan)
    # ---- forecast path context at issue ----
    P = F[np.clip(S[:, None] + np.arange(H), 0, T - 1), np.arange(H) - LMIN]          # (nS, H)
    pm, pmin, pmax, psd = np.nanmean(P, 1), np.nanmin(P, 1), np.nanmax(P, 1), np.nanstd(P, 1)
    ps = np.repeat(np.arange(len(S)), len(leads))
    feats["path_mean"] = pm[ps]; feats["path_min"] = pmin[ps]; feats["path_max"] = pmax[ps]; feats["path_sd"] = psd[ps]
    feats["f0_rel"] = f0 - pm[ps]
    feats["f_prev"] = getF(w - 1, L - 1); feats["f_next"] = getF(w + 1, L + 1)
    feats["f_slope"] = feats["f_next"] - feats["f_prev"]
    # local mean +-3
    loc = np.stack([getF(w + j, L + j) for j in range(-3, 4)], 1)
    feats["f_loc7"] = np.nanmean(loc, 1)
    feats["f_rank48"] = np.nanmean(P[ps, :48] <= f0[:, None], 1)
    # ---- recently realised information (visible at s) ----
    J = np.arange(2, 50)
    V = vis_actual(S[:, None] - J[None, :], S[:, None])                                # (nS, 48)
    Pp = F[np.clip(S[:, None] - J, 0, T - 1), -J - LMIN]                              # forecast for past windows as of s
    E = V - Pp
    feats["a_last"] = V[:, 0][ps]; feats["a_mean4"] = np.nanmean(V[:, :4], 1)[ps]
    feats["a_mean12"] = np.nanmean(V[:, :12], 1)[ps]; feats["a_mean48"] = np.nanmean(V, 1)[ps]
    feats["a_trend"] = feats["a_last"] - feats["a_mean12"]
    feats["nowc_err4"] = np.nanmean(E[:, :4], 1)[ps]; feats["nowc_err24"] = np.nanmean(E[:, :24], 1)[ps]
    feats["f0_minus_alast"] = f0 - feats["a_last"]
    es = []
    for j in range(2, 26):
        wj = s - j
        es.append(vis_actual(wj, s) - getF(wj, L))
    ES = np.stack(es, 1)
    feats["e_same4"] = np.nanmean(ES[:, :4], 1); feats["e_same24"] = np.nanmean(ES, 1)
    feats["e_same_last"] = ES[:, 0]
    feats["a_yday"] = vis_actual(w - 48, s); feats["a_2d"] = vis_actual(w - 96, s); feats["a_lw"] = vis_actual(w - 336, s)
    feats["e_yday"] = feats["a_yday"] - getF(w - 48, L)
    feats["f0_minus_yday"] = f0 - feats["a_yday"]
    # ---- adaptive error memory (settled windows only; regime-tracking bias) ----
    def roll_err(K):
        b = np.clip(s - 49, 0, T - 1); a = np.clip(b - K, -1, T - 1)       # window range (a, b]
        num = ECUM[b, L] - np.where(a >= 0, ECUM[np.maximum(a, 0), L], 0.0)
        den = CCUM[b, L] - np.where(a >= 0, CCUM[np.maximum(a, 0), L], 0.0)
        return np.where(den >= 20, num / np.maximum(den, 1), np.nan).astype(np.float32)
    feats["e_wk"] = roll_err(336); feats["e_mo"] = roll_err(1344)
    acc7, acc28 = [], []
    for k in range(1, 29):
        wk = w - 48 * k; okk = (wk <= s - 49) & (wk >= 0)
        v = np.where(okk, EWIN[np.clip(wk, 0, T - 1), L], np.nan)
        (acc7 if k <= 7 else acc28).append(v)
    A7 = np.stack(acc7, 1); A28 = np.stack(acc7 + acc28, 1)
    feats["e_tod_wk"] = np.nanmean(A7, 1); feats["e_tod_mo"] = np.nanmean(A28, 1)
    # ---- generation mix at issue ----
    g0 = np.clip(S - 2, 0, T - 1); g6 = np.clip(S - 8, 0, T - 1)
    for c in ["wind", "solar", "gas", "coal", "imports", "nuclear", "biomass", "hydro"]:
        i = MIXC.index(c); feats[f"mix_{c}"] = G[g0, i][ps]
    feats["mix_wind_d6"] = (G[g0, MIXC.index("wind")] - G[g6, MIXC.index("wind")])[ps]
    feats["mix_gas_d6"] = (G[g0, MIXC.index("gas")] - G[g6, MIXC.index("gas")])[ps]
    # ---- calendar ----
    tod = TOD[np.clip(w, 0, T - 1)].astype(np.float32); feats["tod"] = tod
    feats["tod_sin"] = np.sin(2 * np.pi * tod / 48); feats["tod_cos"] = np.cos(2 * np.pi * tod / 48)
    feats["dow"] = DOW[np.clip(w, 0, T - 1)].astype(np.float32)
    feats["weekend"] = (feats["dow"] >= 5).astype(np.float32)
    names = list(feats.keys())
    X = np.stack([feats[n].astype(np.float32) for n in names], 1)
    y = np.where((w < T), A[np.clip(w, 0, T - 1)], np.nan).astype(np.float32)
    out = dict(X=X, names=names, s=s, L=L, w=w, f0=f0, y=y)
    if with_seq:
        K = 48
        seq = np.stack([getF(w, L + k) for k in range(1, K + 1)], 1)             # lag 1..48 forecasts of the same target
        out["seq"] = (f0[:, None] - seq).astype(np.float32)                       # revisions relative to current (NaN = missing)
        loc2 = np.stack([getF(w + j, L + j) for j in range(-6, 7)], 1)           # local forecast path at issue
        out["pathloc"] = (loc2 - f0[:, None]).astype(np.float32)
        out["past"] = (V[:, :48][ps] - f0[:, None]).astype(np.float32)           # last 24h visible actuals relative to f0
    return out


def mae(y, f): return float(np.nanmean(np.abs(y - f)))
def rmse(y, f): return float(np.sqrt(np.nanmean((y - f) ** 2)))
