"""Hourly deferral simulator (re-usable for any system with periodic forecast issues) + synthetic forecast generators."""
import numpy as np

class Env:
    def __init__(self, A, bstart, H=96):
        self.A = A.astype(np.float64); self.T = len(A); self.bs = np.asarray(bstart); self.H = H
        self.CA = np.concatenate([[0], np.cumsum(np.nan_to_num(self.A))]); self.CN = np.concatenate([[0], np.cumsum(np.isnan(self.A))])
    def realised(self, tau, d):
        tau = np.asarray(tau); hi = np.clip(tau + d, 0, self.T); lo = np.clip(tau, 0, self.T)
        c = (self.CA[hi] - self.CA[lo]) / d; bad = (self.CN[hi] - self.CN[lo]) > 0
        return np.where(bad | (tau + d > self.T) | (tau < 0), np.nan, c)
    def epoch_view(self, s, j):
        u = s + j; ci = np.searchsorted(self.bs, u, side="right") - 1
        return ci, u - self.bs[np.maximum(ci, 0)], ci >= 0

class PathSet:
    def __init__(self, env, M):
        self.env = env; self.H = env.H; nb = M.shape[0]
        self.cum = np.concatenate([np.zeros((nb, 1)), np.cumsum(np.nan_to_num(M), 1)], 1); self.nan = np.concatenate([np.zeros((nb, 1)), np.cumsum(np.isnan(M), 1)], 1)
    def costs(self, ci, base, omax, d):
        H = self.H; o = np.arange(omax + 1)[None, :]; lo = base[:, None] + o; hi = lo + d; okr = hi <= H
        lo = np.clip(lo, 0, H); hi = np.clip(hi, 0, H); r = np.maximum(ci, 0)[:, None]
        c = (self.cum[r, hi] - self.cum[r, lo]) / d; bad = (self.nan[r, hi] - self.nan[r, lo]) > 0
        return np.where(okr & ~bad, c, np.nan)

def valid_arrivals(env, arr, d, S, pathsets):
    act = env.realised(arr[:, None] + np.arange(S + 1)[None, :], d); ok = np.isfinite(act).all(1)
    for j in range(S + 1):
        ci, base, vv = env.epoch_view(arr, j); ok &= vv
        for ps in pathsets: ok &= np.isfinite(ps.costs(ci, base, S - j, d)).all(1)
    return ok

def run_policies(env, arr, d, S, ps, delta=0.0):
    n = len(arr); ci0, b0, _ = env.epoch_view(arr, 0); static = np.argmin(ps.costs(ci0, b0, S, d), 1)
    started = np.zeros(n, bool); start = np.full(n, S)
    for j in range(S + 1):
        ci, base, _ = env.epoch_view(arr, j); c = ps.costs(ci, base, S - j, d); best = np.argmin(c, 1)
        gain_ok = c[np.arange(n), best] < c[:, 0] * (1 - delta); go = (~gain_ok) | (best == 0) | (j == S); new = go & ~started; start[new] = j; started |= new
    return {"static": static, "replan": start}

def oracle_matrix(env):
    idx = env.bs[:, None] + np.arange(env.H)[None, :]
    return np.where(idx < env.T, env.A[np.clip(idx, 0, env.T - 1)], np.nan)

# ----------------------------------------------------------------------------------------------------------------- generators
def gen_additive(env, kind, mae_l, rng, resid_pool=None, rho=0.98):
    """Classical synthetic forecasts A - noise with per-lead MAE matched to the real forecast. kind: fixed_uniform | iid | ar1 | bootstrap."""
    O = oracle_matrix(env); nb, H = O.shape; sig = (mae_l * 1.2533)[None, :]
    if kind == "fixed_uniform":
        x = 2 * np.nanmean(mae_l) / np.nanmean(env.A); eps = rng.uniform(-x, x, env.T); idx = np.clip(env.bs[:, None] + np.arange(H)[None, :], 0, env.T - 1); noise = eps[idx] * O
    elif kind == "iid": noise = rng.normal(size=(nb, H)) * sig
    elif kind == "ar1":
        z = np.zeros((nb, H)); z[:, 0] = rng.normal(size=nb)
        for l in range(1, H): z[:, l] = rho * z[:, l - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=nb)
        noise = z * sig
    elif kind == "bootstrap":                       # real error paths (A - F) from other issues of the pool
        noise = resid_pool[rng.integers(0, len(resid_pool), nb)]
    return np.where(np.isfinite(O), O - noise, np.nan)

def fit_berkson(env, F, train_mask_blocks, K, buckets, ridge=1.0):
    """Forecast = linear filter of the TRUE series around the target + residual. Per lead-bucket ridge regression of F[k,l] on A[w-K..w+K], tod-free.
    Returns coef dict and residual pool (training issues) with shape [n_train_blocks, H]."""
    nb, H = F.shape; T = env.T; idxw = env.bs[:, None] + np.arange(H)[None, :]
    Xwin = np.stack([np.where((idxw + k >= 0) & (idxw + k < T), env.A[np.clip(idxw + k, 0, T - 1)], np.nan) for k in range(-K, K + 1)], 2)      # (nb, H, 2K+1)
    coefs = {}; fitted = np.full((nb, H), np.nan)
    for b, (lo, hi) in enumerate(buckets):
        sel = np.zeros((nb, H), bool); sel[train_mask_blocks, lo:hi] = True; ok = sel & np.isfinite(F) & np.isfinite(Xwin).all(2)
        Xb = Xwin[ok]; yb = F[ok]; mu = Xb.mean(0); Xc = Xb - mu; ym = yb.mean()
        beta = np.linalg.solve(Xc.T @ Xc + ridge * len(yb) * 1e-3 * np.eye(Xc.shape[1]), Xc.T @ (yb - ym)); coefs[b] = (beta, mu, ym)
        allok = np.isfinite(Xwin[:, lo:hi]).all(2); fitted[:, lo:hi] = np.where(allok, (Xwin[:, lo:hi] - mu) @ beta + ym, np.nan)
    return coefs, fitted, Xwin

def gen_berkson(env, fitted, resid_pool, rng):
    nb = fitted.shape[0]; R = resid_pool[rng.integers(0, len(resid_pool), nb)]; return fitted + R
