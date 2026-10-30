"""Static and rolling split-conformal calibration (CQR; Romano et al. 2019) for lead-bucketed quantile intervals."""
import numpy as np, ukciv as U
LB = np.array([4, 12, 24, 48]); NB = 5
def bucket(L): return np.digitize(L, LB)
def scores(y, lo, hi): return np.maximum(lo - y, y - hi)
def rolling_cqr(s, L, y, lo, hi, valid, alpha=0.2, K=672, min_n=3000, update_every=2):
    """returns adjustment Q (per row) so that interval = [lo-Q, hi+Q]; NaN where not enough settled history."""
    w = s + L; b = bucket(L); sc = np.where(valid, scores(y, lo, hi), np.nan)
    ok = np.isfinite(sc); order = np.argsort(w[ok], kind="stable"); W = w[ok][order]; B = b[ok][order]; SC = sc[ok][order]
    per_b = [(W[B == k], SC[B == k]) for k in range(NB)]
    caps = np.unique(s); Qc = np.full((len(caps), NB), np.nan)
    last = None
    for i, c in enumerate(caps):
        if update_every > 1 and last is not None and (c - last) < update_every and (c // update_every) == (last // update_every):
            Qc[i] = Qc[i - 1]; continue
        hi_w, lo_w = c - 49, c - 49 - K; last = c
        for k in range(NB):
            Wk, Sk = per_b[k]; i0, i1 = np.searchsorted(Wk, [lo_w + 1, hi_w + 1]); n = i1 - i0
            if n >= min_n:
                lev = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n); Qc[i, k] = np.quantile(Sk[i0:i1], lev)
    return Qc[np.searchsorted(caps, s), b]
def static_cqr(L, y, lo, hi, calib_mask, alpha=0.2):
    b = bucket(L); sc = scores(y, lo, hi); Q = np.zeros(NB)
    for k in range(NB):
        v = sc[calib_mask & (b == k) & np.isfinite(sc)]; n = len(v); Q[k] = np.quantile(v, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    return Q[b]
def interval_metrics(y, lo, hi, alpha=0.2):
    cov = (y >= lo) & (y <= hi); width = hi - lo
    iscore = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return dict(coverage=float(cov.mean()), width=float(width.mean()), interval_score=float(iscore.mean()))
