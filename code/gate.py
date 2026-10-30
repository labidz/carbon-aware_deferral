"""Online trust gate: blend weight lambda in {0,.25,.5,.75,1} per lead bucket, re-estimated each capture from settled recent targets."""
import numpy as np, ukciv as U
LB = np.array([4, 12, 24, 48]); NB = 5; LAMS = np.array([0, .25, .5, .75, 1.0])
def lead_bucket(L): return np.digitize(L, LB)
def gate_lambdas(s, L, y, f0, pm, blk, K=336, min_cnt=1500, default=0.5):
    """returns lam_row aligned with rows (NaN-safe). Rows with blk<0 get default."""
    m = (blk >= 0) & np.isfinite(y)
    w = s + L; b = lead_bucket(L); res = y - f0
    SUM = np.zeros((U.T, NB, len(LAMS))); CNT = np.zeros((U.T, NB))
    for li, lam in enumerate(LAMS):
        e = np.abs(res[m] - lam * pm[m])
        np.add.at(SUM[:, :, li], (w[m], b[m]), e)
    np.add.at(CNT, (w[m], b[m]), 1.0)
    cS = np.cumsum(SUM, 0); cC = np.cumsum(CNT, 0)
    caps = np.unique(s); be = np.clip(caps - 49, 0, U.T - 1); bs = np.clip(be - K, 0, U.T - 1)
    sums = cS[be] - cS[bs]; cnt = cC[be] - cC[bs]                              # (nc, NB, nl) , (nc, NB)
    lam_c = np.where(cnt[:, :, None] >= min_cnt, LAMS[np.argmin(sums, 2)][:, :, None] * np.ones(len(LAMS))[None, None, :], default)[:, :, 0]
    ci = np.searchsorted(caps, s)
    return lam_c[ci, b]
