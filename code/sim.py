"""
sim.py - trace-driven simulation of temporal workload shifting on real forecast vintages.

A deferrable, non-interruptible job of d slots (30 min each) arrives at capture slot s with slack S slots:
it may start at any slot tau in [s, s+S]. At each decision epoch u in [s, s+S] the scheduler sees the most recent
available forecast view (capture c <= u; stale when the archive has a gap) and decides start / wait.
Emissions of a started job = mean *actual* carbon intensity over [tau, tau+d-1]  (constant power draw), in gCO2/kWh.
"""
import numpy as np, ukciv as U

H = U.H
CAPS_ALL = U.all_captures()
CAPS = CAPS_ALL[U.complete_view(CAPS_ALL)]
NC = len(CAPS)
_A = U.A.astype(np.float64)
_CA = np.concatenate([[0], np.cumsum(np.nan_to_num(_A))]); _CN = np.concatenate([[0], np.cumsum(np.isnan(_A))])


def to_matrix(s, L, v):
    """Scatter per-sample values v(s,L) to a dense [NC,H] matrix (NaN where absent)."""
    M = np.full((NC, H), np.nan, np.float32)
    ci = np.searchsorted(CAPS, s); ok = (ci < NC) & (CAPS[np.minimum(ci, NC - 1)] == s)
    M[ci[ok], L[ok]] = v[ok]
    return M


def actual_matrix():
    """Oracle path: actual intensity for leads 0..95 of every capture."""
    idx = CAPS[:, None] + np.arange(H)[None, :]
    return np.where(idx < U.T, U.A[np.clip(idx, 0, U.T - 1)], np.nan).astype(np.float32)


class PathSet:
    """Cumulative sums of a [NC,H] path matrix for O(1) window means."""
    def __init__(self, M):
        self.M = M
        self.cum = np.concatenate([np.zeros((NC, 1)), np.cumsum(np.nan_to_num(M), 1)], 1)
        self.nan = np.concatenate([np.zeros((NC, 1)), np.cumsum(np.isnan(M), 1)], 1)

    def costs(self, ci, base, omax, d):
        """mean path value over [base+o, base+o+d) for o=0..omax, per arrival (ci index of capture, base offset)."""
        o = np.arange(omax + 1)[None, :]
        lo = base[:, None] + o; hi = lo + d
        okr = hi <= H
        lo = np.clip(lo, 0, H); hi = np.clip(hi, 0, H)
        r = ci[:, None]
        c = (self.cum[r, hi] - self.cum[r, lo]) / d
        bad = (self.nan[r, hi] - self.nan[r, lo]) > 0
        return np.where(okr & ~bad, c, np.nan)


def realised(tau, d):
    """Actual mean intensity of a job starting at slot(s) tau."""
    tau = np.asarray(tau)
    hi = np.clip(tau + d, 0, U.T); lo = np.clip(tau, 0, U.T)
    c = (_CA[hi] - _CA[lo]) / d
    bad = (_CN[hi] - _CN[lo]) > 0
    return np.where(bad | (tau + d > U.T), np.nan, c)


def epoch_view(s_arr, j):
    """For arrivals at slots s_arr and epoch offset j: capture index (latest <= u) and base offset (staleness)."""
    u = s_arr + j
    ci = np.searchsorted(CAPS, u, side="right") - 1
    return ci, u - CAPS[ci]


def valid_arrivals(a_idx, d, S, pathsets):
    """Arrivals for which the job completes inside known actuals and every epoch view covers all candidates."""
    s = CAPS[a_idx]
    act = realised(s[:, None] + np.arange(S + 1)[None, :], d)
    ok = np.isfinite(act).all(1)
    for j in range(S + 1):
        ci, base = epoch_view(s, j)
        for ps in pathsets:
            ok &= np.isfinite(ps.costs(ci, base, S - j, d)).all(1)
    return ok


def run_policies(a_idx, d, S, ps, delta=0.0, risk=None):
    """Static-commit and rolling-replan policies on path-set ps. Returns dict of start offsets (relative to arrival)."""
    n = len(a_idx); s = CAPS[a_idx]
    ci0, b0 = epoch_view(s, 0)
    c0 = ps.costs(ci0, b0, S, d)
    static = np.argmin(c0, 1)                                    # first minimum (earliest on ties)
    started = np.zeros(n, bool); start = np.full(n, S)
    for j in range(S + 1):
        ci, base = epoch_view(s, j)
        c = ps.costs(ci, base, S - j, d)
        best = np.argmin(c, 1); gain_ok = c[np.arange(n), best] < c[:, 0] * (1 - delta)
        go = (~gain_ok) | (best == 0) | (j == S)
        new = go & ~started
        start[new] = j; started |= new
    return {"static": static, "replan": start}


def cost_of(a_idx, d, offs):
    return realised(CAPS[a_idx] + offs, d)
