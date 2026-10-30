"""Calibrated generator v2: nonlinear (LightGBM) map from the true series around the target (+lead, hour-of-day) to the forecast, plus resampled residual paths."""
import numpy as np, json, time, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from ext_data import load_region, REGIONS
import ext_sim as X
t00 = time.time(); TSPLIT = 92 * 24; K = 6; NS = 6; DS = [1, 2, 4, 8, 12]; SS = [3, 6, 12, 24, 48]
wl = json.load(open("/home/claude/results/workload.json"))["all_gpu_jobs"]["bins"]; sh = [b["gpu_hour_share"] for b in wl if b["slots"]]; WD = {1: sh[0] + sh[1], 2: sh[2], 4: sh[3], 8: sh[4], 12: sh[5]}
ext = json.load(open("/home/claude/results/external.json")); out = {}
def wfrac(env, ps, pol, S, ref):
    num = den = 0
    for d in DS:
        a = ref[(d, S)]["arr"]; asap = ref[(d, S)]["asap"]; orc = ref[(d, S)]["orc"]; off = X.run_policies(env, a, d, S, ps)[pol]; c = env.realised(a + off, d)
        num += WD[d] * (1 - c.sum() / asap.sum()); den += WD[d] * (1 - orc.sum() / asap.sum())
    return num / den
for r in REGIONS:
    R = load_region(r); env = X.Env(R["A"], R["bstart"]); F = R["F"]; nb, H = F.shape; T = env.T
    idxw = R["bstart"][:, None] + np.arange(H)[None, :]
    Xw = np.stack([np.where((idxw + k >= 0) & (idxw + k < T), env.A[np.clip(idxw + k, 0, T - 1)], np.nan) for k in range(-K, K + 1)], 2)        # (nb,H,13)
    lead = np.broadcast_to(np.arange(H)[None, :], (nb, H)); hod = idxw % 24
    feat = np.concatenate([Xw, lead[..., None], hod[..., None], Xw.mean(2, keepdims=True), Xw.std(2, keepdims=True)], 2)
    train_blocks = R["bstart"] + 96 <= TSPLIT; ok = train_blocks[:, None] & np.isfinite(F) & np.isfinite(Xw).all(2)
    m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, n_jobs=1, verbose=-1).fit(feat[ok], F[ok])
    okall = np.isfinite(Xw).all(2); fitted = np.full((nb, H), np.nan); fitted[okall] = m.predict(feat[okall])
    pool = (F - fitted)[train_blocks]; pool = pool[np.isfinite(pool).all(1)]
    O = X.oracle_matrix(env); orc_ps = X.PathSet(env, O); real_ps = X.PathSet(env, F); arrs = np.arange(TSPLIT, T); ref = {}
    for d in DS:
        for S in SS:
            okv = X.valid_arrivals(env, arrs, d, S, [real_ps, orc_ps]); a = arrs[okv]; ref[(d, S)] = dict(arr=a, asap=env.realised(a, d), orc=env.realised(a + X.run_policies(env, a, d, S, orc_ps)["static"], d))
    res = {pol: {S: [] for S in SS} for pol in ["static", "replan"]}
    for sd in range(NS):
        rng = np.random.default_rng(555 + sd); ps = X.PathSet(env, fitted + pool[rng.integers(0, len(pool), nb)])
        for pol in res:
            for S in SS: res[pol][S].append(float(wfrac(env, ps, pol, S, ref)))
    out[r] = {pol: {str(S): v for S, v in res[pol].items()} for pol in res}
    gaps = [100 * (np.mean(res[pol][S]) - ext["regions"][r]["real"][pol][str(S)]) for pol in res for S in SS]; print(r, "MAD %.1f  replan S12 gap %+.1f" % (np.mean(np.abs(gaps)), 100 * (np.mean(res["replan"][12]) - ext["regions"][r]["real"]["replan"]["12"])), f"{time.time()-t00:.0f}s", flush=True)
allg = [100 * (np.mean(out[r][pol][str(S)]) - ext["regions"][r]["real"][pol][str(S)]) for r in REGIONS for pol in ["static", "replan"] for S in SS]
print("OVERALL MAD %.2f  bias %+.2f   (linear calibrated: %.2f)" % (np.mean(np.abs(allg)), np.mean(allg), np.mean([np.mean([abs(100 * (np.mean(ext['regions'][r]['syn']['berkson'][pol][str(S)]) - ext['regions'][r]['real'][pol][str(S)])) for pol in ['static', 'replan'] for S in SS]) for r in REGIONS])))
json.dump({"syn": out, "mad": float(np.mean(np.abs(allg))), "bias": float(np.mean(allg))}, open("/home/claude/results/external_berkson_gbm.json", "w"))
