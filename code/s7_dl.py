"""
VintageNet: deep quantile post-processor for operational carbon-intensity forecasts.
usage: python s7_dl.py MODE     MODE in {A, B}
  A: train 2023-03-13..2023-10-25, early-stop on 2023-11 -> predict 2023-12 .. 2024-02 (blocks 6,7)
  B: train 2023-03-13..2024-01-05 (val = 2024-01-08..2024-02-02 excluded from train), -> predict 2026 blocks (8..11)
Outputs: /home/claude/data/dl_{MODE}.npz  (rows aligned with g_meta; residual quantiles q10,q50,q90) + timing json
"""
import os, sys, time, json, warnings; warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
tf.keras.utils.set_random_seed(int(os.environ.get("SEED", "7")))
import ukciv as U
mode = sys.argv[1]
SEED = int(os.environ.get("SEED", "7")); DROP = float(os.environ.get("DROPOUT", "0.1")); L2W = float(os.environ.get("L2", "0")); REL = int(os.environ.get("REL", "0")); TAG = os.environ.get("TAG", "")
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True)
gs, gL, gf0, gy = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]
gkey = gs * 96 + gL
tis = U.TS[gs]; fi = np.isfinite(gy)
W = np.load("/home/claude/data/wf_main.npz"); blk = W["blk"]
T = pd.Timestamp
if mode == "A":
    tr_m = (tis >= T("2023-03-13")) & (tis < T("2023-10-27")) & fi & (gs % 2 == 0)
    va_m = (tis >= T("2023-11-01")) & (tis < T("2023-11-29")) & fi & (gs % 2 == 0)
    te_m = np.isin(blk, [6, 7]) & fi
else:
    tr_m = (tis >= T("2023-03-13")) & (tis < T("2024-01-05")) & fi & (gs % 2 == 0)
    va_m = (tis >= T("2024-01-08")) & (tis < T("2024-02-02")) & fi & (gs % 2 == 0)
    te_m = np.isin(blk, [8, 9, 10, 11]) & fi
print(mode, "train", tr_m.sum(), "val", va_m.sum(), "test", te_m.sum(), flush=True)


def build(mask, chunk=1500):
    """Re-create full feature tensors (tabular + sequences) for the masked global rows."""
    rows = np.where(mask)[0]; S = np.unique(gs[rows]); out = {k: [] for k in ["X", "seq", "pathloc", "past", "key", "f0", "y"]}
    for i in range(0, len(S), chunk):
        d = U.make_samples(S[i:i + chunk], with_seq=True)
        k = d["s"] * 96 + d["L"]; sel = np.isin(k, gkey[rows])
        for name in ["X", "seq", "pathloc", "past"]: out[name].append(d[name][sel])
        out["key"].append(k[sel]); out["f0"].append(d["f0"][sel]); out["y"].append(d["y"][sel])
    return {k: np.concatenate(v) for k, v in out.items()}


t0 = time.time()
names = list(M["names"])
TR, VA, TE = build(tr_m), build(va_m), build(te_m)
print("features built", f"{time.time()-t0:.0f}s", TR["X"].shape, flush=True)
# ---------------- preprocessing ----------------
ABS = [f"lag{k}" for k in [1,2,3,4,6,8,12,16,24,32,48]] + ["path_mean","path_min","path_max","f_prev","f_next","f_loc7","a_last","a_mean4","a_mean12","a_mean48","a_yday","a_2d","a_lw","vint_mean"]
abs_idx = [names.index(n) for n in ABS]
if REL:
    for D_ in (TR, VA, TE): D_["X"][:, abs_idx] = D_["X"][:, abs_idx] - D_["f0"][:, None]
mu = np.nanmean(TR["X"], 0); sd = np.nanstd(TR["X"], 0) + 1e-6
nan_cols = np.where(np.isnan(TR["X"]).mean(0) > 0.01)[0]
def prep(D):
    X = (D["X"] - mu) / sd; ind = np.isnan(X[:, nan_cols]).astype(np.float32); X = np.nan_to_num(X, nan=0.0).clip(-6, 6)
    tab = np.concatenate([X, ind, np.nan_to_num(D["pathloc"] / 30.0, nan=0.0).clip(-6, 6)], 1).astype(np.float32)
    seq = np.stack([np.nan_to_num(D["seq"] / 30.0, nan=0.0).clip(-6, 6), np.isnan(D["seq"]).astype(np.float32)], 2).astype(np.float32)
    past = np.stack([np.nan_to_num(D["past"] / 50.0, nan=0.0).clip(-6, 6), np.isnan(D["past"]).astype(np.float32)], 2).astype(np.float32)
    y = ((D["y"] - D["f0"]) / 30.0).astype(np.float32)
    return [tab, seq, past], y
Xtr, ytr = prep(TR); Xva, yva = prep(VA); Xte, yte = prep(TE)
for D in (TR, VA, TE):
    for k in ("X", "seq", "past", "pathloc"): D.pop(k, None)
import gc; gc.collect()
QS = tf.constant([0.1, 0.5, 0.9])
# ---------------- model ----------------
def conv_branch(inp, f=16, name=""):
    x = tf.keras.layers.Conv1D(f, 5, padding="same", activation="relu", name=name + "c1")(inp)
    x = tf.keras.layers.Conv1D(f, 5, strides=2, padding="same", activation="relu", name=name + "c2")(x)
    x = tf.keras.layers.Conv1D(f, 3, strides=2, padding="same", activation="relu", name=name + "c3")(x)
    return tf.keras.layers.Dense(32, activation="relu", name=name + "d")(tf.keras.layers.Flatten()(x))
i_tab = tf.keras.Input((Xtr[0].shape[1],), name="tab"); i_seq = tf.keras.Input((48, 2), name="vintage_seq"); i_past = tf.keras.Input((48, 2), name="recent_actuals")
reg = tf.keras.regularizers.l2(L2W) if L2W > 0 else None
b_tab = tf.keras.layers.Dropout(DROP)(tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(i_tab))
h = tf.keras.layers.Concatenate()([b_tab, conv_branch(i_seq, name="seq_"), conv_branch(i_past, name="past_")])
h = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(h); h = tf.keras.layers.Dropout(DROP)(h); h = tf.keras.layers.Dense(64, activation="relu")(h)
raw = tf.keras.layers.Dense(3, name="raw")(h)
def noncross(r):
    med = r[:, 1:2]; up = tf.nn.softplus(r[:, 2:3]) ; lo = tf.nn.softplus(r[:, 0:1])
    return tf.concat([med - lo, med, med + up], 1)
out = tf.keras.layers.Lambda(noncross, name="quantiles")(raw)
model = tf.keras.Model([i_tab, i_seq, i_past], out)
def pinball(y, q):
    e = y[:, None] - q; return tf.reduce_mean(tf.maximum(QS * e, (QS - 1.0) * e))
EP = 14; BS = 2048; steps = int(np.ceil(len(ytr) / BS)) * EP
lr = tf.keras.optimizers.schedules.CosineDecay(2e-3, steps, alpha=0.05)
model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), loss=pinball)
print("params", model.count_params(), flush=True)
best, best_w, bad = 1e9, None, 0
for ep in range(EP):
    t1 = time.time(); perm = np.random.default_rng(ep).permutation(len(ytr)); tl = []
    for i in range(0, len(perm), BS):
        j = perm[i:i + BS]; tl.append(model.train_on_batch([x[j] for x in Xtr], ytr[j]))
    vq = model.predict(Xva, batch_size=8192, verbose=0); vl = float(np.mean(np.abs(yva - vq[:, 1]))) * 30
    print(f"ep {ep} train pinball {np.mean(tl):.4f}  val MAE(q50) {vl:.3f}  {time.time()-t1:.0f}s", flush=True)
    if vl < best - 1e-3: best, best_w, bad = vl, model.get_weights(), 0
    else:
        bad += 1
        if bad >= 3: break
model.set_weights(best_w)
q = model.predict(Xte, batch_size=8192, verbose=0) * 30
t2 = time.time(); _ = model.predict([x[:20000] for x in Xte], batch_size=8192, verbose=0); inf_s = (time.time() - t2) / 20000
res = np.full((len(gs), 3), np.nan, np.float32); idx = np.searchsorted(gkey, TE["key"]); res[idx] = q
np.savez(f"/home/claude/data/dl_{mode}{TAG}.npz", q10=res[:, 0], q50=res[:, 1], q90=res[:, 2])
json.dump({"mode": mode, "params": int(model.count_params()), "val_mae_best": best, "test_mae_q50": float(np.mean(np.abs(yte * 30 - q[:, 1]))), "test_mae_neso": float(np.mean(np.abs(yte * 30))),
           "inference_seconds_per_sample": inf_s, "train_seconds": time.time() - t0}, open(f"/home/claude/results/dl_{mode}{TAG}.json", "w"), indent=1)
print("DONE", mode, "test MAE q50 %.3f vs NESO %.3f" % (np.mean(np.abs(yte * 30 - q[:, 1])), np.mean(np.abs(yte * 30))), flush=True)
