import numpy as np, ukciv as U, time, warnings; warnings.filterwarnings("ignore")
t = time.time()
caps = U.all_captures(); caps = caps[U.complete_view(caps)]
print("complete captures:", len(caps))
Xs, ss, LL, ff, yy = [], [], [], [], []
for i in range(0, len(caps), 1500):
    d = U.make_samples(caps[i:i+1500])
    keep = np.isfinite(d["f0"])                      # keep samples w/o known target too (y NaN) - needed for forecasting views
    Xs.append(d["X"][keep]); ss.append(d["s"][keep]); LL.append(d["L"][keep]); ff.append(d["f0"][keep]); yy.append(d["y"][keep])
    names = d["names"]; print(i, f"{time.time()-t:.0f}s", flush=True)
X = np.concatenate(Xs); np.save("/home/claude/data/g_X.npy", X)
np.savez("/home/claude/data/g_meta.npz", s=np.concatenate(ss).astype(np.int32), L=np.concatenate(LL).astype(np.int16), f0=np.concatenate(ff), y=np.concatenate(yy), names=np.array(names))
print("global samples", X.shape, f"{X.nbytes/1e6:.0f} MB")
