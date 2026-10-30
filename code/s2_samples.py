import numpy as np, ukciv as U, time, warnings; warnings.filterwarnings("ignore")
t=time.time()
cfg = {"train": 2, "val": 2, "cal": 1, "testA": 1, "testB": 1}
for name, stride in cfg.items():
    S = U.split_slots(name, stride=stride)
    d = U.make_samples(S, with_seq=True)
    keep = np.isfinite(d["f0"]) & np.isfinite(d["y"])
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v)==len(keep) else v) for k, v in d.items()}
    out["names"] = np.array(d["names"])
    np.savez_compressed(f"/home/claude/data/samples_{name}.npz", **out)
    print(name, "captures", len(S), "samples", keep.sum(), "X", out["X"].shape, f"{time.time()-t:.0f}s", flush=True)
