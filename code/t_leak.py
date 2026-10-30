import numpy as np, ukciv as U, time
rng = np.random.default_rng(0)
caps = np.concatenate([U.split_slots("train")[::500], U.split_slots("testA")[::300], U.split_slots("testB")[::200]])
print("testing", len(caps), "issue slots")
Fo, Avo, Ao, Go = U.F.copy(), U.Av.copy(), U.A.copy(), U.G.copy()
W = np.arange(U.T)[:, None]; LD = (np.arange(U.NL) + U.LMIN)[None, :]; CAP = W - LD       # capture slot of each cell
bad = 0; t = time.time()
for s in caps:
    S1 = np.array([s]); base = U.make_samples(S1, with_seq=True)
    U.F[:] = np.where(CAP > s, np.nan, Fo); U.Av[:] = np.where(CAP > s, np.nan, Avo)
    U.A[:] = Ao; U.A[s - 48:] = np.nan                 # windows newer than s-49 unknown
    U.G[:] = Go; U.G[s - 1:] = np.nan                  # mix of windows > s-2 unknown
    # keep y out of the comparison (target is allowed to use the future); compare features only
    msk = U.make_samples(S1, with_seq=True)
    U.F[:], U.Av[:], U.A[:], U.G[:] = Fo, Avo, Ao, Go
    same = np.allclose(np.nan_to_num(base["X"], nan=-9e9), np.nan_to_num(msk["X"], nan=-9e9))
    same_seq = all(np.allclose(np.nan_to_num(base[k], nan=-9e9), np.nan_to_num(msk[k], nan=-9e9)) for k in ["seq","pathloc","past"])
    if not (same and same_seq):
        bad += 1; d = np.where(~np.isclose(np.nan_to_num(base["X"], nan=-9e9), np.nan_to_num(msk["X"], nan=-9e9)))[1]
        print("LEAK at slot", s, "features:", sorted(set(np.array(base['names'])[d])))
print("done; slots with leakage:", bad, f"({time.time()-t:.0f}s)")
