import pandas as pd, numpy as np, json
rows = []
for name in ["seren", "kalos"]:
    df = pd.read_csv(f"/home/claude/repos/AcmeTrace/data/job_trace/trace_{name}.csv", usecols=["gpu_num", "type", "state", "duration", "gpu_time", "submit_time"])
    df["cluster"] = name; rows.append(df)
J = pd.concat(rows, ignore_index=True)
print("jobs total", len(J), " gpu jobs", (J.gpu_num > 0).sum())
G = J[(J.gpu_num > 0) & (J.duration > 0)].copy()
G["gpu_hours"] = G.gpu_num * G.duration / 3600.0
edges = [0, 0.5, 1, 2, 4, 8, 12, 1e9]; slots = [1, 2, 4, 8, 16, 24, None]
G["bin"] = pd.cut(G.duration / 3600.0, edges, labels=range(7))
out = {}
for label, sub in [("all_gpu_jobs", G), ("completed_only", G[G.state == "COMPLETED"])]:
    tot = sub.gpu_hours.sum()
    w = sub.groupby("bin", observed=False).gpu_hours.sum() / tot
    n = sub.groupby("bin", observed=False).size()
    out[label] = {"gpu_hours_total": float(tot), "n_jobs": int(len(sub)),
                  "bins": [{"max_hours": edges[i+1] if edges[i+1] < 1e8 else None, "slots": slots[i], "gpu_hour_share": float(w.iloc[i]), "job_share": float(n.iloc[i]/len(sub))} for i in range(7)]}
    print(label, "GPU-h total %.3g" % tot); [print("  <=%sh  slots=%s  GPUh share %.3f  job share %.3f" % (b["max_hours"], b["slots"], b["gpu_hour_share"], b["job_share"])) for b in out[label]["bins"]]
by_type = G.groupby("type").agg(jobs=("gpu_hours", "size"), gpu_hours=("gpu_hours", "sum"), med_dur_s=("duration", "median"), mean_gpus=("gpu_num", "mean")).round(1)
print(by_type)
out["by_type"] = by_type.reset_index().to_dict("records")
# state shares by GPU hours
st = (G.groupby("state").gpu_hours.sum() / G.gpu_hours.sum()).round(3).to_dict(); out["state_gpu_hour_share"] = st; print(st)
# diurnal arrival pattern (submit hour, local time as logged)
G["h"] = pd.to_datetime(G.submit_time, utc=True, format="mixed").dt.hour
out["submit_hour_share"] = (G.groupby("h").size() / len(G)).round(4).to_dict()
json.dump(out, open("/home/claude/results/workload.json", "w"), indent=1, default=str)
