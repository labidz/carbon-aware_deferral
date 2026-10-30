# Code and result files

Everything was run on a single-core, 4 GB, GPU-less machine (Python 3.12; LightGBM 4.7, TensorFlow-CPU 2.21, pandas, numpy, statsmodels,
matplotlib; Node 22 + docx 9 for the manuscript). Paths are hard-coded to /home/claude/... - edit before reuse.
Prerequisite clones (no token needed):
  git clone https://github.com/nmpowell/carbon-intensity-forecast-tracking   # GB forecast vintages (licence: verify)
  git clone https://github.com/InternLM/AcmeTrace                            # GPU job traces (CC BY 4.0)
  git clone https://github.com/carbonfirst/CarbonCast                        # 13-grid external data (Apache-2.0)

## Run order
1. s1_build_dataset.py            curate UK-CIV (tensor, parquet tables, QC, splits)
2. ukciv.py ; t_leak.py           69 leakage-free features; leakage audit with positive control
3. s4a_global.py ; s4b_walkforward.py <variant>   global samples; walk-forward LightGBM (main + ablations)
   s4c_causal.py ; s4d_standalone.py              causal hyper-parameter selection; stand-alone model without the operator forecast
4. s8a_characterize.py            error structure, revision-efficiency and persistence tests (day-clustered)
5. s8b_prob_eval.py, conformal.py, gate.py ; s8e_aci.py   rolling/static conformal calibration; online gate; adaptive conformal inference
6. s8c_forecast_eval.py ; s12d_forecast_extras.py          accuracy tables (paired day-block bootstrap), naive / climatology / stand-alone baselines
7. s7_dl.py A|B (env SEED, DROPOUT, L2, REL, TAG) ; s8d_dl_eval.py ; s8f_dl_ensemble_eval.py   VintageNet and regularised ensemble
8. s5a_workload.py                GPU-hour weights from Acme traces
9. sim.py ; s9_decisions.py ; s9b_lsn.py ; s10_analysis.py ; s10b_lsn_eval.py   deferral simulator, policies, learned policy, statistics
10. s12_extras.py ; s12c_more_decisions.py ; s12b_analysis.py   forecast-free baselines, multi-seed synthetic generators (+ calibrated), contrasts
11. ext_data.py ; ext_sim.py ; s13_external.py ; s13c_external_ml.py ; s13d_berkson_gbm.py   13-grid validation (revisions, transfer, generators)
12. s14_fleet.py                  capacity-constrained fleet replay with real jobs
13. s11_overhead.py ; fig_*.py ; paper_build/*.js   overhead audit, figures, manuscript build

Experiments are resumable (per-block / per-grid / per-seed checkpoints); chain*.sh are idempotent launchers.
results/ holds the small JSON/CSV files every number in the paper is read from.
