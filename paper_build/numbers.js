const fs = require("fs");
const J = (p) => { try { return JSON.parse(fs.readFileSync("/home/claude/results/" + p)); } catch (e) { return null; } };
function csv(p) {
  const lines = fs.readFileSync("/home/claude/results/" + p, "utf8").trim().split("\n"); const h = lines[0].split(",");
  return lines.slice(1).map((l) => { const c = l.split(","); const o = {}; h.forEach((k, i) => { o[k] = (["fc", "pol", "period"].includes(k) || isNaN(Number(c[i])) || c[i] === "") ? c[i] : Number(c[i]); }); return o; });
}
const R = {
  qc: J("qc.json"), ch: J("characterize.json"), fe: J("forecast_eval.json"), pb: J("prob_main.json"), dle: J("dl_eval.json"), dlA: J("dl_A.json"), dlB: J("dl_B.json"),
  dt: J("decision_tables.json"), lsn: J("lsn_eval.json"), lsnm: J("lsn_meta.json"), oh: J("overhead.json"), wl: J("workload.json"), dsum: csv("decisions_summary.csv"),
  tune: J("tuning.json"), nt: J("new_tables.json"), fx: J("forecast_extras.json"), aci: J("aci.json"), ext: J("external.json"), extml: J("external_ml.json"), causal: J("causal_choice.json"), dlens: J("dl_ensemble.json"), fleet: J("fleet.json"), xgbm: J("external_berkson_gbm.json"),
};
// derived helpers -----------------------------------------------------------------------------------
R.W = (fc, pol, S) => R.dt.weighted[`${fc}|${pol}|S${S}`];
R.C = (k) => R.dt.contrasts[k];
R.ref = (fc, pol, period = "all", d = 8, S = 24) => R.dsum.find((r) => r.fc === fc && r.pol === pol && r.period === period && r.d === d && r.S === S);
R.syn = (gen, pol, S) => R.dt.synthetic_gap[`${gen}|${pol}|S${S}`];
R.synDiffs = () => {  // synthetic minus real oracle-fraction, in pp, over generators/policies/slacks
  const out = []; for (const g of ["syn:fixed-uniform", "syn:iid-gauss", "syn:ar1-gauss", "syn:real-error-bootstrap"]) for (const pol of ["static", "replan"]) for (const S of [2, 6, 12, 24, 48]) out.push({ g, pol, S, d: 100 * (R.syn(g, pol, S)[0] - R.syn("NESO", pol, S)[0]) });
  return out;
};
R.errAt = (p, h) => R.ch.error_by_lead.find((r) => r.period === p && Math.abs(r.lead_h - h) < 1e-6);
R.extPool = (nm) => R.extml.accuracy[nm]._pooled;
R.extRegs = () => Object.keys(R.extml.decisions);
R.extCapDelta = (nm, pol, S) => { const rs = R.extRegs(); const d = rs.map((r) => 100 * (R.extml.decisions[r][nm][pol][S] - R.extml.decisions[r].raw[pol][S])); const m = d.reduce((a, b) => a + b, 0) / d.length; let seed = 12345; const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; }; const bs = []; for (let b = 0; b < 4000; b++) { let t = 0; for (let i = 0; i < d.length; i++) t += d[Math.floor(rnd() * d.length)]; bs.push(t / d.length); } bs.sort((a, b) => a - b); return { mean: m, lo: bs[Math.floor(0.025 * bs.length)], hi: bs[Math.floor(0.975 * bs.length)], improved: d.filter((x) => x > 0).length, n: d.length }; };
R.extMaeWins = (nm) => R.extRegs().filter((r) => R.extml.accuracy[nm][r].corrected < R.extml.accuracy[nm][r].raw).length;
R.xgbmStats = () => { const regs = Object.keys(R.xgbm.syn); const per = regs.map((r) => { const v = []; for (const pol of ["static", "replan"]) for (const S of ["3", "6", "12", "24", "48"]) v.push(100 * (R.xgbm.syn[r][pol][S].reduce((a, b) => a + b, 0) / R.xgbm.syn[r][pol][S].length - R.ext.regions[r].real[pol][S])); return v; }); const mad = per.map((v) => v.reduce((a, b) => a + Math.abs(b), 0) / v.length); const sorted = [...mad].sort((a, b) => a - b); return { mad: R.xgbm.mad, bias: R.xgbm.bias, median: sorted[Math.floor(sorted.length / 2)], over10: mad.filter((x) => x > 10).length, n: regs.length }; };
module.exports = R;
