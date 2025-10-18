# analyze_suspicious.py
import pandas as pd, numpy as np
from difflib import SequenceMatcher
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--log", default="generation_log.csv")
ap.add_argument("--outdir", default="suspect")
args = ap.parse_args()

df = pd.read_csv(args.log)
for c in ["turn","q_index","lang_id","ce_query","conf_context","zslow_rule_l1","zslow_think_l1","zlang_norm","mem_items"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["lang","rule_desc","input","target","pred"]:
    if c in df.columns: df[c] = df[c].astype(str).fillna("")
df = df.sort_values(["turn","q_index"]).copy()

sim = [SequenceMatcher(None, str(t), str(p)).ratio() for t,p in zip(df.get("target",""), df.get("pred",""))]
df["sim_ratio"] = sim

# thresholds
ce_p90  = np.nanpercentile(df["ce_query"], 90)
ce_p10  = np.nanpercentile(df["ce_query"], 10)
conf_p90= np.nanpercentile(df["conf_context"], 90)
conf_p10= np.nanpercentile(df["conf_context"], 10)
sim_p10 = np.nanpercentile(df["sim_ratio"], 10)

overconf = df[(df["ce_query"]>=ce_p90) & (df["conf_context"]>=conf_p90)].copy()
underconf= df[(df["ce_query"]<=ce_p10) & (df["conf_context"]<=conf_p10)].copy()
semdrift = df[(df["sim_ratio"]<=sim_p10) & (df["conf_context"]>=conf_p90)].copy()

ts = df.groupby("turn").agg(ce=("ce_query","mean"), conf=("conf_context","mean"), mem=("mem_items","mean")).reset_index()
win = max(21, ((len(ts)//20)*2 + 1))
ts["ce_roll"] = ts["ce"].rolling(window=win, min_periods=5, center=True).mean()
ts["conf_roll"] = ts["conf"].rolling(window=win, min_periods=5, center=True).mean()

def z(s): 
    mu, sd = np.nanmean(s), np.nanstd(s); 
    return (s-mu)/(sd if sd>0 else 1e-8)
ts["z_ce"] = z(ts["ce_roll"]); ts["z_conf"] = z(ts["conf_roll"])
spikes = ts[ts["z_ce"]>2.0].copy()
dips   = ts[ts["z_conf"]<-2.0].copy()

out = Path(args.outdir); out.mkdir(exist_ok=True)
overconf.to_csv(out/"overconfidence_samples.csv", index=False)
underconf.to_csv(out/"underconfidence_correct.csv", index=False)
semdrift.to_csv(out/"semantic_drift_samples.csv", index=False)
spikes.to_csv(out/"ce_spikes.csv", index=False)
dips.to_csv(out/"conf_dips.csv", index=False)

# quick counts
print({
  "overconfidence": len(overconf),
  "underconfidence": len(underconf),
  "semantic_drift": len(semdrift),
  "ce_spikes": len(spikes),
  "conf_dips": len(dips),
})
