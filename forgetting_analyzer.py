
#!/usr/bin/env python3
"""
Forgetting Analyzer for long-run inference logs.

Input CSV (column names are auto-detected; override via CLI if needed):
- turn / t / step / idx
- ce_query / ce / cross_entropy / loss / loss_ce / ce_overall / ce_total
- conf_context (optional)
- zslow_rule_l1 (optional)
- zslow_think_l1 (optional)
- zlang_norm (optional)
- lang / language (optional)
- rule_desc (optional)
- mem_items / memory (optional)

Outputs:
- JSON summary (metrics, thresholds, flags, verdict)
- Optional CSVs for per-language stats and phases
- Optional PNG charts if --plot is provided
"""
import argparse, json, math, statistics
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Utils ----------

def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = [c for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # partial contains case
    for c in cols:
        if c.lower() in candidates:
            return c
    return None

def slope_per_turn(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x_ = x[m]; y_ = y[m]
    if len(x_) < 2:
        return float("nan")
    A = np.vstack([x_, np.ones_like(x_)]).T
    m, b = np.linalg.lstsq(A, y_, rcond=None)[0]
    return float(m)

def rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(y).rolling(window=w, min_periods=max(5, w//6)).mean()
    return s.to_numpy()

def entropy(bits: List[str]) -> float:
    if len(bits) == 0:
        return float("nan")
    from collections import Counter
    c = Counter([str(x).strip() for x in bits])
    total = sum(c.values())
    if total == 0:
        return float("nan")
    return -sum((v/total) * math.log2(v/total) for v in c.values())

# ---------- Core Analyzer ----------

@dataclass
class Thresholds:
    ce_slope: float = 1e-4
    ce_delta: float = 0.20
    conf_slope: float = -1e-4     # flag if <= conf_slope
    conf_ce_corr_min: float = -0.20  # expect negative corr; flag if > this
    zthink_drop_frac: float = 0.15    # 90th pct late < early*(1 - frac) => flag
    zrule_drop_frac: float = 0.15
    entropy_delta: float = 0.40
    lang_ce_delta: float = 0.40

@dataclass
class Config:
    window: int = 120
    early_n: int = 100
    late_n: int = 100

def analyze(df: pd.DataFrame, cfg: Config, th: Thresholds, colmap: Dict[str,str]) -> Dict[str, Any]:
    # Extract columns with safe defaults
    turn = df[colmap["turn"]].to_numpy()
    ce = df[colmap["ce"]].to_numpy()
    conf = df[colmap["conf"]].to_numpy() if colmap["conf"] in df.columns else None
    z_rule = df[colmap["z_rule"]].to_numpy() if colmap["z_rule"] in df.columns else None
    z_think = df[colmap["z_think"]].to_numpy() if colmap["z_think"] in df.columns else None
    z_lang = df[colmap["z_lang"]].to_numpy() if colmap["z_lang"] in df.columns else None
    rule_desc = df[colmap["rule_desc"]] if colmap["rule_desc"] in df.columns else None
    mem = df[colmap["mem"]].to_numpy() if colmap["mem"] in df.columns else None
    lang = df[colmap["lang"]] if colmap["lang"] in df.columns else None

    n = len(df)
    early_n = min(cfg.early_n, n//3 if n>=90 else max(20, n//2))
    late_n = min(cfg.late_n, n//3 if n>=90 else max(20, n//2))

    early = df.iloc[:early_n]
    late  = df.iloc[-late_n:]

    # Basic
    ce_mean = float(np.nanmean(ce))
    ce_med  = float(np.nanmedian(ce))
    ce_last100 = float(np.nanmean(df[colmap["ce"]].tail(100))) if n>=100 else float("nan")

    # Slopes
    ce_slope = slope_per_turn(turn, ce)
    conf_slope = slope_per_turn(turn, conf) if conf is not None else float("nan")
    zrule_slope = slope_per_turn(turn, z_rule) if z_rule is not None else float("nan")
    zthink_slope = slope_per_turn(turn, z_think) if z_think is not None else float("nan")
    zlang_slope = slope_per_turn(turn, z_lang) if z_lang is not None else float("nan")

    # Corr
    corr_ce_conf = float(np.corrcoef(ce, conf)[0,1]) if conf is not None and len(ce)>2 else float("nan")

    # Early/Late comparisons
    ce_early = float(early[colmap["ce"]].mean())
    ce_late  = float(late[colmap["ce"]].mean())
    ce_delta = ce_late - ce_early

    conf_early = float(early[colmap["conf"]].mean()) if conf is not None else float("nan")
    conf_late  = float(late[colmap["conf"]].mean()) if conf is not None else float("nan")

    zthink_early_q = float(np.nanpercentile(early[colmap["z_think"]], 90)) if z_think is not None else float("nan")
    zthink_late_q  = float(np.nanpercentile(late[colmap["z_think"]], 90)) if z_think is not None else float("nan")
    zrule_early_q  = float(np.nanpercentile(early[colmap["z_rule"]], 90)) if z_rule is not None else float("nan")
    zrule_late_q   = float(np.nanpercentile(late[colmap["z_rule"]], 90)) if z_rule is not None else float("nan")

    zthink_drop_frac = ((zthink_early_q - zthink_late_q)/max(1e-8,abs(zthink_early_q))) if (not math.isnan(zthink_early_q) and not math.isnan(zthink_late_q)) else float("nan")
    zrule_drop_frac  = ((zrule_early_q - zrule_late_q)/max(1e-8,abs(zrule_early_q))) if (not math.isnan(zrule_early_q) and not math.isnan(zrule_late_q)) else float("nan")

    rule_entropy_early = entropy(early[colmap["rule_desc"]]) if rule_desc is not None else float("nan")
    rule_entropy_late  = entropy(late[colmap["rule_desc"]])  if rule_desc is not None else float("nan")
    rule_entropy_delta = rule_entropy_early - rule_entropy_late if (not math.isnan(rule_entropy_early) and not math.isnan(rule_entropy_late)) else float("nan")

    # Per-language early/late
    per_lang = {}
    lang_flags = []
    if lang is not None:
        def short_stats(gdf: pd.DataFrame) -> pd.DataFrame:
            return gdf.groupby(colmap["lang"])[colmap["ce"]].agg(["count","mean","median"])
        per_lang["early"] = short_stats(early).reset_index()
        per_lang["late"]  = short_stats(late).reset_index()
        # Flags per language CE drift
        merged = per_lang["early"][ [colmap["lang"], "mean"] ].merge(
                 per_lang["late"][ [colmap["lang"], "mean"] ],
                 on=colmap["lang"], how="outer", suffixes=("_early","_late"))
        merged["delta"] = merged["mean_late"] - merged["mean_early"]
        for _, row in merged.iterrows():
            if not math.isnan(row["delta"]) and row["delta"] > th.lang_ce_delta:
                lang_flags.append((row[colmap["lang"]], float(row["delta"])))

    # Rolling phase skeleton (optional)
    phases = []
    try:
        y_sm = rolling_mean(ce, cfg.window)
        # simple sign-change phase boundaries by slope
        dy = np.diff(y_sm); dx = np.diff(turn)
        slope = np.zeros_like(y_sm); slope[1:] = np.divide(dy, dx, out=np.zeros_like(dy), where=dx!=0)
        mag = np.abs(slope[np.isfinite(slope)])
        thr = np.nanpercentile(mag, 70) if mag.size else 0.0
        cuts=[]; last=-10**9
        for i in range(2, len(slope)):
            if np.isfinite(slope[i]) and np.isfinite(slope[i-1]):
                if np.sign(slope[i]) != np.sign(slope[i-1]) and (abs(slope[i])>thr or abs(slope[i-1])>thr):
                    if i - last > cfg.window*1.3:
                        cuts.append(i); last=i
        bounds = [0] + cuts + [len(df)-1]
        for s,e in zip(bounds[:-1], bounds[1:]):
            seg = df.iloc[s:e+1]
            phases.append({
                "start_turn": int(seg[colmap["turn"]].iloc[0]),
                "end_turn":   int(seg[colmap["turn"]].iloc[-1]),
                "len":        int(len(seg)),
                "ce_mean":    float(seg[colmap["ce"]].mean()),
                "conf_mean":  float(seg[colmap["conf"]].mean()) if conf is not None else float("nan"),
            })
    except Exception as e:
        phases = []

    # ---- Forgetting flags ----
    flags = {}
    reasons = []

    if not math.isnan(ce_slope) and ce_slope >= th.ce_slope:
        flags["ce_slope_drift"] = True; reasons.append(f"CE slope {ce_slope:.3g} ≥ {th.ce_slope}")
    if not math.isnan(ce_delta) and ce_delta >= th.ce_delta:
        flags["ce_early_late_drift"] = True; reasons.append(f"CE late-early {ce_delta:.3g} ≥ {th.ce_delta}")
    if conf is not None:
        if not math.isnan(conf_slope) and conf_slope <= th.conf_slope:
            flags["conf_downtrend"] = True; reasons.append(f"conf slope {conf_slope:.3g} ≤ {th.conf_slope}")
        if not math.isnan(corr_ce_conf) and corr_ce_conf > th.conf_ce_corr_min:
            flags["ce_conf_coupling_broken"] = True; reasons.append(f"corr(CE,conf) {corr_ce_conf:.3g} > {th.conf_ce_corr_min}")
    if z_think is not None and not math.isnan(zthink_drop_frac) and zthink_drop_frac > th.zthink_drop_frac:
        flags["zthink_peak_drop"] = True; reasons.append(f"z_think 90% late drop {zthink_drop_frac:.2%} > {th.zthink_drop_frac:.0%}")
    if z_rule is not None and not math.isnan(zrule_drop_frac) and zrule_drop_frac > th.zrule_drop_frac:
        flags["zrule_peak_drop"] = True; reasons.append(f"z_rule 90% late drop {zrule_drop_frac:.2%} > {th.zrule_drop_frac:.0%}")
    if rule_desc is not None and not math.isnan(rule_entropy_delta) and rule_entropy_delta > th.entropy_delta:
        flags["rule_entropy_collapse"] = True; reasons.append(f"rule entropy delta {rule_entropy_delta:.3g} > {th.entropy_delta}")
    if lang_flags:
        flags["lang_specific_drift"] = True; reasons.append("per-lang late-early deltas "+str(lang_flags))

    verdict = "no_forgetting" if not flags else "possible_forgetting"

    # Compose summary
    summary = {
        "rows": n,
        "columns": list(df.columns),
        "colmap": colmap,
        "config": asdict(cfg),
        "thresholds": asdict(th),
        "metrics": {
            "ce_mean": ce_mean,
            "ce_median": ce_med,
            "ce_last100_mean": ce_last100,
            "ce_slope_per_turn": ce_slope,
            "conf_slope_per_turn": conf_slope,
            "z_rule_slope_per_turn": zrule_slope,
            "z_think_slope_per_turn": zthink_slope,
            "z_lang_slope_per_turn": zlang_slope,
            "corr_ce_conf": corr_ce_conf,
            "ce_early": ce_early,
            "ce_late": ce_late,
            "ce_delta": ce_delta,
            "conf_early": conf_early,
            "conf_late": conf_late,
            "zthink_early_q90": zthink_early_q,
            "zthink_late_q90":  zthink_late_q,
            "zrule_early_q90":  zrule_early_q,
            "zrule_late_q90":   zrule_late_q,
            "zthink_drop_frac": zthink_drop_frac,
            "zrule_drop_frac":  zrule_drop_frac,
            "rule_entropy_early": rule_entropy_early,
            "rule_entropy_late":  rule_entropy_late,
            "rule_entropy_delta": rule_entropy_delta,
        },
        "flags": flags,
        "reasons": reasons,
        "verdict": verdict,
        "per_lang": {
            "early": per_lang["early"].to_dict(orient="list") if "early" in per_lang else None,
            "late":  per_lang["late"].to_dict(orient="list") if "late" in per_lang else None,
        },
        "phases": phases,
    }
    return summary

def autodetect_columns(df: pd.DataFrame) -> Dict[str,str]:
    # Defaults to missing if not found
    m = {}
    m["turn"] = pick_column(df, ["turn","t","step","idx"]) or df.columns[0]
    m["ce"] = pick_column(df, ["ce_query","ce","cross_entropy","loss","loss_ce","ce_overall","ce_total"]) or df.columns[1]
    m["conf"] = pick_column(df, ["conf_context","confidence","conf"] ) or ""
    m["z_rule"] = pick_column(df, ["zslow_rule_l1","z_rule_l1","z_rule","zrule_l1"]) or ""
    m["z_think"] = pick_column(df, ["zslow_think_l1","z_think_l1","z_think","zthink_l1"]) or ""
    m["z_lang"] = pick_column(df, ["zlang_norm","z_lang_norm","z_lang","zlang"]) or ""
    m["lang"] = pick_column(df, ["lang","language"]) or ""
    m["rule_desc"] = pick_column(df, ["rule_desc","rule","desc"]) or ""
    m["mem"] = pick_column(df, ["mem_items","mem","memory","memory_size"]) or ""
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to infer_summary.csv")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--early", type=int, default=100)
    ap.add_argument("--late", type=int, default=100)
    ap.add_argument("--ce_slope", type=float, default=1e-4)
    ap.add_argument("--ce_delta", type=float, default=0.20)
    ap.add_argument("--conf_slope", type=float, default=-1e-4)
    ap.add_argument("--conf_ce_corr_min", type=float, default=-0.20)
    ap.add_argument("--zthink_drop_frac", type=float, default=0.15)
    ap.add_argument("--zrule_drop_frac", type=float, default=0.15)
    ap.add_argument("--entropy_delta", type=float, default=0.40)
    ap.add_argument("--lang_ce_delta", type=float, default=0.40)
    ap.add_argument("--save_json", type=str, default="forgetting_report.json")
    ap.add_argument("--save_csv_prefix", type=str, default="forgetting_out")
    ap.add_argument("--plot", action="store_true", help="Save summary charts as PNGs")
    # Optional manual columns
    ap.add_argument("--col_turn", type=str, default="")
    ap.add_argument("--col_ce", type=str, default="")
    ap.add_argument("--col_conf", type=str, default="")
    ap.add_argument("--col_zrule", type=str, default="")
    ap.add_argument("--col_zthink", type=str, default="")
    ap.add_argument("--col_zlang", type=str, default="")
    ap.add_argument("--col_lang", type=str, default="")
    ap.add_argument("--col_rule_desc", type=str, default="")
    ap.add_argument("--col_mem", type=str, default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    colmap = autodetect_columns(df)
    # Manual overrides
    if args.col_turn: colmap["turn"]=args.col_turn
    if args.col_ce: colmap["ce"]=args.col_ce
    if args.col_conf: colmap["conf"]=args.col_conf
    if args.col_zrule: colmap["z_rule"]=args.col_zrule
    if args.col_zthink: colmap["z_think"]=args.col_zthink
    if args.col_zlang: colmap["z_lang"]=args.col_zlang
    if args.col_lang: colmap["lang"]=args.col_lang
    if args.col_rule_desc: colmap["rule_desc"]=args.col_rule_desc
    if args.col_mem: colmap["mem"]=args.col_mem

    cfg = Config(window=args.window, early_n=args.early, late_n=args.late)
    th = Thresholds(
        ce_slope=args.ce_slope, ce_delta=args.ce_delta,
        conf_slope=args.conf_slope, conf_ce_corr_min=args.conf_ce_corr_min,
        zthink_drop_frac=args.zthink_drop_frac, zrule_drop_frac=args.zrule_drop_frac,
        entropy_delta=args.entropy_delta, lang_ce_delta=args.lang_ce_delta
    )

    summary = analyze(df, cfg, th, colmap)

    # Save JSON
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save per-lang CSVs if present
    if summary["per_lang"]["early"] is not None:
        early_df = pd.DataFrame(summary["per_lang"]["early"])
        late_df  = pd.DataFrame(summary["per_lang"]["late"])
        # The dict has columns as keys → reorient
        early_df = pd.DataFrame(early_df)
        late_df = pd.DataFrame(late_df)
        early_df.to_csv(f"{args.save_csv_prefix}_per_lang_early.csv", index=False)
        late_df.to_csv(f"{args.save_csv_prefix}_per_lang_late.csv", index=False)

    # Save phases CSV
    if summary["phases"]:
        pd.DataFrame(summary["phases"]).to_csv(f"{args.save_csv_prefix}_phases.csv", index=False)

    # Optional plots
    if args.plot:
        turn = df[colmap["turn"]].to_numpy()
        ce = df[colmap["ce"]].to_numpy()
        plt.figure()
        plt.plot(turn, ce)
        plt.xlabel(colmap["turn"]); plt.ylabel(colmap["ce"]); plt.title("CE over turns")
        plt.tight_layout(); plt.savefig(f"{args.save_csv_prefix}_ce.png"); plt.close()

        if colmap["conf"] in df.columns:
            plt.figure()
            plt.plot(turn, df[colmap["conf"]].to_numpy())
            plt.xlabel(colmap["turn"]); plt.ylabel(colmap["conf"]); plt.title("Context confidence")
            plt.tight_layout(); plt.savefig(f"{args.save_csv_prefix}_conf.png"); plt.close()

    # Console summary
    print("[Verdict]", summary["verdict"])
    if summary["reasons"]:
        print("Reasons:"); 
        for r in summary["reasons"]:
            print(" -", r)
    print("Saved:", args.save_json)
    print("Columns:", summary["colmap"])
    print("Metrics.ce_mean:", f"{summary['metrics']['ce_mean']:.4f}")
    print("Metrics.ce_slope_per_turn:", f"{summary['metrics']['ce_slope_per_turn']:.6g}")
    if not math.isnan(summary["metrics"]["corr_ce_conf"]):
        print("corr(CE,conf):", f"{summary['metrics']['corr_ce_conf']:.4f}")
    print()
    return

if __name__ == "__main__":
    main()
