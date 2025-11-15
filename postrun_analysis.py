# postrun_analysis.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

W = pd.read_csv("writes.csv")
E = pd.read_csv("entropy.csv")

for df in (W,E):
    if "step" not in df.columns: df.rename(columns={df.columns[0]:"step"}, inplace=True)

df = pd.merge_asof(E.sort_values("step"), W.sort_values("step"), on="step")

# 구간 나누기
cut = 1000
early = df[df.step <= cut].copy()
late  = df[df.step >  cut].copy()

def z_anoms(x, win=256):
    x = x.astype(float)
    roll_mu = x.rolling(win, min_periods=win//2).mean()
    roll_sd = x.rolling(win, min_periods=win//2).std().replace(0, np.nan)
    z = (x - roll_mu)/roll_sd
    return z, (z.abs() >= 3)

# 1) 기본 통계
out = {}
out["model_entropy_mu"]  = df["model_entropy"].mean()
out["model_entropy_std"] = df["model_entropy"].std()
out["write_rate_per100"] = (df.get("write_decision",0).mean()*100)

# 2) 초반 스파이크 감소 확인
p2p_early = early["model_entropy"].max() - early["model_entropy"].min()
out["early_p2p"] = p2p_early

# 3) 상관/교차상관
def xcorr(a, b, max_lag=200):
    a = pd.Series(a).astype(float); b = pd.Series(b).astype(float)
    m = pd.concat([a,b], axis=1).dropna()
    a = (m.iloc[:,0]-m.iloc[:,0].mean())/m.iloc[:,0].std(ddof=0)
    b = (m.iloc[:,1]-m.iloc[:,1].mean())/m.iloc[:,1].std(ddof=0)
    lags, vals = [], []
    for L in range(-max_lag, max_lag+1):
        if L<0: v = (a[:len(a)+L] * b[-L:]).mean()
        elif L>0: v = (a[L:] * b[:-L]).mean()
        else: v = (a*b).mean()
        lags.append(L); vals.append(v)
    s = pd.Series(vals, index=lags)
    return s

corr_we   = df["writes_entropy_norm"].corr(df["model_entropy"])
xc_surpr  = xcorr(df["surprise_ema"], df["model_entropy"])
lag_star  = int(xc_surpr.abs().idxmax())
val_star  = float(xc_surpr.loc[lag_star])
out["corr_writesEnt_vs_modelEnt"] = corr_we
out["xc_surprise_lag"] = lag_star
out["xc_surprise_corr"] = val_star

# 4) 클램프/포화 진단
thr = df.get("surprise_threshold")
clamp_hit = None
if thr is not None:
    lo, hi = thr.min(), thr.max()
    clamp_hit = ((thr==lo).mean(), (thr==hi).mean())
    out["thr_clamp_lo_ratio"] = clamp_hit[0]
    out["thr_clamp_hi_ratio"] = clamp_hit[1]

# 5) 이상치 비율
z, an = z_anoms(df["model_entropy"])
out["z_ge3_ratio_after1k"] = float(an[df.step>cut].mean())

# 6) 요약 출력
print("=== UZR v0.3 postrun ===")
for k,v in out.items(): print(f"{k}: {v}")

# 7) 빠른 판정(DoD)
def passfail(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")

passfail("writes-entropy vs model-entropy in [-0.35,-0.15]",
         corr_we is not None and -0.35 <= corr_we <= -0.15)

passfail("xcorr: surprise leads at +5~+10 with <= -0.05",
         5 <= lag_star <= 10 and val_star <= -0.05)

passfail("z>=3 after 1k <= 2%",
         out["z_ge3_ratio_after1k"] <= 0.02)

# 8) 참고 플롯
plt.figure(); plt.plot(df.step, df.model_entropy, lw=0.7); plt.title("Model entropy"); plt.savefig("plot_model_entropy.png", dpi=160)
plt.figure(); plt.scatter(df.surprise_ema, df.model_entropy, s=5, alpha=0.5); plt.title("Surprise vs Model entropy"); plt.savefig("plot_scatter.png", dpi=160)
plt.figure(); xc_surpr.plot(); plt.title("XCorr(surprise -> model_entropy)"); plt.xlabel("lag (+ = surprise leads)"); plt.savefig("plot_xcorr.png", dpi=160)
if "writes_entropy_norm" in df:
    plt.figure(); plt.plot(df.step, df.writes_entropy_norm, lw=0.6); plt.title("Writes entropy (norm)"); plt.savefig("plot_writes_entropy.png", dpi=160)
if thr is not None:
    plt.figure(); thr.plot(); plt.title("Surprise threshold"); plt.savefig("plot_threshold.png", dpi=160)
