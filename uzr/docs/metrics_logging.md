# 로그·지표 설명

- 진행표(tqdm)
  - 주요 키: `loss, ema, conf, s(chosen_steps), k(top‑k), ib/it(intent), τ_r, τ_pi, infl, acc, gate, abstain, ...`

- Summary CSV (logu/*.csv)
  - 학습 품질: `loss, ema, ema_raw, ema_raw99, mean_raw_200, perplexity`
  - Self‑Eval: `conf_mean, ent_mean, brier, abstain_ratio`
  - 적응/예산: `chosen_steps, s_max, inflation_rate, high_step_prob, k(top‑k)`
  - 정확도: `rule_acc, top5_acc, avg_top1_prob`
  - 전이: `z_cos, dz_mse, cb_f1, align_cos, jac_surr, dz_norm, dz_norm_std, cb_pos, cb_pred, trans_loss`
  - z 노름: `z_lang_norm, z_logic_norm, z_bridge_norm`
  - 메모리: `mem_size`, (주기적 스냅샷은 luria_logging/ 하위 모듈)
  - Rejector: `rejector_score, tau_r, tau_r_pi`
  - 의지: `intent_bias, intent_toggle`
  - 모드/진단: `mode, amp, coverage, use_autonomous, difficulty_level, composite_score, ...`

- 팁
  - `ema_min_csv/ckpt/delta`로 학습 최저점 추적
  - 전이 지표가 나쁠 때는 lam_* 조정, 버퍼 크기 확인(>=1024)
