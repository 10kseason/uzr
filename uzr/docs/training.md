# 학습 가이드

- 실행 커맨드 예시
  - KoBERT 힌트 + dataset‑mit 혼합 + 자가평가/abstain:
    ```bash
    python -m uzr.train_meta_3brains \
      --device cuda --steps 5000 \
      --z_slow_lang_dim 96 --z_slow_logic_dim 96 --z_bridge_dim 64 \
      --lam_lang 5e-4 --lam_logic 5e-4 --lam_bridge 3e-4 \
      --inner_steps 8 --inner_eta 0.425 \
      --identity "루리아" \
      --dataset_mix_prob 0.45 --kobert_hint --kobert_device cuda \
      --self_eval on --abstain --save uzr_3brains_ckpt.pt
    ```

- 핵심 옵션
  - 일반: `--device`, `--steps`, `--save_every`, `--resume`, `--seed`, `--amp`, `--cosine`
  - 3brains 차원/정규화: `--z_slow_lang_dim`, `--z_slow_logic_dim`, `--z_bridge_dim`, `--inner_steps`, `--inner_eta`
  - 토크나이저/데이터: `--tokenizer {auto,koen,kobert}`, `--dataset_mix_prob`, `--kobert_*`
  - Self‑Eval/Abstain: `--self_eval {on,off}`, `--abstain`
  - 보조 손실: `--id_weight`, `--lam_sem`, `--lam_transfer`

- 루프 개요(세부는 architecture.md)
  - 버킷 매핑으로 이너스텝 후보 → 루리아 의지로 [4..25] 오버라이드
  - top‑k [6..18] 예산(350+는 의지 직접 제어)
  - Self‑Eval: abstain soft 가중 + 정확도 오버라이드(정확하면 down‑weight/abstain 무시)
  - 전이 손실: 버퍼 1024 이상 시 활성, λ는 초기 2배 강화 기본

- 로그/요약 파일
  - Summary CSV: `logu/<timestamp>_train3_s<inner>_t<steps>.csv`
  - 주요 컬럼: `loss, ema, conf_mean, ent_mean, rule_acc, chosen_steps, k(top-k), ib/it(intent), trans_loss, z_cos, ...`
  - tqdm 진행표: `s, k, ib, it, τ_r, τ_pi, ...` 확인 가능

- KoBERT 힌트
  - 로컬 `kobert/` 폴더에 HF 포맷(`config.json`, `pytorch_model.bin`, `vocab.txt`, `tokenizer_*.model`)이 있어야 활성
  - 없으면 힌트 비활성 메시지 후 학습은 계속 진행

- 체크포인트
  - 학습 중 `*_last.pt`, 최저 EMA 기준 `*_best.pt` 저장
  - 메모리 state, 자율 매개변수(near_merge_thr, top_k, rejector 등) 포함

- 권장 팁
  - short run: steps=3~5k, dataset_mix_prob≈0.3~0.5, self_eval=on, abstain on
  - 전이 활성 시 람다 2배 강화 값으로 시작(기본 적용됨), 과도하면 λ를 절반으로 낮춤
