# 전이(멀티모달) 모듈

- 구성요소 (model.py)
  - `CodebookEncoder`: 최근 코드북 토큰 창을 임베딩→집계
  - `MMFuse`: `concat(norm(z), u, cb_vec) → fused`
  - `TransHeadZ`: Δz 예측(head); spectral 옵션 포함
  - `TransHeadCB`: 다음 코드북 분포(BOW) 예측(head)

- 입력 특징
  - z 정규화: `norm_z()` (EMA 통계 사용)
  - u: `[topk_norm, task_type(binary), lang_idx_norm]`
  - cb: 최근 코드북 토큰 id 시퀀스(B 길이 창)

- 손실(기본 2배 강화)
  - `L_delta`(Δz MSE), `L_cos`(z_t1 vs pred cosine), `L_cb`(BCE‑logits BOW), `L_align`(z↔cb 정렬), `L_jac`(자코비안 대리), `L_roll`(옵션)

- 버퍼/학습 조건
  - transition_buffer 길이 ≥ 1024일 때 미니배치 샘플링 후 전이 손실 추가
  - Summary CSV 지표: `z_cos, dz_mse, cb_f1, align_cos, jac_surr, dz_norm, dz_norm_std, trans_loss`

- 튜닝 팁
  - 손실 규모가 과도하면 lam_*을 절반으로 낮춤
  - 코드북 품질이 낮으면 gate 완화/리밸런스 주기 조정 고려
