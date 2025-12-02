# 메모리 시스템

- 구성: `memory.py`의 `CompressedMemory`
  - add/commit/skip/merge/near-merge 정책
  - 정책 임계치: `write_per_100`, `tail_write_per_100`, `near_merge_thr`, `stage_mid_low`, `dup_skip_thr` 등
  - 예측기(learner)로 retrieval 품질 개선; 주기적으로 미니학습

- 학습 중 사용
  - set_policy_thresholds: 스텝/자율구간에 따라 동적으로 갱신
  - add_with_policy: 게이트 통과 시 스케치(key,val) 추가
  - 정비: 리밸런스, LRU+품질 기반 정리, 주기적 codebook commit

- 커밋 게이트(요지)
  - composite_score = surprise(entropy) × z_bridge_norm + 코드북 미사용률 가중
  - gate_prob(sigmoid)와 메모리 충만도 기반 퍼센타일 임계로 결정
  - 정확도 향상 + 긍정 의지면 커밋 강제

- 연산(자율 시)
  - 합성, 분할, 보간, 교차를 주기적으로 선택/수행
  - 최근 정확도 추세/다양성/효율 손실로 분포를 학습
  - 정확도 향상 시 의지로 연산 종류를 결정

- 체크포인트
  - 메모리 items/학습기/정책 파라미터 일체가 학습 CKPT에 저장/복원
