# Luria Manual Integration — Implementation Notes

본 문서는 Luria 매뉴얼(3brain long-term + Luria 확장)에 따라 본 저장소에 반영된 구현 사항과 사용 방법을 요약합니다.

## 구현 개요
- 정책 기반 메모리 쓰기(`add_with_policy`): surprise/entropy/중복/근접/레이트 제한/테일 버킷/웜업/2PC 훅.
- 리트리벌 다양성: 최근 질의 재사용 패널티(`lambda_penalty`)와 80-step 쿨다운, 탐색 펄스 훅.
- 표준 로깅: writes.csv/entropy.csv, 공통 컬럼 유지.
- 차원 불일치 로딩 폴백: 3brains 입력차원을 ckpt로부터 유추.
- 러너 보강: 테일 버킷(top10% CE) 축적·주기적 프로모션.

## 기본 파라미터(권장/기본)
- softmax_temp: 0.07 (retrieval entropy/softmax에 사용)
- lambda_penalty: 0.22 (재사용 패널티 강도)
- write_per_turn: 1
- write_per_100: 6 (오토튜닝 시 8권장), tail_write_per_100: 2
- dup_skip_thr: 0.98, near_merge_thr: 0.90, stage_mid_low: 0.70
- cooldown_steps: 80
- warmup_steps: 300 (웜업 중 shadow_bank 오염 방지를 위해 stage 금지)

## 메모리 API 확장
- add_with_policy(key, val, step, meta=None, bench_callback=None) -> decision
  - bench_callback 제공 시 2PC: stage 로그 → 벤치 → commit/rollback 로깅.
  - meta.bucket == "tail"이면 병합 베타 상한 0.35 적용, 별도 쓰기 예산(2/100) 사용.
- set_policy_thresholds(...): 실행 중 정책 임계치/예산 조정.
- exploration_pulse(topk=8, lambda_pulse=None, temp_pulse=None): 일시적 탐색 강화.

## 로깅 스키마
- writes.csv 공통 컬럼: step, action, reason, sim_max, surprise, surp_norm, entropy, topk, used_key_id, shadow_size, mem_size (+ 필요 시 extra 컬럼)
- entropy.csv: step, entropy
- rollbacks.csv: step, reason, surprise, entropy

## 러너 연계(infer_longrun_standalone.py)
- recent CE 상위 10% 샘플을 tail_q에 저장, 요약 간격마다 최대 5개 프로모션(add_with_policy, bucket='tail').
- memory.write 메타에 {'desc','lang','ce_q','conf'} 포함.
- init_from_retrieval_multi()/model.get_z_from_memory(): topk=None 시 mem.topk 사용.

## 체크포인트 로더(standalone_logged)
- ckpt의 fuse_proj_3brains.weight 입력 차원에서 (lang, logic, bridge) 차원 유추 폴백.
- readout 행 수(258)로 ByteTokenizer/KoEnTokenizer 자동 선택.

## 남은 항목(후속 작업 권장)
- Self-TaskGen/Rule Transfer/Policy Auto-Tuning 루프 본격 연결.
- Rest Hook: 과열 감지 → 탐색/학습 일시 중단/복구.
- 2PC 벤치 콜백을 러너에서 실제 평가 지표(미니벤치)로 연결.

