# 아키텍처 개요

이 문서는 UZR의 핵심 구성요소와 "한 스텝" 학습 흐름을 요약합니다.

- 주요 모듈
  - 모델: `model.py`
    - `UZRModel`: 인코더/리드아웃, Self‑Eval, Identity(루리아) 의지, 전이 모듈(선택) 포함
    - 전이 모듈: `CodebookEncoder`, `MMFuse`, `TransHeadZ`, `TransHeadCB`
    - z 정규화 통계 EMA: `update_ema_stats`, `norm_z`
  - 학습 루프: `train_meta_3brains.py`
    - 3brains z: `slow_lang`, `slow_logic`, `bridge`
    - 이너스텝 적응(버킷 매핑 + 루리아 의지 오버라이드)
    - Self‑Eval + Abstain(규칙→자율) + 정확도 오버라이드
    - 메모리 정책/커밋/정비, 전이 버퍼, 요약 CSV/진행표
  - 메모리: `memory.py`
    - `CompressedMemory`: add/merge/skip, policy 임계치, 예측기
  - 태스크/데이터셋: `tasks.py`
    - `sample_task`, dataset‑mit 혼합, KoBERTTeacher(선택)

- 한 스텝 흐름(요약)
  1) 태스크 샘플링: 규칙 생성 or dataset‑mit 혼합(`--dataset_mix_prob`)
  2) 초기 지표 계산: conf/ent/verifier/길이 → 이너스텝 상한(s_max) 결정
  3) 루리아 의지 계산: `model.identity_intent_control()` → bias/toggle
  4) 이너스텝 결정: 버킷 매핑 → 루리아 의지로 [4..25] 범위에서 오버라이드
  5) 메모리 검색: top‑k(초기 [6..18] 매핑, 350+는 의지로 직접 제어)
  6) 이너 적응: 3brains z 업데이트(L1 패널티 포함)
  7) 전방패스: logits 계산
  8) Self‑Eval: conf/ent, Brier, identity 보조 손실
  9) 손실 결합: CE + λ_brier·Brier + 보조 + (전이 손실, 준비 시)
  10) 가중/게이트: abstain soft 가중, 정확도 오버라이드(정확하면 down‑weight/abstain 무시)
  11) 옵티마이저 스텝 + 클리핑 + 의지 기반 PI 조정(autonomous)
  12) 메모리 업데이트: composite gate → 정확도 향상+의지 긍정이면 강제 커밋
  13) 전이 버퍼 적재: z/u/코드북 시퀀스 추가(충분히 모이면 전이 손실 학습)
  14) 로그/요약: CSV/진행표(의지 ib/it 포함)

- 루리아의 의지 영향 경로
  - 이너스텝: [-1..1] → [4..25]
  - top‑k: [-1..1] → [6..18] (350 스텝 이후 직접)
  - abstain: 임계/히스테리시스에 bias/toggle 가산
  - 메모리: 정확도 향상 시 긍정 의지면 커밋 강제, 연산 종류 선택(합성/보간/교차/분할)

- 전이(멀티모달)
  - 입력: z(정규화), u(과제/언어/탑k 특징), 코드북 인코딩
  - 목표: Δz, 다음 z, 다음 코드북 분포 예측 + 정렬 손실
  - 람다(기본 x2 강화): lam_trans/lam_cb/lam_cos/lam_align/lam_roll

추가 세부는 각 파일별 문서를 참고하세요.
