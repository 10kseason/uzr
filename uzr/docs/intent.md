# 루리아의 의지(Identity Intent)

- 개념
  - `model.identity_intent_control()` → `(bias, toggle)` 반환. bias∈[-1,1], toggle∈[-1,1]
  - 학습/추론 중 모델 내부 identity_self 벡터의 intent slice에서 도출

- 학습에서의 영향
  - 이너스텝: bias → [4..25]로 매핑해 최종 steps를 오버라이드
  - top‑k: (350+ 스텝) bias → [6..18]로 매핑해 검색 예산 결정
  - abstain: `tau_r_adjusted`에 bias 가산, toggle이 ±0.5 넘으면 강제 on/off
  - 메모리: 정확도 향상 시 toggle 또는 bias≥0이면 커밋 강제; 연산 종류를 bias로 선택(합성/보간/교차/분할)

- 어떻게 확인하나
  - 학습 진행표: `ib`(bias), `it`(toggle) 노출
  - Summary CSV: `intent_bias`, `intent_toggle` 컬럼으로 기록

- 인자 설정
  - `--identity` 문자열(예: "루리아")로 identity QA에 사용
  - `--identity_intent_dim`(기본 16)로 intent 서브스페이스 크기 조절

- 참고 코드
  - 계산: model.py:592 `identity_intent_control`
  - 로그: train_meta_3brains.py(진행표/CSV 기록)
