# 데이터셋·토크나이저

- TCodebook 슬라이딩 토크나이저(기본)
  - 구조 해시 전용 토커나이저(Gt=4, Kt=256 → 1024 구조 토큰 + 특수 토큰 오프셋)
  - 슬라이딩 윈도우(window=8, stride=4 기본)로 잘라서 인코딩하며 ids→text 복원은 지원하지 않음(디버그용 ids→구조토큰 문자열만 제공)
  - `--tokenizer tcode`(또는 `--tokenizer auto`, 기본)로 사용, 윈도우/스트라이드는 `--tcode_window/--tcode_stride`로 조정

- KoEn 토크나이저
  - (비활성) 기존 바이트/심플 토크나이저였으나 현재 경로에서는 사용하지 않음

- KoBERT (로컬, legacy opt-in)
  - 경로: 프로젝트 루트의 `kobert/`
  - 필요 파일: `config.json`, `pytorch_model.bin`(또는 `model.safetensors`), `vocab.txt`, `tokenizer_*.model`, `tokenizer_config.json`
  - 자동 선택하지 않음(legacy). `--tokenizer kobert`로 명시적으로만 사용 가능.
  - KoBERTTeacher(마스크드LM) 힌트는 `transformers`가 필요하며 로컬 HF 모델이 없으면 비활성 메시지 후 진행

- dataset‑mit (KMMLU‑KO)
  - 기본 경로: `dataset-mit/mmlu_KO-KR.csv`
  - `--dataset_mix_prob`으로 규칙 태스크와 혼합(0.0은 비활성)
  - KoBERT 힌트는 옵션(`--kobert_hint`); 토큰 길이는 `--kobert_max_seq_len`

- PAD/특수 토큰 주의
  - KoBERT의 PAD id는 0이 아닐 수 있음 → 손실/지표에서 항상 `ignore_index=tok.PAD` 사용
