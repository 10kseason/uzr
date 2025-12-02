# 문제 해결 (Troubleshooting)

- KoBERT 힌트 비활성 메시지
  - 원인: 로컬 `kobert/`가 HF 포맷 모델이 아님
  - 조치: `config.json`, `pytorch_model.bin`(또는 `model.safetensors`), `vocab.txt`, `tokenizer_*.model` 존재 확인

- torch.qr 경고
  - 원인: `torch.qr` 폐기 예정(`codebook.py`)
  - 조치: `torch.linalg.qr(..., mode='reduced')`로 교체 권장

- std() DoF<=0 경고 또는 NaN
  - 원인: B=1에서 `std(unbiased=True)` 사용
  - 조치: `model.update_ema_stats`는 `unbiased=False`로 패치 완료(본 저장소)

- GPU 메모리 부족
  - 조치: `--steps`/배치 축소, 전이 버퍼 활성(1024) 지연, 전이 람다 축소

- dataset‑mit 파일 없음
  - 원인: CSV 경로 불일치
  - 조치: `dataset-mit/mmlu_KO-KR.csv` 경로 확인 또는 `--dataset_mit_path` 지정

- 로그가 비어있음
  - 원인: 요약 CSV 경로/권한 문제
  - 조치: `logu/` 디렉터리 쓰기 권한 확인
