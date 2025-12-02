# UZR 문서 (요약/목차)

이 디렉터리는 UZR 저장소의 핵심 구성요소, 학습/추론 흐름, 메모리/전이/자가평가(Abstain), KoBERT 힌트, 로그/지표 등을 정리한 문서 모음입니다. 백업 폴더(예: `백업 ( 이 폴더는 손대지마세요)/`)와 임시 파일은 문서화 범위에서 제외했습니다.

- 읽는 순서 추천
  1) `architecture.md` — 전체 구조/한 스텝 흐름
  2) `training.md` — 학습 CLI, 옵션, 권장 세팅
  3) `intent.md` — 루리아의 의지(Identity Intent)와 제어 경로
  4) `memory.md` — 메모리 정책/연산/게이트
  5) `transition.md` — 전이(멀티모달) 모듈과 지표
  6) `datasets_tokenizer.md` — 토크나이저/데이터셋 사용법
  7) `metrics_logging.md` — Summary CSV/진행표 항목 설명
  8) `troubleshooting.md` — 경고/오류 대응

- 빠른 시작
  - 학습 예시 (GPU, KoBERT 힌트+dataset‑mit 혼합):
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
  - 채팅 예시:
    ```bash
    python -m uzr.cli_luria --device cuda --resume uzr_3brains_ckpt_last.pt
    ```

- 주요 파일(코드)
  - 모델/전이: `model.py`
  - 학습 루프: `train_meta_3brains.py`
  - 메모리: `memory.py`
  - 태스크/데이터셋/KoBERT 힌트: `tasks.py`
  - 토크나이저(KoBERT-lite): `utils/kobert_tokenizer_lite.py`
  - 채팅 CLI: `cli_luria.py`, `chat_cli.py`
  - 오케스트레이터/추론: `uzr_orchestrator_cli.py`, `infer_longrun*.py`

문서와 코드가 불일치하면 코드가 우선합니다. 개선 제안은 환영합니다.
