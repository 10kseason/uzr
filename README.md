# UZR — Universal Z‑Rule Runner

**UZR**는 메타학습(Meta-learning)을 기반으로 한 언어 규칙 추론 시스템입니다. 단일 네트워크 `f_θ(x, z)`가 **작은 latent 벡터 `z`만 적응**시켜 few-shot 예시로부터 언어 규칙을 학습하고 적용합니다.

## 핵심 개념

**목표:**
- Few-shot context pairs로부터 규칙을 추론
- 긴 대화 세션 지원 (split state: `z_slow + z_fast`)
- Confidence 기반 게이팅 및 압축 메모리 리플레이
- Meta-cognition (자기평가) 메커니즘

**핵심 방정식:**

**규칙 추정 (inference-time):**
```
z* = arg min_z Σ_{(x,y)∈C} ℓ(f_θ(x,z), y) + λ ℛ(z)
```

**예측:** `ŷ = f_θ(x_q, z*)`

**장기 안정성:** `z = z_slow + z_fast`, proximal updates with soft-threshold, confidence-gating, periodic replay

---

## 프로젝트 구조

### `uzr/` 패키지 (핵심 모듈)

**핵심 파일:**
- `model.py` — Tiny Transformer encoder with **FiLM** conditioning, byte/KoEn tokenizer
- `memory.py` — Compressed memory with sketch, cosine-similarity retrieval, write-on-surprise policy
- `tasks.py` — Synthetic language-like rule families (한/영/일 지원)
- `meta_core.py` — Self-evaluation head, abstain policy, confidence/entropy estimation

**학습 스크립트:**
- `train_meta.py` — Meta-learning loop: z-only inner updates
- `train_meta_3brains.py` — 3-brain architecture (language, logic, bridge) 학습
- `train_meta_patched_v2.py` — 향상된 버전 (confidence gating, memory policy)

**추론 스크립트:**
- `infer_longrun.py` — Long-turn (10k+ turns) 추론 데모
- `infer_longrun_standalone.py` — 독립 실행 버전
- `cli_luria.py` — 대화형 CLI (Luria)

**문서:**
- `AGENTS.md` — 에이전트/개발 가이드라인
- `DOCS_LURIA.md` — Luria manual 구현 노트
- `CHANGELOG.md` — 상세 변경 이력

### 메인 폴더 (실험 스크립트)

**대화형 인터페이스:**
- `chat_cli.py` / `chat_cli_standalone.py` — 간단한 대화 CLI
- `chat_cli_adaptive.py` — 적응형 CLI

**추론 & 평가:**
- `infer_longrun_standalone_logged.py` — 로깅 포함 추론
- `infer_longturn-logged-use.py` — 긴 대화 추론 with logging
- `infer_longrun_writer_5000.py` — 5000 step 실험
- `infer_longrun_exclude_rules.py` — 규칙 제외 실험

**분석 도구:**
- `forgetting_analyzer.py` — Catastrophic forgetting 분석
- `analyze_suspicious.py` — 의심스러운 패턴 분석
- `ood_composition_eval.py` — Out-of-distribution 조합 평가
- `uzr_ood_predict.py` / `uzr_ood_predict_v2.py` — OOD 예측

**유틸리티:**
- `inspect_checkpoint_dimensions.py` — 체크포인트 차원 검사
- `byte_tokenizer_decoder.py` — Byte tokenizer 디코더
- `probe_utils.py` — 프로빙 유틸리티

---

## 설치 및 실행

### 요구사항
- Python 3.10+
- PyTorch 2.2.0+
- NumPy, tqdm

### 설치

```bash
pip install -r requirements.txt
```

### 빠른 시작

**1. 메타학습 (토이 실험):**
```bash
python -m uzr.train_meta --steps 500 --device cpu
```

**2. 3-brain 학습:**
```bash
python -m uzr.train_meta_3brains --steps 1000 --device cuda
```

**3. 장기 추론 데모:**
```bash
python -m uzr.infer_longrun --turns 200 --device cpu
```

**4. 대화형 CLI (Luria):**
```bash
python -m uzr.cli_luria --device cpu --ckpt uzr_3brains_ckpt_best.pt
```

### 주요 실험 스크립트

**로깅 포함 추론:**
```bash
python infer_longrun_standalone_logged.py \
  --ckpt uzr_ckpt_best.pt \
  --turns 100 \
  --inner_steps 5 \
  --device cpu
```

**Forgetting 분석:**
```bash
python forgetting_analyzer.py
```

**OOD 평가:**
```bash
python ood_composition_eval.py
```

---

## 주요 기능

### 1. Split-State 아키텍처
- `z_slow`: 장기 규칙 표현
- `z_fast`: 단기 적응

### 2. 압축 메모리 시스템
- Write-on-surprise 정책
- Shadow bank (1k buffer)
- Near-duplicate merge/skip
- Entropy-based filtering
- 80-step cooldown for diversity

### 3. Meta-Cognition
- Self-evaluation head (confidence estimation)
- Abstain policy (불확실한 답변 거부)
- Brier score calibration
- Dynamic inner-step scheduling

### 4. 3-Brain Architecture
- **Language brain**: 언어 패턴 처리
- **Logic brain**: 논리 규칙 처리
- **Bridge**: 두 영역 통합

### 5. Policy-based Memory Writing
- Surprise normalization
- Rate limiting (write budget)
- Tail-bucket for high-CE samples
- 2PC (two-phase commit) with mini-bench

---

## 주요 실험 결과 (예시)

실험 로그는 다음 파일에 저장됩니다:
- `infer_summary.csv` / `infer_summary.json`
- `generation_log.csv`
- `detailed_step_log.csv`
- `sample_log.csv`, `retrieval_log.csv`, `threshold_log.csv`

---

## 라이선스 & 기여

이 프로젝트는 연구 및 교육 목적의 프로토타입입니다.

---

## Notes

- 모델은 **small** (hidden=256, z_dim=128). CLI 플래그로 조정 가능
- Byte-level tokenization 또는 KoEn tokenizer 사용
- `tasks.py`에 커스텀 규칙 패밀리 추가 가능
- **Proximal L1** on `z`: 규칙 sparsity/cleanliness 강제
- Confidence metric: negative loss / entropy 기반
- Memory sketch: averaged token embeddings + `z_slow` snapshot

---

## 참고사항

- Docker 설정 파일은 현재 불완전하여 리포지토리에서 제외되었습니다.
- 로컬 환경에서 Python으로 직접 실행하는 것을 권장합니다.

---

**잠재공간 실험실 시즌2** — Meta-learning for language rule inference
