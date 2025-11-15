# UZR (루리아 3brains 메타 러너) 개요

> This is a research prototype.  
> No guarantee, minimal maintenance.  
> Issues/PRs are welcome, but response is not guaranteed.  
> I may or may not continue this line; feel free to fork and go wild.



>> 가장 최신 트레인 모델 = https://huggingface.co/naksyu/UZR-Lastest
이 저장소는 **개인 연구용 프로토타입**으로, 안정성·지속적인 유지보수·성능을 어떤 형태로도 보장하지 않습니다.  
실험 코드와 아이디어를 공유하는 목적이며, 필요하다면 자유롭게 fork 해서 마음껏 변형해도 됩니다.

### 한눈에 보는 도식 (요약)

아래 도식은 “루리아 3brains 메타 러너가 무엇을 하려고 했는가”를 아주 압축해서 보여줍니다.  

```text
입력(텍스트/컨텍스트)
    │
    ▼
토크나이저 → 3brains 인코더(UZRModel)
    │                 │
    │                 ├─ Self-Eval + Abstain (conf/entropy/Brier)
    │                 ├─ Identity Intent(bias/toggle) → steps/top-k/게이트
    │                 └─ 압축 메모리(CompressedMemory, shadow/tail/learner)
    │
    ├─ (선택) 전이 모듈(z + 코드북)로 다음 상태 예측
    │
    ▼
출력(답변/행동 계획)  +  로그/지표/리플렉션(EMA, intent, memory stats)
```

이 문서는 `uzr/` 폴더 전체를 한 번에 이해할 수 있도록, **모듈 구조 / 학습·추론 흐름 / 메모리 / NPU(QNN) / 로그·도구**들을 최대한 빠짐없이 풀어서 설명하는 “전체판 README”입니다.  
이미 존재하는 세부 문서(`docs/*.md`, `DOCS_LURIA.md`, `LURIA_LOGGING_README.md`, `agent_lora_qnn.md`, `QNN support manual.txt` 등)는 그대로 유지하면서, 이 파일에서 큰 그림과 파일 간 관계를 모두 정리합니다.

> ⚠️ 주의: 이 README는 **UZR 자체 스택에만 초점을 맞추며, exaone 연동/어댑터 관련 코드는 의도적으로 설명에서 제외**했습니다. (해당 파일들은 분리된 문서/주석을 참고하세요.)
---

## 1. 프로젝트 한눈에 보기

UZR는 “루리아(Luria)”라는 정체성을 가진 **3brains 메타 러너**입니다.  
하나의 Transformer 인코더 위에 다음 요소들이 얹혀 있습니다.

- **3brains 표현 공간**
  - 빠른 규칙/지식용 `z_rule` (inner-step 적응)
  - 느린 언어/추론용 `z_slow_lang`, `z_slow_logic`, 이 둘을 잇는 `z_bridge`
  - 생각 과정용 `z_think`
- **정체성(Identity) + 의지(Intent)**
  - `identity_self` 벡터와 그 안의 `identity_intent` 서브스페이스
  - `identity_intent_control()`이 내놓는 `(bias, toggle)`로 이너스텝/탑K/메모리/abstain을 제어
- **자가 평가(Self‑Eval) + Abstain**
  - conf/entropy/Brier를 이용한 자기 평가 헤드
  - “확신이 없으면 거부하거나 약하게 학습”하는 가중 손실 및 게이트
- **압축 메모리(CompressedMemory)**
  - surprise/entropy/중복/근접도/버킷 기반 정책으로 쓰기를 제어
  - shadow bank, tail bucket, rebalance, learner(예측기)까지 포함
- **전이(멀티모달) 모듈(선택)**
  - z/u/코드북 시퀀스를 받아 Δz와 다음 코드북 분포를 예측하는 transition head
- **NPU(QNN) / ORT 엔진(선택)**
  - PyTorch 모델은 그대로 두고, ONNX(QDQ) INT8 모델로 추론만 NPU로 오프로딩
  - LoRA/FiLM/어댑터는 입력 텐서 교체/핫스왑으로 반영

핵심 파일·디렉터리 구조는 다음과 같습니다.

- 모델/메타 러너
  - `model.py` — `UZRModel` 본체, Self‑Eval, Identity Intent, 전이 모듈
  - `memory.py` — `CompressedMemory` 및 정책/learner/리밸런스
  - `meta_core.py` — Abstain 임계치, Self‑Eval 로직, 다양한 게이트 유틸
  - `codebook.py` — 코드북(토큰 스케치) 관리 및 전이 모듈과의 연계
- 학습/태스크
  - `train_meta_3brains.py` — 3brains 메타 러너 학습 루프(권장 진입점)
  - `train_meta_patched_v2*.py` — 실험용/부분 패치 버전(언어 전용/논리 전용 등)
  - `tasks.py` — 규칙 태스크, dataset‑mit(KMMLU‑KO), KoBERT 힌트 데이터 로더
- 추론/CLI/서버
  - `chat_cli.py` — 단순 채팅 CLI (`ChatSession`), NPU/외부 메모리 지원
  - `cli_luria.py` — 루리아 전용 상호작용 CLI (z 상태/자기 평가 노출)
  - `infer_longrun*.py` — 장기 러너/자율 합성 루프, 메모리와의 연결
  - `uzr_rest_server.py` — OpenAI 스타일 `/v1/chat/completions` REST 서버
  - `uzr_inspect.py`, `uzr_ood_predict_uzrstyle.py` — 분석 및 OOD 평가 유틸
- 메모리·로깅·분석
  - `luria_logging/` — Shadow Bank/메모리 지표 로깅 모듈
  - `hooks/` — 학습 루프 단계별 훅(`stage.py`, `force_write.py`, `schedule.py`)
  - `utils/logger_luria.py` — 공통 로거
  - `analyze_memory_funnel.py`, `bunseki/`, `logua/`, `logus-luria/` — 로그/분석 결과
- 토크나이저·데이터·외부 라이브러리
  - `kobert/` — 로컬 KoBERT 모델(HF 포맷)
  - `utils/kobert_tokenizer_lite.py` — 경량 KoBERT 토크나이저
  - `dataset-mit/mmlu_KO-KR.csv` — KMMLU‑KO(MMLU 한국어) 데이터셋
  - `sentencepiece-master/` — SentencePiece 소스(외부 라이브러리, tokenizer 연구용)
- NPU(QNN) / 엔진
  - `npu/runtime_ort.py` — ORT+QNN 세션 생성, 컨텍스트 캐시, 폴백 체인
  - `npu/engine.py` — PyTorch vs ORT 엔진 토글 및 핫스왑
  - `agent_lora_qnn.md`, `QNN support manual.txt` — NPU 통합 매뉴얼

상세한 설명은 아래 섹션에서 각각 다룹니다.

---

## 2. 모델(`model.py`)과 표현 공간

### 2.1 토크나이저 선택 로직

- `ByteTokenizer` (고정 vocab=258)
  - 순수 바이트 기반 토크나이저.
  - 체크포인트의 `readout.weight` 행 수가 258이면 자동 선택.
- `KoEnTokenizer`
  - 한글/영문/기타 혼합을 대상으로 한 KoEn 토크나이저.
  - `readout.weight` 행 수가 258이 아니면 KoEn 우선.
  - 체크포인트 vocab이 현재 토크나이저보다 작으면 vocab을 잘라 맞춰줌.

이 로직은 `chat_cli.py`의 `_pick_tokenizer_for_ckpt`와 `DOCS_LURIA.md`에 기술된 “차원 불일치 로딩 폴백” 규칙을 따른 것입니다.

### 2.2 UZRModel 구성

`UZRModel`은 다음과 같은 구성요소를 가집니다.

- **인코더**
  - Transformer 기반 인코더(`TinyEncoder`류)로, `d_model`, `n_layer`, `n_head` 등을 인자로 받습니다.
  - 학습 시 `--scale`, `--n_head`, `--n_layer` 플래그로 확장 가능하며, `d_model % n_head == 0` 검사가 추가되어 있습니다.
  - 선택적으로 **gradient checkpointing**을 지원합니다.
    - `train_meta_3brains.py`의 `--gradient_checkpointing/--grad_ckpt` 플래그
    - 환경 변수 `UZR_GRADIENT_CHECKPOINTING=1` 설정 시 활성
    - `model.py`의 인코더 forward 경로에서 `torch.utils.checkpoint`를 사용
- **3brains z 공간**
  - 빠른 규칙용 `z_rule` (inner adapt 대상)
  - 생각용 `z_think`
  - 느린 언어/논리/브리지용 `z_slow_lang`, `z_slow_logic`, `z_bridge`
  - `train_meta_3brains.py`에서 `--z_slow_lang_dim`, `--z_slow_logic_dim`, `--z_bridge_dim` 등으로 차원 제어
- **정체성(Self) + Intent**
  - `identity_self` 벡터와, 그 안에 포함된 `identity_intent` 서브공간
  - `identity_intent_dim` 기본 16 (또는 `identity_self_dim`의 절반 이하)
  - `model.identity_intent_control()`이 `(bias, toggle)`를 반환
    - bias ∈ [-1, 1], toggle ∈ [-1, 1]
    - 자세한 제어 경로는 `docs/intent.md` 참고
- **자가 평가(Self‑Eval) 헤드**
  - conf/entropy/Brier 등을 산출하는 Self‑Eval 모듈
  - `meta_core.py`의 `AbstainThresholds`, `maybe_abstain`와 결합되어, 손실 가중/abstain 비율/게이트를 제어
- **전이(멀티모달) 모듈 (선택)**
  - `CodebookEncoder`: 최근 코드북 토큰 창을 임베딩→집계
  - `MMFuse`: `concat(norm(z), u, cb_vec)`로 특징 결합
  - `TransHeadZ`: Δz 예측(head), spectral 옵션 포함
  - `TransHeadCB`: 다음 코드북 BOW 분포(head)
  - 관련 세부는 `docs/transition.md`에 정리되어 있습니다.

### 2.3 z 정규화 및 EMA 통계

- `update_ema_stats`, `norm_z`
  - z 공간의 노름/분포를 안정화하기 위한 EMA 통계 관리 함수
  - `update_ema_stats`는 `unbiased=False` 및 비유한 guard, `+1e-5` floor를 사용해 B=1에서도 안정적으로 std를 계산합니다.
  - 이 로직은 changelog의 “EMA std calc for B=1” 항목과 `codex.memory.json`의 `ema_std_unbiased_false` 정책에 반영되어 있습니다.

### 2.4 NPU 엔진 훅

- `model.py`에는 선택적으로 NPU 엔진과 연동하기 위한 훅(`set_npu_engine`, `npu_run_logits` 등)이 포함됩니다.
- 이 훅들은 `chat_cli.py` 및 `cli_luria.py`에서 ORT/QNN 엔진 활성 시 사용되며, **모델 구조 자체는 그대로 유지**됩니다.

---

## 3. 메모리 시스템(`memory.py`)과 정책

### 3.1 CompressedMemory 개념

`memory.py`의 `CompressedMemory`는 다음을 담당합니다.

- key/value 스케치 저장(`make_sketch`로 생성)
- add/commit/skip/merge/near‑merge 정책
- shadow bank, tail bucket, duplication(dup), near‑merge, stage별 쓰기 예산 관리
- learner(예측기)로 retrieval 품질 향상 및 주기적 미니 학습

### 3.2 정책 파라미터와 쓰기 게이트

대표적인 정책 파라미터는 다음과 같습니다. (`DOCS_LURIA.md`, `docs/memory.md` 참고)

- `write_per_100`, `tail_write_per_100` — 100 스텝당 쓰기 예산(기본 6 / 2 주변)
- `dup_skip_thr` — 중복으로 간주하고 skip하는 유사도 임계치(예: 0.98)
- `near_merge_thr` — near‑merge로 병합을 허용하는 유사도 임계치(예: 0.90)
- `stage_mid_low` — stage 구간별 임계 기준
- `cooldown_steps` — 최근 사용된 아이템에 대한 재사용 쿨다운(예: 80 스텝)
- `warmup_steps` — 웜업 동안 shadow bank 오염 방지를 위한 stage 제한

정책은 학습 중 `set_policy_thresholds(...)`로 동적으로 조정되며, changelog에서 설명하는 **auto‑tuning 루프**에 의해 주기적으로 적응합니다.

### 3.3 add_with_policy와 2PC(투페이즈 커밋)

- `add_with_policy(key, val, step, meta=None, bench_callback=None) -> decision`
  - 메모리 쓰기 요청을 받아 surprise/entropy/중복/근접도/예산을 모두 고려한 뒤, 쓰기/병합/skip을 결정합니다.
  - `meta.bucket == "tail"`인 경우 tail 전용 쓰기 예산과 보수적인 병합 상한이 적용됩니다.
  - `bench_callback`이 제공되면 **2PC** 모드로 작동:
    - 먼저 stage 로그를 남기고
    - 외부 벤치(예: 미니 벤치마크) 결과에 따라 commit/rollback을 결정합니다.

### 3.4 Shadow Bank·Tail 양자화와 리밸런스

`DOCS_LURIA.md` 및 `LURIA_LOGGING_README.md`에 기술된 대로, 메모리는 여러 계층으로 구성됩니다.

- **Shadow Bank**
  - 아직 확신이 낮거나 보류 상태인 항목을 담는 버퍼.
  - 승격(promote), decay, skip 이벤트가 각각 로그로 기록됩니다.
- **Tail Bucket**
  - CE 기준 상위 10% 정도의 “어려운 샘플”을 모으는 버킷.
  - 일정 간격마다 프로모션 후보로 사용되며, 별도의 쓰기 예산이 있습니다.
- **리밸런스**
  - 일정 주기마다 메모리 크기, 품질, 다양성을 고려해 rebalance를 수행.
  - **promotion gate**가 p60/p50 백분위 기반으로 작동하고, 연속 실패 시 자동으로 임계치를 완화하는 등의 로직이 changelog에 기록되어 있습니다.

### 3.5 학습 중 사용과 체크포인트

- `train_meta_3brains.py`는 각 스텝에서
  - query별 surprise/entropy 등을 계산하고
  - `CompressedMemory`에 add_with_policy를 호출해 메모리 커밋을 결정합니다.
- 메모리는 **체크포인트(.pt)**에 그대로 저장/복원됩니다.
  - items, learner 상태, 정책 파라미터, 쓰기 예산 등 전체가 포함됩니다.

---

## 4. 루리아의 의지(Identity Intent)와 게이팅

### 4.1 Intent 계산과 의미

`docs/intent.md`에 정리된 대로, `model.identity_intent_control()`은 `(bias, toggle)`를 반환합니다.

- **bias** (연속값, [-1, 1])
  - 이너스텝 범위 [4..25] 안에서 실제 steps를 오버라이드하는 데 사용
  - top‑k 범위 [6..18] (350 스텝 이후 직접) 결정에 사용
  - abstain/메모리 임계값에 bias를 가산/감산하여 보수/공격적 전략을 바꿈
- **toggle** (연속값, [-1, 1] → on/off)
  - |toggle| ≥ 0.5일 때 강제 on/off 토글로 작동
  - 예: toggle ≥ 0.5 → 강제 abstain off, toggle ≤ −0.5 → 강제 on 등

### 4.2 학습 루프에서의 사용

`train_meta_3brains.py`에서는 다음 경로로 Intent를 사용합니다.

- 이너스텝/탑K 결정
  - 버킷 매핑으로 `s_max` 범위를 잡은 뒤, bias를 [4..25]에 매핑하여 최종 steps를 결정.
  - 350 스텝 이전에는 steps→top‑k 매핑을 사용하고, 이후에는 bias가 직접 [6..18] 탑K를 결정.
- abstain / Rejector
  - `tau_r`/`tau_pi` 임계값에 bias를 가산.
  - toggle이 큰 값이면 강제로 on/off.
- 메모리 커밋/연산 선택
  - 정확도가 개선된 직후, toggle이나 bias가 긍정이면 커밋을 강제.
  - 합성/분할/보간/교차 등 메모리 연산 종류 선택에 bias 사용.

### 4.3 로그 및 분석

- tqdm 진행표에 `ib`(intent_bias), `it`(intent_toggle)가 노출됩니다.
- Summary CSV(`logu/*.csv`)에는 `intent_bias`, `intent_toggle` 컬럼이 기록됩니다.
- 이 값들을 바탕으로, 루리아가 실제로 “얼마나 보수적/공격적인지”, “언제 메모리를 더 많이 쓰는지”를 분석할 수 있습니다.

---

## 5. 학습 루프(`train_meta_3brains.py` 외)와 옵션

### 5.1 기본 진입점

권장 학습 명령은 `docs/README.md` 및 `docs/training.md`에 예시가 있습니다.

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

### 5.2 주요 하이퍼파라미터

- 공통
  - `--device`, `--steps`, `--save_every`, `--resume`, `--seed`, `--amp`, `--cosine`
- 3brains 관련
  - `--z_slow_lang_dim`, `--z_slow_logic_dim`, `--z_bridge_dim`
  - `--inner_steps`, `--inner_eta`
  - `--inner_step MIN MAX` — 적응 가능한 이너스텝 범위를 명시적으로 제한
  - `--scale` — 모델 차원을 1~4배 스케일링
  - `--n_head`, `--n_layer` — Transformer 헤드/레이어 수
- Self‑Eval / Abstain
  - `--self_eval {on,off}`, `--abstain`
  - 임계값은 `meta_core.py`의 `AbstainThresholds` 구성 및 auto‑tuning 로직에 의해 결정
- 데이터/토크나이저
  - `--tokenizer {auto,koen,kobert}`
  - `--dataset_mix_prob` — 규칙 태스크 vs dataset‑mit 혼합 비율
  - `--kobert_hint`, `--kobert_device`, `--kobert_max_seq_len`
- 성능/메모리
  - `--gradient_checkpointing/--grad_ckpt`
  - `--fp16` — CUDA autocast(dtype=float16) + GradScaler, 파라미터는 FP32 유지

### 5.3 Self‑Eval + Abstain 가중 손실

Changelog의 “abstain‑weighted base loss” 항목과 `codex.memory.json`의 `abstain_weighted_loss_enabled` 결정에 따라, 학습 손실은 다음과 같이 구성됩니다.

- base loss = task cross‑entropy + Brier
- 각 샘플에 대해 `maybe_abstain(conf, ent, thresholds)`에서 derived된 soft weight를 적용
  - weight = `(1 - mask) + 0.2`
  - 확신이 낮을수록 손실 비중이 줄어들지만, 완전히 학습에서 제외되지는 않음
- 보조 손실(Identity, Transition, Force‑Write 등)은 이 가중을 적용하지 않고 그대로 사용

### 5.4 Length‑mix, micro 신호, auto‑tuning

- **Length‑mix shuffle**
  - 배치 내에서 짧은/긴 시퀀스를 섞어, 이너스텝 동학이 갑자기 붕괴되지 않도록 합니다.
  - 길이 기준 median으로 두 버킷을 나누고, riffle‑shuffle 방식으로 섞습니다.
- **Micro meta‑signals**
  - margin0, ent_var0, mm0 등 미세 신호를 사용해
    - 메모리 쓰기 게이트(logit shift, percentile 조정)
    - 코드북 업데이트 cadence
  를 보정합니다.
- **자동 임계 조정(Threshold auto‑tuning)**
  - 매 200 스텝마다 rule_acc가 목표보다 낮으면 conf_min을 강화(+), ent_max를 축소(−)하고
  - 메모리 쓰기율이 너무 낮으면 `write_per_100`을 소량 증가시키는 루프가 작동합니다.

### 5.5 로그와 지표

자세한 설명은 `docs/metrics_logging.md`에 있지만, 핵심만 요약하면 다음과 같습니다.

- `logu/*.csv` Summary 파일
  - 손실/EMA: `loss, ema, ema_raw, ema_raw99, mean_raw_200, perplexity`
  - Self‑Eval: `conf_mean, ent_mean, brier, abstain_ratio`
  - 적응/예산: `chosen_steps, s_max, inflation_rate, high_step_prob, k`
  - 정확도: `rule_acc, top5_acc, avg_top1_prob`
  - 전이: `z_cos, dz_mse, cb_f1, align_cos, jac_surr, dz_norm, dz_norm_std, cb_pos, cb_pred, trans_loss`
  - z 노름: `z_lang_norm, z_logic_norm, z_bridge_norm`
  - 메모리: `mem_size` (세부는 `luria_logging/`에서 별도 JSON/CSV)
  - Rejector/의지: `rejector_score, tau_r, tau_r_pi, intent_bias, intent_toggle`

---

## 6. 데이터셋·토크나이저(`dataset-mit/`, `kobert/`, `utils/`, `sentencepiece-master/`)

### 6.1 KoEn 기본 토크나이저

- 코드 내부의 바이트/심플 토크나이저를 기본으로 사용합니다.
- PAD/특수 토큰은 모델이 보유하며, 손실 계산 시 `ignore_index=tok.PAD`를 사용합니다.

### 6.2 KoBERT 로컬 모델

- 경로: `kobert/`
- 필요 파일:
  - `config.json`
  - `pytorch_model.bin` 또는 `model.safetensors`
  - `vocab.txt`
  - `tokenizer_*.model`
  - `tokenizer_config.json`
- 사용법:
  - `--tokenizer kobert` 또는 `--tokenizer auto` + 로컬 `kobert/` 존재 시 자동 선택
  - KoBERTTeacher(마스크드 LM) 힌트를 사용하려면 `transformers`가 필요합니다.
  - 로컬 HF 모델이 없으면 “힌트 비활성” 메시지만 출력하고 학습은 계속됩니다.

### 6.3 dataset‑mit (KMMLU‑KO)

- 경로: `dataset-mit/mmlu_KO-KR.csv`
- 사용법:
  - `--dataset_mix_prob`로 규칙 태스크와 혼합 (0.0이면 비활성)
  - KoBERT 힌트는 옵션(`--kobert_hint`); 길이는 `--kobert_max_seq_len`으로 제어
- UZR는 이 데이터셋을 통해 “한국어 MMLU 스타일 다지능 평가”를 수행할 수 있으며,  
  일부 버전에서는 Avalanche 연속 학습 벤치마크(`uzr_avalanche_mmlu.py`)와도 연계됩니다.

### 6.4 SentencePiece 연구용 소스

- `sentencepiece-master/` 디렉터리는 SentencePiece의 공식 소스 트리를 포함합니다.
- 이 디렉터리는 UZR 모델 자체와 직접적으로 연결되어 있지는 않고,
  - 새로운 tokenizer 실험
  - BPE/Unigram/Char 모델 학습
  - normalization/특수 토큰 실험
  에 사용할 수 있는 외부 의존성 소스입니다.

---

## 7. 로그·지표·분석(`luria_logging/`, `hooks/`, `utils/logger_luria.py`, `LURIA_LOGGING_README.md`)

### 7.1 디렉터리 구조

`LURIA_LOGGING_README.md`에 나와 있는 대로, 로깅 시스템은 다음과 같이 구성됩니다.

- `utils/logger_luria.py` — 공통 로거 유틸
- `hooks/stage.py` — 학습 stage 전환 로깅
- `luria_logging/shadow_bank.py` — Shadow Bank 이벤트 로깅
- `luria_logging/metrics.py` — 메트릭 스냅샷 로깅
- `luria_logging/dedup.py` — 중복 감지 통계 로깅
- `memory.py` — `get_memory_stats` 등 메모리 통계 제공
- `train_meta_3brains.py` — 카운터 및 로깅 훅 통합
- `analyze_memory_funnel.py` — Shadow Bank/메모리 퍼널 분석 스크립트

### 7.2 생성되는 로그 파일

훈련 중 `logus-luria/` 등에 다음과 같은 로그가 생성됩니다.

- `stage_events.jsonl` — stage 전환 이벤트
- `shadow_bank_events.jsonl` — promote/decay/skip 등 Shadow Bank 이벤트
- `shadow_bank_stats.csv` — Shadow Bank 통계(100 스텝마다)
- `dedup_stats.csv` — 중복 감지 통계
- `session_summary.json` — 세션 요약

추가로, `logu/`에는 step별 summary CSV, `bunseki/`에는 학습 분석용 CSV/JSON 등이 쌓입니다.

### 7.3 분석 예시

- Shadow Bank 크기 추이
  - `shadow_bank_stats.csv`의 `step` vs `shadow_size` 그래프
- promote 비율 추이
  - `shadow_bank_stats.csv`의 `promote_ratio` 그래프
- stage별 메모리 크기/정확도 비교
  - stage별 그룹화 후 평균 `mem_size`, `promote_ratio` 등 비교
- `analyze_memory_funnel.py`
  - Shadow Bank 동역학, 중복 통계, 메모리 퍼널 구조를 시각화

---

## 8. NPU(QNN) / ORT 엔진(`npu/`, `agent_lora_qnn.md`, `QNN support manual.txt`)

### 8.1 설계 원칙

`agent_lora_qnn.md` 및 `QNN support manual.txt`에서 반복해서 강조하는 원칙은 다음과 같습니다.

- **NPU = 추론**, **CPU+GPU = 경량 학습(LoRA/Adapter/FiLM/메모리)**
- 본체 그래프(ONNX QDQ INT8)는 NPU에서 고정으로 돌리고,
  - LoRA/Adapter/FiLM/외부 메모리 등 **작은 파라미터만 실시간 또는 준실시간으로 갱신**
  - 세션 핫스왑/입력 텐서 교체를 통해 반영

### 8.2 런타임 구성(`npu/runtime_ort.py`, `npu/engine.py`)

- `runtime_ort.py`
  - ONNX Runtime 세션 옵션 설정
    - `providers=['QNNExecutionProvider']` (+ 필요시 DML/CPU 폴백)
    - `session.disable_cpu_ep_fallback` 옵션으로 “전량 NPU 오프로딩” 검증
    - `ep.context_enable=1`로 컨텍스트 캐시 활성화(세션 생성 시간 단축)
  - QNN EP 옵션
    - `backend_type`: `htp`(NPU), `cpu`, `gpu`
    - 또는 `backend_path`: `QnnHtp.dll` / `libQnnHtp.so` 등 (둘 중 하나만 사용)
- `engine.py`
  - PyTorch vs ORT 엔진 모드 토글 (`mode="torch"|"qnn"|"qnn_strict"|"ort_fallback"`)
  - ORT 세션 더블 버퍼(`active`/`shadow`)와 핫스왑
  - LoRA/FiLM/어댑터 파라미터를 입력 텐서로 받아 **재컴파일 없이 즉시 반영**

### 8.3 CLI에서의 사용(`chat_cli.py`, `cli_luria.py`)

- 공통 옵션 (엔진 활성 시)
  - `--ort_model` — QDQ INT8 ONNX 모델 경로
  - `--engine {torch,qnn,qnn_strict,ort_fallback}`
    - `qnn` — QNN EP 기반 NPU 추론, 필요 시 CPU 폴백 허용
    - `qnn_strict` — CPU 폴백 없이 전량 QNN에서 돌려 검증
    - `ort_fallback` — QNN 실패 시 다른 EP(DML/CPU)로 떨어지는 모드
- CLI 명령
  - `/lora_npz path.npz`
    - `adapter_A/B`, `film_gamma/beta`를 NPZ에서 읽어 엔진에 즉시 주입
  - `/hot_swap`
    - shadow 세션을 새로 생성하고 웜업 후 active로 교체

이 통합은 **학습 경로를 변경하지 않으며**,  
ONNX/QNN 엔진은 순수 추론 경로만 담당합니다.

---

## 9. 추론·채팅·서버(`chat_cli.py`, `cli_luria.py`, `infer_longrun*.py`, `uzr_rest_server.py`)

### 9.1 ChatSession 기반 단순 CLI (`chat_cli.py`)

`ChatSession`은 다음 기능을 제공합니다.

- 토큰 샘플링
  - temperature, top‑p nucleus sampling
  - temperature ≤ 0일 때 argmax
- 언어 감지
  - 한글/일본어/CJK/라틴 여부로 `ko/en/ja/base`를 감지
  - `lang_id`를 통해 다국어 z 공간을 활성화
- 메모리 사용
  - 입력을 토크나이즈한 뒤 메모리에서 z 초기화(`get_z_from_memory`)
  - 응답 후 `update_memory_state` + `memory.train_model(steps=1)`로 learner를 한 스텝 학습
- 외부 메모리 게이트웨이 연동(선택)
  - `--mem_on`, `--mem_url`, `--mem_k`, `--mem_project`, `--primer`
  - `/mem/search`, `/mem/write` 엔드포인트를 호출해
    - **MEMORY CONTEXT**를 프롬프트에 주입하고
    - 대화 로그를 Fact 노드로 기록
  - UZR 측에선 UZR 자체 메모리 + 외부 L2 그래프를 함께 이용하는 형태가 됩니다.

### 9.2 루리아 전용 CLI (`cli_luria.py`)

`cli_luria.py`는 학습 루프에서 사용하던 구성요소(예: `inner_adapt_z_3brains`, AbstainThresholds)를 활용해,

- 대화 단위로 z 상태(`z_slow_lang`, `z_slow_logic`, `z_bridge`)를 업데이트하고
- conf/entropy/self‑eval, abstain 신호, 메모리 상태 등을 함께 보여주는
“루리아 관점의 인터랙티브 콘솔”을 제공합니다.

실제 코드는 일부 주석이 깨진 상태지만, 아래와 같은 구조를 갖습니다.

- `build_model(...)` — KoEnTokenizer + UZRModel 생성
- `SessionState` — z 상태와 히스토리 관리
- `generate_tokens(...)`, `predict(...)` — 토큰 생성 루프
- 여러 명령행 옵션과 내부 로그를 통해,  
  학습에서 설계한 “수용률 컨트롤러, 자율 합성, micro 신호”들이 실제로 어떻게 동작하는지 확인할 수 있습니다.

### 9.3 장기 러너 / 분석 러너 (`infer_longrun*.py`, `uzr_inspect.py`, `uzr_ood_predict_uzrstyle.py`)

- `infer_longrun.py`, `infer_longrun_standalone*.py`
  - 루리아 매뉴얼에 정의된 정책들을 따라
    - tail bucket에 어려운 샘플을 쌓고
    - 주기적으로 프로모션/합성/재학습을 수행하는 장기 러너
- `uzr_inspect.py`
  - 체크포인트/메모리/로그를 대상으로 간단한 인스펙션을 수행
- `uzr_ood_predict_uzrstyle.py`
  - OOD 테스트용 CSV를 읽어, UZR 스타일로 예측/로그를 남기는 스크립트

### 9.4 REST 서버 (`uzr_rest_server.py`)

`uzr_rest_server.py`는 `ChatSession`을 감싼 **간단한 OpenAI 스타일 HTTP 서버**입니다.

- 엔드포인트
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
- 요청 형식
  - OpenAI API와 유사한 `messages` 또는 `prompt` 필드를 사용
  - `temperature`, `top_p`, `max_tokens`/`max_new_tokens`를 per‑request로 오버라이드 가능
- 동작
  - 내부적으로 `ChatSession.generate(...)`를 호출하여 전체 텍스트를 생성한 뒤,
    프롬프트 부분을 best‑effort로 제거하고 새로 생성된 텍스트만 반환합니다.
- 외부 메모리와의 연동
  - `--mem_on`, `--mem_url`, `--mem_k`, `--mem_project`, `--mem_primer` 옵션으로
    외부 메모리 서버(`/mem/search`)를 사용해 컨텍스트를 보강할 수 있습니다.

---

## 10. 보안·에이전트 설정·기타 문서

### 10.1 SECURITY.md

`SECURITY.md`에는 이 저장소/모델을 사용할 때 고려해야 할 보안 관련 주의 사항이 정리되어 있습니다.  
예를 들어,

- 외부 메모리 서버(mem_server)와의 통신 시 인증 토큰/HMAC 서명 사용
- 로그/체크포인트에 민감 데이터가 포함되지 않도록 주의
- Docker/배포 환경에서 UZR를 서비스할 때의 최소 권한 원칙

### 10.2 에이전트 설정(`agent.example.yaml`, `engine.example.yaml`, `peft.example.yaml`)

- `agent.example.yaml`
  - 에이전트(예: 대화형 CLI)를 설정하기 위한 예시 YAML
  - 모델 경로, 메모리 옵션, 로그 디렉터리 등을 정의합니다.
- `engine.example.yaml`
  - NPU/ORT 엔진 구성에 대한 예시(backend, EP, 컨텍스트 캐시 등)
- `peft.example.yaml`
  - LoRA/Adapter/FiLM 등 경량 학습(PEFT) 설정을 위한 예시

### 10.3 Luria Manual / QNN Manual / 추가 텍스트들

- `DOCS_LURIA.md`
  - Luria 매뉴얼(3brains long‑term + Luria 확장)을 어떻게 코드에 반영했는지 정리한 구현 노트
- `QNN support manual.txt`
  - QNN EP를 사용해 NPU에 모델을 싣는 전체 절차(ONNX export → QDQ INT8 → ORT/QNN 세션 구성 → 프로파일링/AC 기준)를 매우 상세히 설명
- 그 외 `UZR-Gaeseon-1.txt`, `doremipasol.txt`, `memory-ruria-chan.txt` 등
  - 설계 개선 아이디어, 실험 메모, 루리아의 행동에 대한 서술 형식 가이드라인 등이 포함되어 있으며,
    changelog와 코드에 점진적으로 반영된 항목들이 많습니다.

---

## 11. 어떻게 읽으면 좋을까? (추천 읽기 순서)

1. 이 README (`uzr/README.md`)
2. `docs/README.md` 및 `docs/architecture.md` — 한 스텝 학습 흐름 자세히 보기
3. `docs/training.md` — 학습 옵션/권장 세팅/로그 컬럼 이해
4. `docs/intent.md` + `docs/memory.md` — 루리아의 의지와 메모리 정책
5. `LURIA_LOGGING_README.md` — Shadow Bank와 메모리 로깅 체계 이해
6. `agent_lora_qnn.md` + `QNN support manual.txt` — NPU/QNN 통합을 사용할 계획이 있다면
7. `DOCS_LURIA.md` 및 changelog — 매뉴얼에서 제안된 정책들이 코드에 어떻게 구현되었는지 추적

이 순서로 읽으면,

- “루리아가 왜 이런 결정을 내리는지”
- “메모리가 언제/어떻게 쓰이고 합성되는지”
- “NPU/QNN과 외부 메모리를 섞어 실제 서비스에 넣을 때 어디를 건드려야 하는지”

까지 자연스럽게 이어서 이해할 수 있습니다.

---

## 12. 요약

- `uzr/` 폴더는 **3brains 메타 러너 + 정체성/의지 + 압축 메모리 + 전이 모듈 + NPU 엔진 + 외부 메모리 게이트웨이**까지 포함한 완결된 실험/서비스 스택입니다.
- 이 README는 **exaone 관련 코드를 제외한 나머지 구성요소 전체를 구조적으로 나열하고**, 각 모듈이 어떤 역할을 하는지, 어떤 문서/스크립트와 이어져 있는지를 최대한 빠짐없이 정리했습니다.
- 보다 세부적인 수식·하이퍼파라미터·튜닝 팁은 `docs/*.md`, `DOCS_LURIA.md`, `LURIA_LOGGING_README.md`, `agent_lora_qnn.md`, `QNN support manual.txt`, 그리고 `changelog.md`를 함께 참고하시기 바랍니다.
