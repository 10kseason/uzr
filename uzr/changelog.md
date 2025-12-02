# Changelog (uzr)

All notable changes by the agent are recorded here.

## 2025-11-23 (Train 1.285x tuning note)

- Shared 1.285x-scaled train_meta_3brains.py flag values for a stronger run (e.g., --lr 0.00257, --inner_eta 0.546, --inner_steps 8, --steps 19275, --warmup_steps 1285, --autonomous_step 1928, id/sem/transfer/bridge/lang/logical regs ×1.285); guidance only, no code changes.

## 2025-11-22 (Bugfix: cli_luria sampling crash)

- Fixed `cli_luria.py:generate_tokens` to initialize `ids` and compute probabilities correctly; removed references to undefined variables (`ids/probs/top_k_idx/top_k_probs`) and added temperature+top-k/top-p sampling on the last-step logits so the CLI no longer raises at runtime.

## 2025-11-22 (Docs review, no code changes)

- Read full docs set (docs/README.md, architecture/training/intent/memory/transition/datasets_tokenizer/metrics_logging/troubleshooting, DOCS_LURIA.md, uzr model arc-2.txt, memory for Tokernizer.txt, Infra memory guide) to comply with doc-first policy; confirmed structural tokenizer (TCodebook-only) and memory/intent policies.
- No code/config changes were made in this session; record added per AGENTS.md requirement.

## 2025-11-22 (Docs reorg & structural-only policy)

- Moved infra docs into `docs/` (Infra memory for codex or ai agents.txt, DOCS_LURIA.md, memory for Tokernizer.txt) and added `docs/uzr model arc-2.txt` summarizing model.py/memory.py/meta_core.py for structural tokenizer context.
- Updated AGENTS.md to force doc references to `docs/`, note tokenizer docs, and add a codex.memory split policy (chapter 2 file if >30KB).
- README.md/docs updated to point to `docs/DOCS_LURIA.md` and reflect structural-only tokenizer usage (KoEn/KoBERT disabled); infra/tokernizer memos now live under docs/.
- Structural-only tokenizer path reinforced across CLIs/longrun/memory/orchestrator/inspect (KoEn/KoBERT blocked; ByteTokenizer only for 258-row legacy checkpoints).
- Added tokenizer work log template/details to `docs/memory for Tokernizer.txt`.

## 2025-11-22 (Tokenizer: TCodebook default, KoBERT legacy)

- codebook.py: Updated TCodebook defaults to Gt=4/Kt=256 (1024 structural symbols) to align with the structural tokenizer plan.
- utils/struct_tokenizer.py: Added a sliding-window TCodebook tokenizer (one-way, ids→struct-tokens only) with configurable window/stride and reserved specials; exposes vocab size (offset + 4×256).
- train_meta_3brains.py: Default tokenizer is now the TCodebook structural tokenizer (`--tokenizer auto|tcode`), with new knobs `--tcode_window/--tcode_stride/--tcode_gt/--tcode_kt`; KoBERT is marked legacy and only used when explicitly requested, and the resolved tokenizer choice is saved to checkpoints.
- chat_cli.py: load_checkpoint/_pick_tokenizer now understand structural/TCodebook checkpoints (and legacy KoBERT when explicitly tagged) instead of auto-selecting KoBERT; adds Path-based resolver for legacy kobert_dir.
- docs/datasets_tokenizer.md: Documented the new structural tokenizer as default and clarified KoBERT is legacy/opt-in only.
- follow-up: KoEn/KoBERT are now disabled across training/chat/longrun paths (structural-only selection enforced), CLI Luria/longrun CLIs use TCodebookTokenizer, and a dedicated memo file `memory for Tokernizer.txt` documents tokenizer workflow notes.

## 2025-11-22 (Docs: infra memory guide & codex memory repair)

- Infra memory for codex or ai agents.txt: Added a concise infra/memory guide for AI agents (AGENTS.md rules, key entrypoints, commands for training/chat/rest/mem_server, NPU/QNN pointers, data/tokenizer notes, and do-not-touch areas).
- codex.memory.json: Rebuilt the previously corrupted file into valid JSON, preserving all existing facts/commands/todos; fixed the intent top-k string newline, and added facts noting the new guide plus the repair (with a note about the unrecoverable 2025-10-26 fragment).
- Rationale: Give agents a single orientation sheet and restore the long-term memory file so downstream tools can load policies again.
- Compatibility: Documentation/metadata only; no runtime behavior changes.

## 2025-11-15 (Docs: UZR README overview)

- README.md: Added a top-level `uzr/README.md` that gives a full-structure overview of the UZR stack (3brains model, identity intent, CompressedMemory, transition module, training loop, logging, datasets/tokenizers, NPU/QNN engine, REST/CLI) while explicitly excluding exaone-specific adapters/orchestrators from the narrative.
- README.md: Linked together existing docs (`docs/*.md`, `docs/DOCS_LURIA.md`, `LURIA_LOGGING_README.md`, `agent_lora_qnn.md`, `QNN support manual.txt`) so humans and agents can navigate from a single entry point and understand how policies described in the manuals map onto concrete files.
- README.md: Prefixed the document with an explicit research-prototype disclaimer (no guarantees, minimal maintenance, forking encouraged) and a compact ASCII schematic that explains, in one glance, what the Luria 3brains runner is trying to do and roughly how reliable it is intended to be for complex QA/decision tasks.
- Rationale: Provide a single, long-form README in Korean that describes everything implemented in `uzr/` except exaone integration, so both people and LLM agents can quickly reconstruct the architecture and find the right files without scanning the whole tree.
- Compatibility: Documentation-only change; no runtime behavior or public interfaces were modified.

## 2025-11-13 (REST: OpenAI-style UZR server)

- uzr_rest_server.py: Added a minimal HTTP server exposing OpenAI-style `/v1/chat/completions` and `/v1/completions` endpoints around `ChatSession` + `uzr_3brains_ckpt_best.pt` (or any UZR checkpoint). Supports per-request `temperature`, `top_p`, and `max_tokens`, and can optionally hydrate from `mem_server.py` via `--mem_on`/`--mem_url`/`--mem_k`/`--mem_project`.
- Rationale: Allow remote MMLU/QA-style benchmarks to talk to UZR over REST using standard OpenAI-compatible clients, and to test long-term memory when coupled with the external memory gateway.
- Compatibility: Optional additive entry point (`python -m uzr.uzr_rest_server`); existing CLIs and training scripts are unchanged.

## 2025-11-13 (Bench: Avalanche continual learning on MMLU)

- uzr_avalanche_mmlu.py: Added a benchmark script that uses UZR encoder embeddings on `dataset-mit/mmlu_KO-KR.csv` to build an Avalanche continual learning scenario (one experience per MMLU subject) and trains a simple linear classifier head with the `Naive` strategy.
- uzr_avalanche_mmlu.py: Relaxed Avalanche import guard to show the full traceback on failure (not just a generic message), making it easier to debug environment/dep issues when `avalanche-lib` is partially installed or missing submodules.
- Rationale: Provide a ready-made Avalanche integration so UZR can be evaluated under continual learning benchmarks (subject-incremental MMLU) without changing the core model or memory code.
- Compatibility: Optional; depends on the external `avalanche-lib` package. Existing training/inference paths are unaffected.

## 2025-11-13 (Train: gradient checkpointing toggle)

- train_meta_3brains.py:358, 365 — Added `--gradient_checkpointing/--grad_ckpt` flag and environment toggle `UZR_GRADIENT_CHECKPOINTING=1` when enabled; prints a short notice at startup.
- model.py:215-240 — TinyEncoder now supports layer-wise gradient checkpointing using `torch.utils.checkpoint`. It activates only when the env toggle is on, in training mode, and when inputs require grad; falls back to the original path otherwise.
- Rationale: Reduce activation memory during training large `n_layer`/`d_model` runs without changing model semantics.
- Compatibility: Off by default. Enable with `--gradient_checkpointing` on training runs.

## 2025-11-13 (Train: length‑mix shuffle)

- train_meta_3brains.py: Added `shuffle_mix_by_length()` and applied it to both context and query pairs right after sampling (and replay injection). It splits by median length of `len(x)+len(y)` and riffle‑merges short/long buckets to mix sequence lengths per batch.
- Rationale: Prevents prolonged runs of only short or only long samples, stabilizing inner‑step dynamics so `s_max` can flex freely without sudden collapses when a long batch appears.
- Compatibility: No API changes; pure reordering within the sampled pairs.

## 2025-11-13 (Train: FP16 flag)

- train_meta_3brains.py: Added `--fp16` flag. When set, enables CUDA autocast with `dtype=torch.float16` and turns on GradScaler. This reduces activation memory; parameters remain in FP32 for stability.
- Rationale: Provide an explicit memory‑saving option when VRAM is tight, separate from generic `--amp`.
- Compatibility: Off by default. Can be combined with `--gradient_checkpointing`.

## 2025-11-11 (Bugfix: EMA std calc for B=1)

- model.py:848-865 — In `update_ema_stats`, compute std with `unbiased=False` and guard non-finite values before updating EMA. This avoids PyTorch's DoF<=0 warning and NaN propagation when batch size is 1. Adds a small floor `+1e-5` as before.
- Rationale: Prevent initialization-time NaNs in `ema_std` that destabilize z normalization and training when small batches occur.
- Compatibility: No API changes; behavior matches previous for large batches, with improved numerical safety.

## 2025-11-11 (Train: abstain-weighted base loss)

- train_meta_3brains.py:1310-1330 — Combine task CE and Brier as base loss and apply a soft weight derived from `maybe_abstain(conf, ent, thresholds)` per answer by pro.txt. The weight is `(1 - mask) + 0.2` averaged over the batch, so low‑confidence/high‑entropy samples contribute less without hard-skipping. Auxiliary losses (identity, transition, force-write) remain unweighted. Thresholds come from `load_meta_config()` via `_thr = AbstainThresholds(...)`.
- Rationale: Encourage cheaper learning on uncertain samples while keeping gradients flowing; aligns with the manual’s “값싼 학습” guidance.
- Compatibility: No interface changes. Optimizer already includes `model.self_eval` params via `model.parameters()`.

### Also
- Periodic auto-tuning (every 200 steps): adjust `_thr.conf_min`/`_thr.ent_max` toward stricter thresholds when `rule_acc < 0.85`, relax otherwise; and bump memory `write_per_100` via `mem.set_policy_thresholds(...)` when recent write rate is below 6, per manual’s guidance.

## 2025-11-11 (Train: replay injection + reflection)

- train_meta_3brains.py:1009-1030 — Inject replayed failures into query pairs with probability `replay_ratio` (meta default 0.25). Samples up to 4 items via `ReplayBuffer.sample()` and appends as extra Q pairs, providing cheap exposure to recent mistakes.
- train_meta_3brains.py:1978-1988 — Add `ReflectionLogger` and write periodic memos every 200 steps with top failure categories and metrics `{acc, ema}`. Output files live under `reflection/`.
- Rationale: Aligns with manual’s “리플레이·리플렉션” loop to balance curricula and provide auto-tuning hints.
- Compatibility: No API changes; variable batch size for query pairs is supported by the existing loop.

## 2025-11-09 (Memory: gating relaxation & safety tweaks)

- memory.py: Relaxed write surprise clamp to [0.18, 0.40], nudged the base threshold to 0.41, and raised dup/merge/stage knobs (dup_skip=0.95, near_merge=0.84, stage_mid_low=0.73) so the gate covers a broader but safer band.
- memory.py: Increased warmup_steps to 200, lowered rebalance safeguards (`shadow_too_small` ≤ 20, `min_shadow_to_promote`=40), and tightened dedup skip limit to 0.95 to keep promotion attempts flowing earlier.
- memory.py: Documented the new range, updated policy setter defaults, and ensured state serialization rehydrates the refreshed values.

## 2025-11-09 (Model: identity intent subspace)

- model.py: Default `identity_self_dim` is now 32 (identity embedding frozen) and automatically splits into a 16-d intent head that produces rule/think gates; the fusion block consumes both the core identity slice and the intent-derived signals via new projections.
- model.py: Identity intent now drives an abstain bias/toggle head plus multi-time-scale self-referential adapters (fast vs slow EMA) that modulate rule/think/fused z; `update_self_referential()` keeps the slow path synced after each optimizer step.
- train_meta_3brains.py: Exposed `--identity_intent_dim` (default 16, auto-clamped) so checkpoints record the split; scaled runs keep the ratio and the argument is passed through to `UZRModel`.
- train_meta_3brains.py: The rule-based/autonomous abstain logic now reads the identity intent toggle (force on/off) and bias (shifts `tau_r_adjusted`), giving the model an internal switch for abstain decisions.
- chat_cli.py, infer_longrun.py, infer_longrun_standalone.py, mem_server.py: Updated checkpoint fallbacks to the 32-d identity and forward optional `identity_intent_dim` for consistent inference defaults.
- memory.py: `add_with_policy` honors `luria_intent_force` (from meta or `set_write_intent`), bypassing warmup/entropy/dup/rate gates when intent is `True` and skipping immediately when `False`; max_items defaults to 32k and all primary entrypoints (train_meta_3brains, chat_cli, infer_longrun*, mem_server, orchestrator, CLI) pass the identity intent toggle before writing.

## 2025-11-09 (Tasks: 15× rule expansion)

- tasks.py: Introduced marker/repeat/rotate/vowel/digit helper rules plus 400+ deterministic prefix/suffix factories; base pool now keeps the legacy anchor entries for compatibility while expanding the remainder parametrically.
- tasks.py: Added Korean digit→sino/verb/batchim markers and English -ing/vowel-shift/short-word rules, wiring them into `RULE_FACTORIES_KO/EN` so the few-shot generator has substantially more language-aware templates.
- Impact: `sample_task` API/flags stay the same, but available transformations scale beyond 15× without touching downstream callers.

## 2025-11-08 (Bugfix: z_for_q initialization ordering)

## 2025-11-08 (Feature: KoBERT tokenizer + dataset-mit auto)

- utils/kobert_tokenizer_lite.py: Added a lightweight KoBERT-compatible tokenizer. Uses local SentencePiece model when available; otherwise falls back to a SPIECE-style heuristic. Exposes PAD/BOS(EOS)/UNK ids and `encode`/`decode` for training.
- train_meta_3brains.py:
  - CLI: `--tokenizer {auto,koen,kobert}` (default `auto`). Auto selects KoBERT when local `kobert/` exists.
  - Tokenizer selection wired so model `vocab_size` follows chosen tokenizer.
  - dataset-mit: Auto-enable mixing (`dataset_mix_prob=0.35`) when local CSV exists and user did not specify a value. Honors existing `--kobert_*` hint flags.
- Intent: Use local `kobert` and `dataset-mit` in training by default when present, without breaking existing runs. Improved path resolution to find `uzr/kobert` when running from project root.

## 2025-11-08 (Compat: KoBERT PAD id in losses/metrics)

- model.py: Added `UZRModel.set_tokenizer_specials(pad_id,bos_id,eos_id)` and made `brier_from_logits_conf` use model's `pad_token_id` by default.
- train_meta_3brains.py: Pass `ignore_index=tok.PAD` to all CE/brier/confidence proxy calls; set model specials from tokenizer after construction.
- Rationale: KoBERT `[PAD]` is not 0, so masking must use tokenizer-provided PAD id. Ensures memory, model, and codebook paths remain consistent with subword tokenization.
- Compatibility: Backward-compatible; explicit flags override auto-detection.

- train_meta_3brains.py:1040-1100,1120-1160 — Re-ordered 3brains adaptation so that `z_for_q`/`z_bridge_star` are always computed before transition collection. Moved multi-sample adaptation out of a `try/except ... else` block and into the main path alongside the `--single_z` branch; reduced the transition section to `try/except` only. Fixes `UnboundLocalError: z_for_q` when the transition collection raises and skips the `else` branch.
- Intent: Guarantee `z_for_q` is defined prior to use in `logits_q = model(Xq, z_for_q)`, and ensure `z_bridge_star` exists before `update_ema_stats`.
- Compatibility: Behavior unchanged apart from improved stability (no training logic or hyperparameters altered).

## 2025-11-06 (Tasks: dataset-mit + KoBERT 힌트 연동)

- tasks.py:430-620 — `DatasetMiTSampler`와 `KoBERTTeacher`를 추가해 dataset-mit(MMLU KO) QA 샘플을 로드하고 필요 시 KoBERT masked-LM 힌트를 입력 컨텍스트에 주입. `sample_task`는 `dataset_sampler` 인자를 받아 확률적으로 실데이터를 반환하므로 기존 랜덤 룰 체인은 기본값 그대로 유지됨(호환성 영향 없음, 명시적으로 mix_prob>0일 때만 활성).
- train_meta_3brains.py:330-360,420-460,919-924 — CLI에 `--dataset_mix_prob/--dataset_mit_path/--kobert_hint/--kobert_dir/--kobert_device/--kobert_max_seq_len`를 추가하고, KoBERT 힌트와 dataset 샘플러를 초기화해 학습 루프에서 `sample_task(..., dataset_sampler=…)`를 호출. 새 옵션 기본값은 기존 동작과 동일하며 opt-in 시에만 추가 계산이 발생.

## 2025-11-06 (Mem server: UZR mirror for real-time L2→UZR)

- mem_server.py: optional in‑process UZR mirror. When env `UZR_CKPT` is set, the server loads a UZR checkpoint and starts a background subscriber that mirrors `UPSERT_NODE` writes into an in‑process `CompressedMemory` using model encoder embeddings.
  - Config: `UZR_CKPT` (required to enable), `UZR_DEVICE` (default `cpu`).
  - Endpoint: `GET /mem/uzr_stats` exposes `{enabled, items, step, max_items}`.
  - Observability: publishes `UZR_MIRROR` events on `/mem/stream` with `{id,status,reason}`.
  - Robustness: mirror is optional; server continues to work without PyTorch if `UZR_CKPT` is unset or load fails.
- codex.memory.json: add `mem_server_uzr_mirror` fact and `run_mem_server_uzr_mirror` command.

추가 확장
- /mem/backfill: 기존 L2 노드를 UZR 메모리로 백필(부팅 시 UZR_BACKFILL_ON_START=1로 자동 수행 가능).
- 타입 가중 미러: Decision/Preference는 즉시 커밋, Episode는 trust≥0.6일 때 커밋.
- Dreamer 링크 완화: `ern` 태그 쌍은 관계 엣지 임계치 0.72로 완화(기본 0.78).

## 2025-11-06 (Orchestrator: UZR-first 자율 컨텍스트 혼합)

- uzr_orchestrator_cli.py: 내부 UZR CompressedMemory에서 top‑k를 검색해 `[MEMORY CONTEXT — INTERNAL]` 블록을 만들고, 외부 메모리 컨텍스트와 혼합하여 프롬프트 앞에 배치.
  - 기본값 `--uzr_first` 활성(내부→외부 순서). 비활성은 `--no_uzr_first`로 전환.
  - `--mix_ctx_k`로 내부 top‑k 크기 조절(기본 mem_k 사용).
  - 외부 메모리 서버가 없거나 느려도 내부 메모리만으로 자율 동작 유지.

추가 보강
- Consciousness Frame 도입: `[SENSORY]`/`[INTENT]` 블록을 포함하여 LLM(의식) 지침을 명시.
- ERN 기록: 매 턴 conf/entropy 요약과 함께 `ern` 태그로 2PC 커밋(게이트 기준 충족 시).
- 토큰 예산 제어: `--int_ctx_max_lines`/`--ext_ctx_max_lines`로 컨텍스트 라인 제한.
- 온라인 학습 적응: conf 높을수록 `train_model` 미니스텝을 2로 상향(기본 1).
- 외부 LLM 장애 시 UZR 폴백: external generate 실패 시 ChatSession을 통해 UZR 경로로 응답 생성(최대 256 토큰) 후 지속 동작.
  - 폴백 토큰 상한 1024로 상향(환경/성능에 따라 조정 가능).
- Consciousness Trace: `logu/conc_trace.jsonl`에 매 턴 SENSORY/INTENT/컨텍스트 라인수/used_ids/conf/ent/폴백 여부 기록.
- Reflection ERN: 커밋 성공 시 간단 자기반성 노드를 추가로 기록(`reflection`,`ern` 태그)하고 원 노드와 연결.

## 2025-11-06 (EXAONE×UZR Memory Gateway MVP)

- Added `mem_server.py`: minimal external memory gateway implementing `/mem/search`, `/mem/write`, `/mem/stream` (SSE).
  - L1: append-only JSONL with blake2b hash-chain at `logus-exaone/mem_events.jsonl`.
  - L2: symbolic graph persisted to `logus-exaone/l2_graph.json` with `etag/version` and 409 conflict handling.
  - Shadow routing: nodes with `trust<0.6` marked `shadow=true` and excluded from default search.
  - ANN: dependency-free toy embedding + cosine; replaceable in future.
- Updated `codex.memory.json` facts/commands to include the memory gateway and run command.
- Dreamer: background consolidation thread added (idle-driven). Creates `Episode` nodes and `relates_to` edges from recent events; logs to `dreamer_stats.csv`; emits `DREAM_START/DREAM_DONE` SSE.
- Docker: added `memory` service in `docker-compose.yml` to run the gateway with dreamer enabled.
- CLI: `chat_cli.py` gains external memory gateway integration (opt-in) — flags: `--mem_on`, `--mem_url`, `--mem_k`, `--primer`. When enabled, it
  - hydrates top-k items from `/mem/search` into a `[MEMORY CONTEXT]` block before the prompt,
  - optionally prepends a session `[PRIMER]` built at start,
  - writes a minimal MAL envelope to `/mem/write` logging the interaction and linking to hydrated items.

## 2025-11-06 (EXAONE integration: adapter + CLI)

- Added `exaone_adapter.py`: pluggable EXAONE generator with `http` backend (POST to EXA endpoint) and optional `hf` backend (local transformers; no hard dep).
- Added `exaone_cli.py`: EXAONE chat CLI with external memory hydration/logging (`--mem_on`, `--mem_url`, `--mem_k`, `--primer`, `--mem_project`). Default `--max_tokens=15000`.
- Adapter HTTP modes: `raw/generate`, `openai-chat`, `openai-completion`, `tgi` with flexible response parsing; supports general LM Studio servers.
- HF backend dynamically caps `max_new_tokens` based on model context (input length aware) with default cap 15k.
- Memory prompting: Both CLIs prepend a short `[MEMORY GUIDELINES]` block to encourage consistent use of retrieved items.

## 2025-11-06 (Mem server: UZR autonomous policies)

- mem_server: add UZR-style autonomous controls
  - Write gate: duplicate skip (`dup_skip_thr`), near-merge staging (`near_merge_thr`), trust-based shadow routing (`trust_shadow_thr`).
  - Shadow bank persistence (`shadow_bank.json`) and promotion via `GET /mem/rebalance`.
  - Per-author rate limiting (`writes_per_min`).
  - Two-phase commit: `POST /mem/prepare` → ticket → `POST /mem/commit`.
- Events enriched with `action` (accepted/staged/skipped/updated) and author.

## 2025-11-06 (Orchestrator: route LLM via UZR path)

- Added `uzr_orchestrator_cli.py`: ensures any external LLM call goes through UZR path.
  - Loads UZR checkpoint for self-eval gating; hydrates memory; builds primer/guidelines.
  - Generates via external LLM (EXAONE/LM Studio) but commits memory with 2PC only if UZR self-eval confidence ≥ `--conf_min` (default 0.65).
  - Supports `--backend http|hf`, `--exa_url`, `--http_mode`, `--exa_model`, `--max_tokens=15000`, `--mem_*` options.
  - New: also writes to UZR internal CompressedMemory via `add_with_policy` every turn (warmup=1, bench=gated by conf), updates state with `update_memory_state`, and trains predictor `train_model(steps=1)` online.
- Multi-tenant memory: both `exaone_cli.py` and `chat_cli.py` support `--mem_project` to isolate by project (e.g., `exaone` vs `uzr`), and server logs now include `author` on write events for SSE filtering.

## 2025-11-06 (Hotfix: z-rule dim mismatch)

- model.py: ensure `get_z_from_memory()` pads/trims returned z to model `z_dim` and normalizes.
  - Adds safety in `_fuse_z` to pad/trim `z_rule`/`z_think` to expected dims before fusion.
  - Fixes runtime error: `mat1 and mat2 shapes cannot be multiplied (1x482 and 738x128)` when memory predictor returns 64‑D z but model expects 128‑D.

Files touched
- mem_server.py: new file
- codex.memory.json: facts/commands appended
- changelog.md: this entry

Compatibility impact
- New optional local service. No breaking changes to existing scripts.

## 2025-11-06 (EXAONE×UZR External Memory plan + ckpt survey)

- Added persistent facts to `codex.memory.json` capturing EXAONE memory gateway plan and UZR mapping.
  - Keys: `exaone_memory_gateway_plan`, `uzr_write_policy_runtime`, `repr_text_usage`.
  - Measurements: `ckpt_best_model_dims` (vocab=11624,d_model=256,max_len=512,z_dim=128,z_think=64,z_lang=32,num_langs=3,identity_self_dim=2,n_layers=4),
    `ckpt_best_memory_stats` (items=1323,shadow=55,learn_fields=('avg_emb','z_slow'),ema_loss≈7.79e-4,key/avg_emb=256D,z_slow=64D,repr_text≈53.4%,meta.desc=707).
- Updated `codex.memory.json` todos with planned items:
  - "EXAONE+UZR 로컬 CLI 통합" (search→generate→write loop, rebalance/stats hooks)
  - "MAL 최소 스펙 매핑" (UPSERT_NODE/ADD_EDGE→make_sketch/meta 변환, etag 충돌 처리)

Files touched
- codex.memory.json: facts appended; todos extended
- changelog.md: this entry

Compatibility impact
- Documentation/metadata only. No runtime behavior changes.

## 2025-11-02 (3brains: scale flag)

- Add `--scale` flag to `train_meta_3brains` to multiply key dimensions by an integer factor (1..4, default 1).
  - Scaled args: `d_model`, `z_dim`, `z_think_dim`, `z_lang_dim`, `identity_self_dim`, `z_slow_lang_dim`, `z_slow_logic_dim`, `z_bridge_dim`, `sem_dim`.
  - Change: `train_meta_3brains.py:325-386` (arg parse and scaling block).
  - Extended: also scales `CompressedMemory.learn_hidden` and `CodebookManager.t_cfg.dt` to keep components aligned.
    - Change: `train_meta_3brains.py:420-430` (CompressedMemory init), `train_meta_3brains.py:449-459` (CodebookManager.init).
- Add `--n_head` and `--n_layer` flags (default 4) for Transformer config; enforce `d_model % n_head == 0`.
  - Change: `train_meta_3brains.py:325-336` (args), `train_meta_3brains.py:472-487` (pass-through), divisibility check near scaling block.
  - Auto recommendation: if flags not provided, set `n_head` to closest divisor of `d_model` near `d_model/64` (clamped to 4..32), and `n_layer = 4 + 2*(scale-1)` (clamped to 4..12).
    - Change: `train_meta_3brains.py:387-403` (auto-rule + guard).
- Adaptive inner-step range flag `--inner_step MIN MAX` added; clamps adaptive step selection and pre-350 top-k mapping to the provided range. `s_base` is set to `inner_steps` clamped in range.
  - Change: `train_meta_3brains.py:332` (args), `train_meta_3brains.py:728-744` (s_min/s_max/s_base init), `train_meta_3brains.py:215-237` (choose_adaptive_steps signature/logic), `train_meta_3brains.py:932-948` (call/update), `train_meta_3brains.py:956-964` (top-k mapping uses range).

## 2025-11-02 (Micro loop: 15× sample-wise gating)

- Micro-level meta-cognition signals added and wired into memory write gate and codebook updates without changing inner-steps.
  - Signals: `margin0`(top1-top2), `ent_var0`(token entropy variance), `mm0`(memory mismatch on query), combined into `micro_risk` [0..1].
  - Gating: logit shift `+ micro_gain_logit*(0.5-micro_risk)` and percentile reduction `- micro_gate_beta*micro_risk` (risk↑ → 선택적/보수적).
  - Codebook: high risk(≥0.6)시 shadow 누적 빈도를 즉시↑(모든 스텝), 평상시는 4스텝 주기 유지.
  - CSV: `micro_risk, margin0, ent_var0, mm0` 추가 기록.
  - Flags: `--micro_off`(비활성), `--micro_gain_logit`(기본 3.0), `--micro_gate_beta`(기본 0.15).
  - Change: `train_meta_3brains.py` 가드/계산/게이팅/CSV 헤더 및 쓰기부 다수 라인.

Compatibility impact
- Backward-compatible; default scale=1 preserves prior behavior. Higher scales increase model capacity and compute proportionally.

## 2025-11-03 (Multimodal Transition: z + codebook)

- Model: add optional multimodal transition module (z/u/codebook) with EMA normalization and dual heads.
  - Added: `CodebookEncoder`, `MMFuse`, `TransHeadZ`, `TransHeadCB` classes; `UZRModel.init_transition_module()`, `update_ema_stats()`, `norm_z()`.
  - Change: `model.py` (new helpers; transition init method) — around class definitions and UZRModel methods.
- Training: integrate transition buffer and losses into 3brains loop (minimal invasive wiring).
  - Buffer: `(z_t, u_t, cb_t) → (z_{t+1}, cb_{t+1})` collected across steps; window=8192.
  - Loss: Δz dir + 1‑step cos + BOW(BCE) + z↔cb align + jacobian surrogate (+ optional rollout placeholder).
  - Changes: `train_meta_3brains.py`
    - Args: add `--cb_recent_len` (default 24).
    - Init: compute `cb_vocab` from `CodebookManager` and call `model.init_transition_module(...)`.
    - Loop: collect transitions after z adaptation; add transition loss when buffer ≥1024.
- Metrics: summary CSV gains transition columns: `z_cos, dz_mse, cb_f1, align_cos, jac_surr, dz_norm, dz_norm_std, cb_pos, cb_pred, trans_loss`.
- Compatibility impact
  - Backward-compatible by default; transition heads are initialized inside `train_meta_3brains.py` and only used there.
  - No changes to existing entry points; main optimizer updates all params (no separate zcb optimizer).


## 2025-10-31 (Choebang-yak: memory policy tuning)

- Promotion gates and minimums adjusted per Choebang-yak
  - Primary surprise cutoff set to p60 with p50 fallback; auto-relax 5p on two consecutive no-promotion cycles: `memory.py:1410`.
  - Added auxiliary channel for plain-but-useful samples with low k3_avg_sim (<=0.58): `memory.py:1450`.
  - Ensure a small minimum of promotion attempts per rebalance (<=3 within budget): `memory.py:1499`.
  - Rebalance CSV logging extended with `eligible_cnt`, `relax_bonus`, and `actual_promoted`; RB_SKIP carries `promote_fail_reason` placeholder: `memory.py:1504`.

- Merge conservatism and cluster requirement
  - Merge requires k=3 cluster presence; disallow 1-N fallback merges by requiring at least 3 items: `memory.py:600`.
  - Default near-merge threshold softened to 0.80 (from 0.90): `memory.py:54`, `memory.py:1568`.

- L2 regularization and key norm scaling
  - Reduce L2 penalty weight 1e-4 -> 5e-5; key_norm_scale 0.85 -> 0.90 for `scaled_l2` mode: `memory.py:73`.

- Surprise gate relaxation by novelty band
  - For 0.30 < sim_max <= 0.45, relax gate further (factor 0.50 -> 0.35) to raise acceptance (~65%): `memory.py:645`.

Compatibility impact
- Backward-compatible; affects only memory write/rebalance behavior. Expect:
  - Fewer merges for small clusters; steadier stage→promote flow.
  - Slightly higher acceptance in mid-novelty band; promotions maintain floor when budget allows.
  - Rebalance CSV gains extra diagnostics; downstream tooling should tolerate added columns.

## 2025-10-31 (Rebalance cutoff + logging, follow-up)

- Rebalance promotion cutoff relaxed p60 → p55; fallback p50 remains when no eligible.
  - Change: `memory.py:1410-1475` (p_primary now 55).
- Rebalance CSV logging extended with diagnostics
  - Added: `eligible_cnt`, `eligible_before`, `eligible_after`, `retry_count`, `slice_len`, `util_ratio`, `cutoff_surprise`, `cutoff_z`, `promoted`, `promote_fail_reason` across RB_START/RB_DONE/RB_SKIP.
  - Change: `memory.py:1398-1408,1470-1612`.
- writes.csv: add surprise diagnostics to explain missing values
  - Added fields: `diag_surp_has_z`, `diag_surp_knn_tried`, `diag_surp_knn_with_z`, `diag_surp_computed`, `diag_surp_reason`.
  - Change: `memory.py:560-760` in add-with-policy `_log_write(..., extra=...)` calls.

- Round-internal recovery and selection upgrades
  - Dual cut for promotions: `cut = max(pctl, mu+z*sigma)` with z=0.25, retry z=0.10.
  - In-round retries: if empty → pctl-5 (p55→p50), then util_ratio 1/3→1/2.
  - Decouple attempts from budget: ensure min attempts (6 with budget, epsilon 4 without), and `slice_len ≥ 2×attempts`.
  - Safety pin improved: pick up to 3 from top-8 under constraints `k3≤0.62`, `recency≥0.5`, `sim_max≤0.92` instead of single top-1.
  - Change: `memory.py:1458-1566,1570-1612`.

Compatibility impact
- Backward-compatible; only reduces promotion cutoff slightly and adds CSV columns.

## 2025-10-31 (UZR-Gaeseon-1: EMA_all, PI, critic-lite)

- EMA_all(평가식 교체)
  - 정의: EMA_all = EMA_raw + λ_abs · abstain_ratio.
  - 스케줄: λ_abs 1.0 → 2.0, 3k~10k 선형 증가(이후 고정).
  - 로깅: summary CSV에 `ema`(=EMA_all), `ema_raw`, `lambda_abs` 추가 기록.
  - 변경: `train_meta_3brains.py:1889-1998` (CSV 라이터), `train_meta_3brains.py:1640-1660` (EMA 계산/대입).

- 수용률 컨트롤러(PI) 도입
  - 목표 수용률 스케줄: 3k:0.15 → 10k:0.35 → 15k:0.45(구간별 선형), 이후 0.45 유지.
  - 200-step 창에서 r(실제 수용률)을 집계하고, e=r-t로 `tau_r_pi` 바이어스를 PI(kp=0.5, ki=0.05)로 갱신.
  - 적용: 자율 구간에서 `tau_r_adjusted = tau_r + tau_r_pi`로 문턱치 조정(클램프 ±0.20), CSV에 `acc_r_200`, `acc_t`, `tau_r_pi` 기록.
  - 변경: `train_meta_3brains.py:654-661, 1323-1366(τ_r 적용), 1666-1684(PI 업데이트), 1913-1998(CSV)`.

- critic-lite(자율합성 기준 주입)
  - R = α·consistency + β·spec_coverage + γ·anti-copy, α=1, β=0.5, γ=0.25.
  - 프록시: consistency=conf_mean, spec_coverage=1−ent/3, anti-copy=1−trigram overlap(Xq,Yq).
  - self-eval pass면 τ_r 상향(+ηR), fail이면 하향(−ηR), η=0.05.
  - 변경: `train_meta_3brains.py:1337-1366` (τ_r 조정 로직에 통합).

호환성 영향
- 요약 CSV 헤더 확장(ema_raw/lambda_abs/acc_r_200/acc_t/tau_r_pi). 기본 학습 경로는 평가지표만 갱신되며, 거부 흡착(runaway abstain) 완화 기대.

## 2025-10-31 (Jeong-sang-hwa: 로그/지표 정렬 2분 패치)

- Summary CSV에 진단용 지표 추가 및 조건 명시
  - `ema_raw99`, `ema_all99`(β=0.99), `mean_raw_200(raw)` 추가.
  - 추가 필드: `mode`(train/eval), `amp`(on/off), `eval_set_id`.
  - 목적: `EMA_all − EMA_raw ≈ λ_abs·abstain_ratio`를 즉시 검증하고, 모드/AMP 혼선 제거.
  - 변경: `train_meta_3brains.py` 헤더와 쓰기부 업데이트.

호환성 영향
- 순수 로그 확장으로 학습 경로에는 영향 없음. 분석 스크립트는 새 컬럼을 허용해야 함.

## 2025-10-30 (Hotfix: CodebookManager.init() commit_steps parameter)

- Add missing commit_steps parameter to CodebookManager.init() static method
  - Fixed TypeError when calling CodebookManager.init() with commit_steps argument: `codebook.py:636`
  - The static factory method now accepts commit_steps parameter (default: 1000) and forwards it to the constructor: `codebook.py:631-640`
  - Resolves error in train_meta_3brains.py:460 where commit_steps was being passed but not accepted

Compatibility impact
- Backward-compatible: existing calls without commit_steps will use default value (1000).
- Calls that were previously failing with TypeError will now work correctly.

## 2025-10-30 (trainer: integrate feels-first/goat into 3brains)

- train_meta_3brains: codebook-driven gates, conserve mode, EMA integrity, ZC handling
  - CSV header expanded with diagnostics and EMA integrity: `train_meta_3brains.py:396-406` (fields: conserve_on, conserve_reason, lr_scale, fw_prob, surprise_gate_delta, ema_min_*)
  - CodebookManager commit cadence aligned to checkpoint interval: `train_meta_3brains.py:438-443`
  - Meta header printed once at start: `train_meta_3brains.py:...` (after best_path init)
  - Conserve mode (3k~6k): LR×0.7, raise surprise gate base +0.05, FW prob→5% with revert: `train_meta_3brains.py:716-748`
  - Codebook entropy/dead-ratio hooks: tau_r+=0.01 when entropy<0.75*uniform, 400-step surprise gate relaxation when dead>0.35: `train_meta_3brains.py:1549-1573`
  - Surprise_eff = 0.7*entropy + 0.3*codebook-rarity used in composite_score: `train_meta_3brains.py:1604-1642`
  - ZC tokens reused for logging, shadow EMA throttled (skip <500, then every 4th step): `train_meta_3brains.py:1660-1665`
  - Coverage-aware LR now also respects conserve mode: `train_meta_3brains.py:1333-1337`
  - EMA integrity tracked (csv min vs ckpt best) and persisted in checkpoints/final save: `train_meta_3brains.py:1743-1756, 1770-1780, 1822-1828`

Compatibility impact
- Backward-compatible defaults. New logs/fields only extend summary CSV and checkpoint payload.
- Memory write thresholds are adjusted temporarily under specific codebook signals; bounded and reverted to avoid drift.

## 2025-10-30 (Docs: apply feels-goat to doremipasol)

## 2025-10-30 (QNN support: ORT EP wiring)

## 2025-10-30 (Chat: ORT/QNN engine + LoRA hot-swap)

## 2025-10-30 (Luria CLI: ORT/QNN + docs)

- cli_luria.py adds optional ORT/QNN NPU inference path
  - New args: `--ort_model`, `--engine {torch,qnn,qnn_strict,ort_fallback}`
  - Engine-backed token generation with PyTorch fallback; adapt_z remains on CPU
  - Commands: `/lora_npz <path>`, `/hot_swap`
- agent_lora_qnn.md: agent/chat/CLI 통합 사용 가이드(LoRA×QNN)

Compatibility impact
- Optional and additive. Default behavior unchanged unless flags are set and onnxruntime-qnn is installed.

- chat_cli integrates optional ORT/QNN engine for NPU inference
  - New args: `--ort_model`, `--engine {torch,qnn,qnn_strict,ort_fallback}`
  - Engine-backed generation path with PyTorch fallback
  - Commands when engine active:
    - `/lora_npz <path>`: load adapter_A/B and film_gamma/beta from npz and swap into engine
    - `/hot_swap`: recreate and warm up session (shadow→active)
- model.py: added optional NPU engine hooks (`set_npu_engine`, `npu_run_logits`)

Compatibility impact
- Optional. If onnxruntime-qnn is absent or flags not provided, behavior is unchanged.

- Added minimal QNN support per QNN manual v1.0
  - New package `npu/` with runtime and engine wrappers:
    - `npu/runtime_ort.py`: QNN strict session, QNN→DML→CPU fallback session, context cache, profiling options
    - `npu/engine.py`: ORT-based engine with adapter/FiLM hot-swap inputs and shadow→active hot swap
    - `npu/__init__.py`: public exports
  - No training path changes; designed to be imported by inference code.

Compatibility impact
- Requires onnxruntime or onnxruntime-qnn installed on target device. Code guards import and raises a clear message.
- Does not affect existing PyTorch training/inference paths unless imported.

- Applied feels-goat formatting to report
  - `doremipasol.txt:1-128` now includes Meta line, Key Metrics summary, Diagnostics (<=5 lines), Plots (2 in body), and one-line Takeaway.
  - Kept existing patch details; added structure without removing content.

Compatibility impact
- Docs-only; improves consistency and reviewer speed.

## 2025-10-30 (Docs: feels-goat guidelines cleanup)

- Rewrote corrupted guideline into a clean, actionable KO document
  - `feels-goat.txt:1-123` rewritten (mojibake → structured "수정·보강 지침").
  - Sections: A(구조) B(내용 보강) C(코드 연동) D(로깅 스키마) E(문장 에디트) F(부록 제안).
  - Added Meta-line template, Best EMA Integrity check, conserve-mode pseudo-code, 6 code linkage suggestions, rebalance/EMA/conserve logging fields, and editing rules.

Compatibility impact
- Docs-only; no code changes. Improves reviewer alignment and report consistency.

## 2025-10-30 (Fix abstain loop; raise near-merge; warmup status)

- Abstain runaway fix (sign and bounds)
  - Emergency handler now raises `tau_r` instead of lowering it, with clamp to [0.50, 0.90]: `train_meta_3brains.py:1315`.
  - Length bonus corrected: for `seq_len > 64`, threshold increases by +0.02 (harder to abstain): `train_meta_3brains.py:1284`.

- Autonomous start step made configurable
  - New flag `--autonomous_step` (default 3000) controls when autonomous gating/threshold adaptation begins: `train_meta_3brains.py:340`.
  - `use_autonomous_abstain` gating and display updated to use `args.autonomous_step`: `train_meta_3brains.py:690`, `train_meta_3brains.py:1565`.

- Memory thresholds retuned to restore stage→promote flow
  - near-merge threshold centered at 0.88 before autonomous; learnable param clamped to [0.82, 0.95] after: `train_meta_3brains.py:692`.
  - Optim param for near-merge now initialized to 0.88: `train_meta_3brains.py:525`.
  - `dup_skip_thr` raised to 0.97 and `stage_mid_low` set to 0.60 via `set_policy_thresholds(...)`: `train_meta_3brains.py:703`.

- Warmup return status clarified
  - Warmup branch logs `action="defer"` and now returns `status="deferred"` (was `"staged"`), to avoid confusion: `memory.py:546`.
  - add_with_policy docstring updated to include `deferred`: `memory.py:505`.

Compatibility impact
- Backward-compatible; behavior changes are guarded by defaults (`--autonomous_step=3000`).
- Memory growth expected to increase (fewer merges, more stage/promote); abstain ratio expected to drop below 0.6 under smoke test.

### 2025-10-30 (Hotfix: promote path opens per "fixa now")

- Rebalance/promote hotfix (p60 gate with fallbacks, minimums, safety pin)
  - Surprise gate relaxed from p75 → p60; fallback to p50; if still empty, take top-k by surprise: `memory.py:1396-1440`.
  - Minimums: `min_shadow_to_promote=200` (skip only if no candidates), `min_candidates=max(16, int(0.05*shadow_size))`: `memory.py:1442-1460`.
  - Cap attempts per window to 8 (also respect budget when >0); ensure at least 1 attempt via safety pin when `shadow_size>80`: `memory.py:1462-1497`.
  - Add rebalance log CSV with events `RB_SKIP/RB_START/RB_DONE` recording reasons and counts: `memory.py:1370-1380,1499-1504`.

- Scheduling alignment
  - Trainer continues to call `mem.rebalance()` every 50 steps; inside memory, we skip with reason when `shadow_size<=50`.

- Near-merge threshold and autonomous deferral
  - Pre-autonomous near-merge fixed at 0.90 to preserve stage path: `train_meta_3brains.py:694`.
  - Default `--autonomous_step` deferred to 3500 for short-run stability: `train_meta_3brains.py:340`.

### 2025-10-30 (Adjust: promote intensity per "adjust now")

- Surprise gate slightly tightened
  - Primary surprise cutoff raised p60 → p65; fallback remains p50: `memory.py:1438-1454`.

- Promotion attempts reduced
  - Cap per rebalance window lowered 8 → 6; still respects budget when >0: `memory.py:1486-1490`.

Intent
- Stabilize promote behavior and damp oscillations while keeping flow open via fallback.

Compatibility impact
- Backward-compatible; only affects gating and attempt caps. Expect slightly fewer promotions in high-surprise regimes.

## 2025-10-28 (mem_size≈2 원인 분석 - 코드 변경 없음)

- mem_size 정의와 관측 지점
  - writes.csv에 기록되는 `mem_size`는 커밋된 메모리 항목 수(`len(self.items)`)이다: `memory.py:201`.
  - 학습 요약 CSV에서도 `"mem_size": len(mem.items)`로 기록된다: `train_meta_3brains.py:1461`.

- 초기 단계에서 mem_size가 작게 유지되는 주된 원인
  - 웜업 차단: 스텝이 `warmup_steps`보다 작으면 기록을 보류(`staged`) 처리한다: `memory.py:544`.
  - 레이트 리밋 웜업 램프: `_rate_limit_ok`에서 유효 예산이 `int(write_per_100 * ramp)`로 축소되며, `ramp = step / _warmup_ramp_steps(=1000)`이므로 초반 100~300스텝 구간은 100스텝당 0~3건 수준으로 제한된다: `memory.py:399, 760, 771`.
  - 유사도 기반 병합/스테이징: `near_merge_thr` 이상은 merge(크기 변동 없음), `stage_mid_low` 이상은 shadow_bank으로 스테이징(커밋 아님), `dup_skip_thr` 이상은 skip 처리: `memory.py:599, 620, 627`.
  - 엔트로피 게이트: `entropy_check_start` 이후에만 적용되고, floor 설정에 따라 보류될 수 있다(학습 스크립트는 보통 floor=0.0): `memory.py:549-550`.
  - 승격 의존: 스테이징만 쌓이면 `rebalance()` 호출에 의존해 프로모션이 일어난다. 학습 루프에는 주기 호출이 있으나: `train_meta_3brains.py:1374`.

- 실행 경로별 참고
  - 학습 스크립트: `mem.set_policy_thresholds(...)`, `mem.softmax_temp=0.8`, `setattr(mem, "entropy_check_start", 32)` 등 완화가 적용되어 있으나, 위 ‘레이트 리밋 램프’ 영향으로 초반 mem_size는 수 건 수준에 머물 수 있음.
  - CLI(Luria): 기본 생성자 사용으로 `warmup_steps=100`, `_warmup_ramp_steps=1000`, 쓰기 호출도 사용자 상호작용(/saved, /yes) 시에만 발생하여 mem_size가 아주 천천히 증가할 수 있음: `cli_luria.py:361`.
  - 안전 스위치: 환경변수 `UZR_MEM_WRITE=0`이면 전부 보류 처리한다(진단 체크 권장): `memory.py:517`.

- 권장 점검/튜닝(행동 지침)
  - writes.csv의 `action/reason`, `shadow_size`, `mem_size`를 함께 확인해 보류/스테이징 비율을 진단.
  - 초기 가속이 필요하면: `warmup_steps` 축소, `_warmup_ramp_steps` 축소(예: 200~300), `write_per_100/tail_write_per_100` 상향, `stage_mid_low` 소폭 하향, `near_merge_thr` 소폭 상향.
  - `rebalance()`를 주기적으로 호출하는지 확인(스테이징만 쌓이는 상황 방지).

Compatibility impact
- 코드 변경 없음. 문서/진단만 반영.

## 2025-10-28 (L2 regularization and dynamic surprise threshold)

- L2 normalization with configurable modes
  - Added `key_norm_mode` ("unit" or "scaled_l2") and `key_norm_scale` (default: 0.85): `memory.py:128-131`.
  - Replaced hard-coded F.normalize with `_apply_l2_norm(tensor, target_norm)` method: `memory.py:273-295`.
  - Supports both unit norm (L2=1.0) and scaled norm (L2=0.85) for keys and z_slow: `memory.py:435-437`.
  - Added `_l2_penalty(tensor)` for computing regularization loss with weight 1e-4: `memory.py:297-301`.
  - L2 penalty logged in writes.csv for monitoring: `memory.py:196,523-524,530-531,545-546,563-565,574-576,588-590`.

- Dynamic surprise threshold with EMA-based adaptation
  - Added `_surprise_threshold_ema` tracked via `_update_surprise_threshold(current_surprise)`: `memory.py:132-133,367-378,493`.
  - `_get_surprise_threshold(step)` computes adaptive threshold: base + alpha * surprise_ema * sigmoid_ramp: `memory.py:380-407`.
  - Formula: threshold = 0.40 + (-0.20) * surprise_ema * ramp, clamped to [0.20, 0.80]: `memory.py:136-140,396-406`.
  - Sigmoid-based warmup ramp (k=12.0) for smooth threshold activation over first 1k steps: `memory.py:143,303-308,390-393`.
  - Write-on-surprise gate now uses dynamic threshold instead of fixed percentile: `memory.py:534-540`.
  - Surprise threshold logged in writes.csv: `memory.py:197,521,529,545-546,563-565,574-576,588-590`.

- Enhanced logging schema
  - Extended writes.csv with `l2_penalty` and `surprise_threshold` columns: `memory.py:179-180,196-197`.
  - Extended rollbacks.csv with same fields: `memory.py:585-586`.
  - All commit/merge/skip/rollback actions now log L2 penalty and current surprise threshold.

- State persistence
  - Added L2 and dynamic threshold parameters to "lekiltan" state_dict section: `memory.py:1053-1066`.
  - Restore with safe defaults in load_state_dict: `memory.py:1196-1209`.

Compatibility impact
- Backward-compatible: older checkpoints load with default values (scaled_l2 mode, scale=0.85).
- Logging schema extended: writes.csv and rollbacks.csv gain l2_penalty and surprise_threshold columns.
- Dynamic threshold adapts to surprise signal strength, lowering threshold when surprise is high (more permissive writes).

## 2025-10-28 (Lekiltan enhancements: improved instrumentation and control)

- Enhanced entropy measurement and logging
  - `_retrieval_entropy` now returns tuple (entropy_normalized, entropy_raw): `memory.py:187-226`.
  - Soft clipping with tanh instead of hard clipping for smooth normalization: `memory.py:207-215`.
  - Extended `_log_write` to include `entropy_raw` field: `memory.py:149-182`.
  - Updated entropy.csv to log entropy, entropy_raw, and entropy_max: `memory.py:220-225`.
  - All add_with_policy log calls now include entropy_raw: `memory.py:357,407,412,430,442-443,452-453,463-464`.

- Warmup ramp for gradual policy activation (0→1000 steps)
  - Added `_warmup_ramp_steps` (default: 1000) to __init__: `memory.py:125`.
  - `_warmup_ramp_factor(step)` returns linear ramp [0.0, 1.0]: `memory.py:255-259`.
  - Write budget (write_per_100, tail_write_per_100) scales with ramp in `_rate_limit_ok`: `memory.py:490-512`.
  - Prevents initial over-reaction during first 1k steps.

- Surprise alert mechanism (EMA-8 based)
  - Added `_surprise_ema8` state with alpha=0.25 (≈8-step window): `memory.py:121`.
  - `_update_surprise_ema(surprise)` updates EMA on each surprise computation: `memory.py:261-267,383`.
  - `_check_surprise_alert(step)` activates 8-step alert when EMA exceeds median+1σ: `memory.py:269-283`.
  - Alert state logged in extra fields (alert_active) for 2PC and commit logs: `memory.py:435,442-443,452-453,463-464`.
  - Serves as "early warning" signal for local adjustments (not a hard gate).

- Z-score local dampen hook (model entropy based)
  - Added `_model_entropy_window` (256-step rolling window): `memory.py:122`.
  - `_update_model_entropy_window(model_entropy)` tracks model entropy from meta: `memory.py:285-289,393-394`.
  - `_check_zscore_dampen(step)` activates 16-step dampen when |z-score| >= 3.0: `memory.py:291-307`.
  - Dampen reduces write_per_100 by 2 and tail_write_per_100 by 1 during active period: `memory.py:492,497-498,506-508`.
  - Local, temporary response to entropy spikes.

- State persistence for Lekiltan features
  - Added "lekiltan" dict to state_dict with surprise_ema8, model_entropy_window, alert_active_until, dampen_active_until, warmup_ramp_steps: `memory.py:920-927`.
  - Restore in load_state_dict with safe defaults: `memory.py:1047-1057`.

Compatibility impact
- Backward-compatible: older checkpoints without "lekiltan" state load normally with initialized defaults.
- Logging schema extended: writes.csv gains entropy_raw column, entropy.csv gains entropy_raw and entropy_max.
- Alert/dampen states are optional meta signals; runners can provide model_entropy in meta for full functionality.

## 2025-10-26 (CLI fixer: memory persistence)

- Persist memory in CLI checkpoints and restore on load
  - Save `CompressedMemory.state_dict()` in session checkpoints: `cli_luria.py:176`.
  - Restore memory when resuming via `--resume`: `cli_luria.py:260`.
  - Restore memory on `/load` command when present: `cli_luria.py:401`.

- Add train-compatible last snapshot on save/exit/autosave
  - New helper `save_train_compatible_last(...)`: `cli_luria.py:193`.
  - `/save` writes both session and `uzr_3brains_ckpt_last.pt`: `cli_luria.py:329`.
  - Autosave now writes both formats and logs: `cli_luria.py:288`.
  - Exit/signal hooks always write latest snapshots: `cli_luria.py:295`.

- Strengthen autosave coverage
  - Unified `maybe_autosave()` and invoked after step increments in `/saved`, `/yes`, and predict paths: `cli_luria.py:527`, `cli_luria.py:581`, `cli_luria.py:617`.

Compatibility impact
- Backward-compatible; older checkpoints without `memory` load normally. When present, memory is restored.
- Produces additional `uzr_3brains_ckpt_last.pt` file in CWD to simplify training resume (`--resume uzr_3brains_ckpt_last.pt`).

- Memory runtime history serialization (optional context)
  - `memory.py`: include `rt_state` with `session_state`, `state_history` (tail), and `input_history` in `state_dict`: `memory.py:810`.
  - Restore `rt_state` in `load_state_dict` with tensors stored on CPU: `memory.py:900`.

## 2025-10-25 (CLI teaching docs)

- Verify teaching docs alignment with CLI implementation
  - uzr-cli-teaching-step.txt: Provides minimal Chat CLI skeleton and workflow; content matches current capabilities.
  - uzr-cli-teaching-step-final.txt: Lists 6 critical fixes; all are present in `uzr/cli_luria.py`.
    - None-safe conf/entropy formatting: `cli_luria.py:23-25`.
    - `torch.load` weights_only fallback: `cli_luria.py:222-228`, `cli_luria.py:291-294`.
    - Atomic save with timestamped default path: `cli_luria.py:27-32`, `cli_luria.py:172-186`.
    - `/load` command: `cli_luria.py:285-301`; `/save`: `cli_luria.py:279-283`.
    - Robust `/saved` parser (first '=' split, quote strip): `cli_luria.py:188-203`.
    - Autosave hook: `cli_luria.py:501-504`.
  - CSV logs include tokens/conf/entropy, matching docs: print and logging at `cli_luria.py:479-486`, `cli_luria.py:483-486`.
  - No code changes required; documentation consistent with current code.
  - Compatibility: No impact.

## 2025-10-21 (meta-cognition)

- Add meta cognition module and config (no modifications to existing code)
  - meta_core.py: SelfEvalHead (confidence), abstain policy, error tagger, reflection logger.
    - New APIs: `SelfEvalHead`, `maybe_abstain`, `tag_error`, `ReplayBuffer`, `ReflectionLogger`, `meta_step()`.
  - config/meta.json: default thresholds and knobs (lambda_brier, conf_min, ent_max, etc.).
  - Scope: standalone utilities; callers can import without touching `model.py`/runners.

- Integrate SelfEval into model and expand infer logs
  - model.py:
    - Optional `self.self_eval` (initialized when available) and helpers: `confidence(x)`, `sequence_entropy(logits)`, `brier_from_logits_conf(...)`.
    - Backward-compatible; `forward` unchanged.
  - infer_longrun.py / infer_longrun_standalone.py:
    - Add SelfEval-based metrics to CSV: `conf_self_c/q`, `ent_c/q`, `brier_c/q`, `abstain_c/q`.
    - Memory meta now includes `conf_self_*` and `ent_*` snapshots.

- Logging quality-of-life
  - infer_longrun.py / infer_longrun_standalone.py / infer_longrun_standalone_logged.py:
    - Default CSV path auto-set to `logu/<YYYYMMDD_HHMMSS>_s{inner}_t{turns}_{ckpt_stem}.csv` when `--summary_csv` not provided or default.
    - Ensures `logu/` directory exists.

- Inner-step adjust by model (dynamic budget)
  - meta_core.py: add `inner_steps_from_conf(conf, s_max, s_min, k, mid)`.
  - infer_longrun.py / infer_longrun_standalone.py:
    - Inner loops use dynamic `chosen_steps` computed from initial confidence; early stop when confidence >= 0.8.
    - CSV adds: `conf0`, `chosen_steps`, `tries`, `best_conf`, `gate_pass`, `compute_tokens`.
    - Memory meta carries `conf0/chosen_steps/tries/best_conf` for traceability.

- SelfEval toggle (wrapper flag + env)
  - infer_longrun_standalone_logged.py: add `--self_eval {on,off}` to enable/disable SelfEval.
  - model.py: respects `UZR_SELF_EVAL` env to construct or skip `SelfEvalHead`.

- Logged runner wrapper refresh
  - Replaced broken standalone logged script (malformed `--identity` arg) with a thin wrapper that delegates to `uzr.infer_longrun_standalone`.
  - Wrapper maps `--turn`→`--turns`, passes through args, sets `UZR_SELF_EVAL`, and auto-names summary CSV under `logu/`.

- Logged-use runner updates
  - infer_longturn-logged-use.py:
    - Add `--self_eval {on,off}` flag and propagate via `UZR_SELF_EVAL` env.
    - Auto-name summary CSV under `logu/<ts>_s{inner}_t{turns}_{ckpt}.csv` when default path is used.
    - Dynamic inner-step scheduling using initial confidence with early stop; CSV extended with `conf0, chosen_steps, tries, best_conf, compute_tokens`.

- Korean challenges
  - infer_longrun_standalone.py:
    - Append two Korean tasks (index tagging and length sort) to challenge suite.
    - Relax suite size assertion to allow >=20.

- Train script alignment (3brains)
  - train_meta_3brains.py:
    - Integrate meta_core: add SelfEval toggle (`--self_eval`), abstain gating (`--abstain`), and Brier regularizer added to loss.
    - Log conf/entropy/brier to new CSV (`--summary_csv`, auto-named under `logu/` when default).
    - Memory writes use `add_with_policy(...)` and periodic `rebalance()`; meta includes conf/ent/brier.
    - Resume loads model state with `strict=False` for compatibility with older checkpoints.

## 2025-10-21

- Add AGENTS.md and codex.memory.json
  - AGENTS.md: 작업 원칙, 코드 스타일, 메모리 사용 규칙 추가.
  - codex.memory.json: 기본 스키마 생성(`preferred_language: ko`).

- Long-term memory policy (3brain manual) integration
  - memory.py:
    - add_with_policy: write-on-surprise, shadow bank(1k), near-dup merge/skip, entropy floor, rate limiting.
    - rebalance: 중복 정리 및 보류 승격.
    - 로깅 지원: writes.csv, entropy.csv (log_dir 설정 가능).
    - state_dict/load_state_dict에 정책 상태 포함.
  - infer_longrun.py / infer_longrun_standalone.py:
    - 메모리 쓰기를 add_with_policy로 교체, 메타(desc, ce_q, conf) 전달.
    - 주기적으로 rebalance 호출.

- 3brains 차원 불일치(size mismatch) 호환 로더
  - infer_longrun.py: 모델 생성 시 cfg의 3brains 파라미터 적용.
  - infer_longrun_standalone.py:
    - cfg 기반 생성 + `fuse_proj_3brains.weight` 입력 차원에서 차원 유추 폴백.

- 실행 래퍼 추가
  - infer_longrun_standalone_logged.py: `--turn` → `--turns` 매핑 및 import 경로 보정.

- Top-level script hotfix (parent dir)
  - ../infer_longrun_standalone_logged.py: 체크포인트의 `fuse_proj_3brains.weight` 크기와 불일치 시, 가중치 입력 차원에서 3brains 차원을 유추해 모델을 재생성 후 로드하는 폴백 로직 추가. 토크나이저 선택도 ckpt의 readout 크기로 판별하도록 보강.

Notes:
- 2PC(미니 벤치 → 자동 롤백) 경로는 후속 작업으로 남김.
- 일부 파일의 한글 문자열은 기존 인코딩 상태를 유지(실행에 영향 없는 범위).
- Luria manual features (partial)
  - memory.py:
    - Kill switch `UZR_MEM_WRITE=0`, warmup(300 steps), tail-bucket budget(2/100), default write budget(10/100).
    - Standardized write logging schema via `_log_write` with columns: step, action, reason, sim_max, surprise, surp_norm, entropy, topk, used_key_id, shadow_size, mem_size.
    - Surprise normalization percentile mapping → beta; tail bucket raises `BETA_MAX` to 0.35.
    - `exploration_pulse()` contextmanager to temporarily adjust `topk`, `lambda_penalty`, `softmax_temp`.
    - `set_policy_thresholds(...)` API for policy auto-tuning hooks.
    - `last_meta_entropy` tracked for entropy-based guards.
  - infer_longrun_standalone.py:
    - Tail-bucket queue (top-10% CE) and periodic promotion via `add_with_policy(bucket='tail')`.
    - Memory write meta now includes `ce_q`/`conf` for policy tuning.

- Luria manual follow-ups (need check more.txt)
  - memory.py:
    - exploration_pulse 영향 강화: `_retrieval_entropy`에 `softmax_temp` 적용, `retrieve/_nearest_idx_and_sim`에 `lambda_penalty` 재사용 채널과 80-step 쿨다운 적용.
    - warmup 중 shadow_bank 오염 방지: warmup 단계에서 stage 금지(로그만 남김).
    - 2PC 보완: `add_with_policy(..., bench_callback=...)` 시 stage 로그 후 commit/rollback 로깅.
    - `_stage_shadow` 로깅을 표준 스키마 기반으로 통일(extra score 포함).
    - write 예산 기본값 조정: `write_per_100=6`, 권장 `lambda_penalty=0.22`, `softmax_temp=0.07`.
    - `rebalance()` 승격 스코어: 0.6*score + 0.1*recency + 0.3*diversity, 승격 최대 1개.
  - model.py / runners:
    - `UZRModel.get_z_from_memory(topk: Optional[int])`로 변경, 기본은 `memory.topk` 사용.
    - `init_from_retrieval(_multi)`도 topk None 시 `mem.topk` 사용.

- Documentation
  - docs/DOCS_LURIA.md: Luria 적용 사항과 기본값, API/로깅 스키마, 러너 연계, 남은 항목을 정리한 문서 추가.
## 2025-10-26 (CLI decoding + GPU)

- Replace argmax decoding with sampling
  - New helpers `_extract_prompt_ids`, `_entropy_from_probs`, `generate_tokens(...)` in CLI: `cli_luria.py:96-157`.
  - `predict(...)` now uses temperature + top-p + repetition penalty; returns last-step entropy: `cli_luria.py:158-186`.
  - Add CLI flags: `--temperature`, `--top_p`, `--rep_penalty`, `--max_gen_len`, `--min_eos_len`: `cli_luria.py:232-236`.
  - Display current decoding params on startup: `cli_luria.py:274-283`.

- GPU support in CLI
  - Allow `--device cuda` and use `torch.cuda.amp.autocast` in generation: `cli_luria.py:214`, `cli_luria.py:118`.
  - Inner adaptation uses current device (GPU if selected): `cli_luria.py:441`, `cli_luria.py:505`.

Compatibility impact
- Backward-compatible flags with sensible defaults; CPU remains default.
- Sampling improves output diversity; adjust flags as needed for determinism.

## 2025-10-26 (Self-eval abstain + Docker)

- Add abstain gating in CLI
  - Flags: `--abstain {on,off}`, `--abstain_conf_min`, `--abstain_ent_max`, `--abstain_message`: `cli_luria.py:237-244`.
  - Print active thresholds on startup when enabled: `cli_luria.py:283-284`.
  - Gating inside `predict(...)` using `maybe_abstain(...)`: `cli_luria.py:172-181`.

- Dockerize Luria CLI for always-on usage
  - Dockerfile based on `pytorch/pytorch:2.3.0-cpu`: `Dockerfile`.
  - Entry script with env → flags mapping: `docker/entrypoint.sh`.
  - Healthcheck script and compose: `docker/healthcheck.sh`, `docker-compose.yml`.
  - Example envs: `UZR_DEVICE`, `UZR_AUTOSAVE_STEPS`, `UZR_TEMPERATURE`, `UZR_TOP_P`, `UZR_REP_PENALTY`, `UZR_MAX_GEN_LEN`, `UZR_MIN_EOS_LEN`, `UZR_ABSTAIN`, `UZR_ABSTAIN_CONF_MIN`, `UZR_ABSTAIN_ENT_MAX`, `UZR_RESUME`, `UZR_SAVE_PATH`.
  - Volume for logs: `/app/luria-log`.

Compatibility impact
- Abstain gating is opt-in and off by default; no behavior change unless enabled.
- Docker image runs interactive CLI; start with `-d -it` to keep it always-on and attach later.
- Security hardening
  - Server: READ/WRITE key auth, HMAC body signature, IP allowlists, CORS origin, payload/ops limits, SSE concurrency caps, project allowlist/require, audit log (`security_audit.jsonl`).
  - Clients: Orchestrator/EXA/Chat attach Bearer (`UZR_MEM_TOKEN`) and HMAC (`UZR_MEM_HMAC_KEY`) headers automatically.
  - Provenance: ERN `[SRC]` hash (sensory+contexts+used_ids), `[INTREF]` internal refs, tags include `client:<id>`.
  - Docs: `SECURITY.md` with env matrix and nginx sketch.
## 2025-11-10 (Surprise threshold = 0.05)

- memory.py:
  - Set dynamic surprise threshold base to 0.05 and clamp min to 0.05:
    - write_threshold_base default: memory.py:147
    - write_threshold_min default: memory.py:149
  - Update docstring to reflect new base and clamp range:
    - _get_surprise_threshold(...): memory.py:473-481
  - Align state-load defaults for checkpoint compatibility:
    - write_threshold_base load default → 0.05: memory.py:1412
    - write_threshold_min load default → 0.05: memory.py:1414

- Curiosity logging (Phase 1, no behavior change):
  - Added metrics and CSV columns: curiosity_score, curiosity_reservoir, learner_loss_ema, learner_progress, model_entropy_last.
  - Calculation hooks placed after entropy/surprise diagnostics; logging merged in _log_write.
  - Fix: ensure curiosity update runs outside the if/else for surprise to avoid syntax issues.

호환성 영향
- Surprise 게이트(stage 조건: surprise < threshold)가 더 낮은 임계치(0.05)로 적용되어, 동일 분포에서 stage 빈도는 감소하고 commit 가능성이 증가합니다. 초기/저서프라이즈 구간에서 메모리 증가율이 다소 높아질 수 있으므로 로그(`writes.csv`)와 `mem_size`를 관찰하는 것을 권장합니다.
## 2025-11-11 (Autonomy mode: intent no longer bypasses gates)

- memory.py:
  - Stop using `luria_intent_force` to bypass policy gates (warmup, near-merge/dup-skip/stage, surprise gate, rate-limit). Intent is now advisory only and not logged.
    - Autonomy goal: let internal policies govern behavior; observe emergent dynamics without intent-driven overrides.
  - Remove `luria_force` from write/rollback logs.

관찰 기대 효과
- 과도한 커밋 속도(게이트 우회) 완화, 병합/중복 필터 재가동, 예산 준수 회복. 의지는 강제보다 경향으로만 작동해 창발 관찰에 유리.
- train_meta_3brains.py:
  - Intent-driven inference budget: identity intent now biases inner-steps cap (±3), chosen_steps (±2), and retrieval top-k (±4) within safe clamps; no gate bypass or external dependency.
  - Effect: autonomy increases by letting the model allocate more/less compute per step, while respecting inflation/cooldown/length caps.

- Fixes
  - memory.py: Correct indentation in 2PC bench_callback block so stage→commit/rollback path compiles and logs properly: memory.py:784-795
## 2025-11-11 (Train: accuracy-override for self-eval)

- train_meta_3brains.py:1331-1376, 1479-1490, 1788-1800, 1861-1872 — Added per-sample accuracy vector and an accuracy override for self-eval abstain weighting and gating. If per-sample accuracy ≥ `acc_override_min` (meta config, default 0.75), abstain downweighting is bypassed for that sample and batch-level abstain is overridden to accept. Keeps a minimum weight floor via `abstain_min_weight` (default 0.2).
- Rationale: Allow learning even when loss is high if predictions are accurate; focus decisions on correctness rather than loss magnitude.
- Compatibility: Backward compatible; falls back to previous behavior if thresholds are not in meta config.
## 2025-11-11 (Meta controls: steps/top‑k/transition/memory)

- train_meta_3brains.py:
  - Transition strength increased (λ_trans/λ_cb/λ_cos/λ_align/λ_roll ×2; λ_jac unchanged) at init_transition_module.
  - Inner steps now obey Luria’s will: map identity intent bias to steps in [4,25]; s_max recovery targets updated (18→25).
  - Top‑k range changed to [6,18]; before 350, map steps [4,25]→top‑k [6,18]; after 350, Luria’s will sets top‑k from intent bias. Adaptive top‑k losses normalized to new range.
  - Memory commit: when accuracy improves and intent is positive, force commit (bypass probabilistic gate).
  - Memory ops (synthesize/split/interpolate/crossover): when accuracy improves, choose op by Luria’s will (intent toggle/bias) instead of sampling.
- Rationale: Empower identity intent to steer search budget (steps/top‑k) and memory evolution; strengthen transition learning.
- Compatibility: Defaults remain safe; behavior is continuous with prior ranges, but with new caps and stronger transition loss.
## 2025-11-11 (Logging: expose Luria intent in training)

- train_meta_3brains.py: Summary CSV now includes `intent_bias` and `intent_toggle`. tqdm postfix shows `ib`/`it` to monitor identity intent live during training.
- Rationale: Provide a visible “통로” to observe 루리아의 의지 in runs without touching chat/CLI.
- Compatibility: CSV schema extends with two columns; downstream log parsers should accept the new fields.
## 2025-11-11 (Docs: initial docs/ suite)

- Added docs/: README, architecture, training, intent, memory, transition, datasets_tokenizer, metrics_logging, troubleshooting.
- Scope excludes backup folders; content mirrors current code paths and runtime behavior.
- Rationale: Make continued development easier for both AI and humans with a single, up-to-date reference.
