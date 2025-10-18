# UZR — Universal Z-Rule Runner Architecture

https://huggingface.co/naksyu/UZR/tree/main                 <<<<<<<< MODEL


## 개요 (Korean)
- **모델 파이프라인**: 입력 문자열은 바이트 단위 토크나이저로 정규화되어 BOS/EOS 토큰과 패딩이 포함된 시퀀스로 변환됩니다.【F:uzr/model.py†L10-L28】
- **표현 학습기**: TinyEncoder 트랜스포머가 토큰 임베딩과 위치 임베딩을 결합해 문맥 표현을 만들고, FiLM 계층이 언어·추론 잠재 벡터를 결합하여 토큰별 로짓을 산출합니다.【F:uzr/model.py†L43-L146】
- **잠재 상태 융합**: `UZRModel`은 규칙 추론(`z_rule`), 사고 보조(`z_think`), 언어 식별자 임베딩을 결합해 단일 조건 벡터로 투영합니다.【F:uzr/model.py†L62-L145】
- **압축 메모리**: `CompressedMemory`는 평균 임베딩과 느린 잠재 벡터를 정규화된 키와 함께 저장하고, 코사인 유사도 기반 검색 및 MLP 학습기로 복원값을 추정합니다.【F:uzr/memory.py†L37-L200】【F:uzr/memory.py†L282-L287】
- **장기 추론 루프**: 추론 스크립트는 태스크 샘플링 후 컨텍스트를 인코딩하고, 메모리에서 초기 `z_fast`를 받아 내부 경사 하강으로 조정한 뒤 느린 상태와 메모리를 업데이트합니다.【F:uzr/infer_longrun.py†L6-L170】
- **태스크 생성기**: `sample_task`는 언어별 규칙 팩토리를 조합해 컨텍스트·쿼리 예시와 설명을 반환하며, 수학 등 구조화된 입력을 선택적으로 혼합합니다.【F:uzr/tasks.py†L361-L420】

## Overview (English)
- **Model pipeline**: Input strings are normalized by a byte-level tokenizer that inserts BOS/EOS markers and padding before batching them as tensors.【F:uzr/model.py†L10-L28】
- **Representation learner**: A TinyEncoder transformer fuses token and positional embeddings, while a FiLM layer conditions the sequence states with language and reasoning latents before projecting token logits.【F:uzr/model.py†L43-L146】
- **Latent fusion**: `UZRModel` merges rule inference (`z_rule`), thinking support (`z_think`), and language embeddings into a single control vector via a learned projection.【F:uzr/model.py†L62-L145】
- **Compressed memory**: `CompressedMemory` stores normalized keys with averaged embeddings and slow latents, serving cosine-similarity retrieval and an auxiliary MLP regressor to reconstruct stored codes.【F:uzr/memory.py†L37-L200】【F:uzr/memory.py†L282-L287】
- **Long-run inference loop**: The inference script samples tasks, encodes contexts, seeds `z_fast` from memory, refines it with inner gradient steps, then updates the slow state and memory sketch.【F:uzr/infer_longrun.py†L6-L170】
- **Task generator**: `sample_task` blends language-specific rule factories to emit context/query pairs and human-readable descriptions, optionally injecting structured cases such as arithmetic.【F:uzr/tasks.py†L361-L420】
