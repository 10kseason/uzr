에이전트 LoRA×QNN 통합 가이드

개요
- 목적: 에이전트(대화/루리아 CLI)에서 추론은 NPU(QNN, ONNX Runtime)로, LoRA/FiLM/메모리 등 즉시 반영형 파라미터는 입력 텐서 교체로 반영합니다. 학습(미세적응)은 CPU/GPU에서 수행합니다.
- 전제: QNN 지원 ONNX(QDQ) 모델과 onnxruntime-qnn 설치가 대상 장치에 준비되어 있어야 합니다.

구성 요소
- npu/runtime_ort.py: QNN 전용/폴백 세션 생성, 컨텍스트 캐시/프로파일 옵션 지원.
- npu/engine.py: ORT 엔진(‘qnn’|‘qnn_strict’|‘ort_fallback’), 어댑터/FiLM 파라미터 hot-swap, 세션 hot-swap.
- chat_cli.py, cli_luria.py: 선택적으로 ORT/QNN 엔진을 사용하도록 확장.

사용 방법
1) 패키지 설치(장치별)
   - pip install onnxruntime-qnn
2) ONNX(QDQ) 모델 준비
   - QNN support manual.txt의 export/quantize 규칙(opset≥17, 정적 shape, QDQ INT8) 준수.
3) 대화 CLI(chat_cli)
   - PyTorch 경로(기존):
     - python -m uzr.chat_cli --ckpt ckpt.pt --device cuda
   - QNN/NPU 경로:
     - python -m uzr.chat_cli --ckpt ckpt.pt --device cpu --ort_model models/model_int8_qdq.onnx --engine qnn
     - 검증 모드(전량 오프로딩): --engine qnn_strict
     - 폴백 체인: --engine ort_fallback
   - 엔진 활성 시 명령:
     - /lora_npz path.npz  # adapter_A/B, film_gamma/beta를 npz에서 로드 후 즉시 반영
     - /hot_swap           # 세션 재생성 및 웜업(shadow→active)
4) 루리아 CLI(cli_luria)
   - 옵션/명령 동일: --ort_model, --engine, /lora_npz, /hot_swap
   - 적응(adapt_z)은 CPU에서 유지, 추론은 엔진 사용. 메모리 커밋은 기존 정책 준수(add_with_policy).

운영 팁
- 컨텍스트 캐시(ep.context_enable=1)로 세션 생성 지연을 줄이세요.
- 검증 시 CPU 폴백을 끄고(qnn_strict) 전체 오프로딩 여부를 먼저 확인합니다.
- LoRA/FiLM 학습은 외부 루틴에서 수행하고 결과를 npz로 내린 뒤 /lora_npz로 주입합니다.
- 문제가 생기면 /hot_swap으로 세션을 재생성하고 간단히 워밍업하세요.

참고
- QNN support manual.txt: QNN EP 옵션, export/quantize, 컨텍스트 캐시/프로파일, SSR 대응 등 상세 절차.
- 본 통합은 학습 경로를 변경하지 않습니다. PyTorch 경로와 병행 가능한 독립 추가입니다.

