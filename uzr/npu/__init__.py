"""npu 패키지: ONNX Runtime(QNN EP) 기반 추론 지원.

이 모듈은 QNN 매뉴얼(v1.0)을 따르며, NPU는 추론, CPU/GPU는 경량 미세학습(어댑터/게인/메모리)만 수행하는 구성을 목표로 합니다.
"""

from .runtime_ort import (
    create_qnn_session,
    create_qnn_session_strict,
    create_ort_session_with_fallback,
)
from .engine import OrtEngine

__all__ = [
    "create_qnn_session",
    "create_qnn_session_strict",
    "create_ort_session_with_fallback",
    "OrtEngine",
]

