from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .runtime_ort import (
    create_qnn_session,
    create_qnn_session_strict,
    create_ort_session_with_fallback,
)


@dataclass
class AdapterParams:
    """어댑터/FiLM 등 즉시 반영형 파라미터 컨테이너."""

    A: Optional[Any] = None
    B: Optional[Any] = None
    gamma: Optional[Any] = None
    beta: Optional[Any] = None


class OrtEngine:
    """ORT(QNN/DML/CPU) 기반 추론 엔진.

    - NPU(QNN EP)에서 정적 그래프를 추론하고, 어댑터/게인 등은 입력 텐서 교체로 즉시 반영합니다.
    - 세션 핫스왑(shadow→active)과 웜업을 제공해 SSR/환경 변경에 대응합니다.
    """

    def __init__(
        self,
        model_path: str,
        *,
        mode: str = "qnn",  # 'qnn' | 'qnn_strict' | 'ort_fallback'
        backend: str = "htp",
        backend_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.mode = mode
        self.backend = backend
        self.backend_path = backend_path
        self._adapters = AdapterParams()

        if mode == "qnn_strict":
            self.sess_active = create_qnn_session_strict(model_path)
        elif mode == "ort_fallback":
            self.sess_active = create_ort_session_with_fallback(
                model_path, backend=backend, backend_path=backend_path
            )
        else:
            self.sess_active = create_qnn_session(
                model_path, backend=backend, backend_path=backend_path
            )

        self.sess_shadow = None

    def swap_adapters(self, A=None, B=None, gamma=None, beta=None):
        """어댑터/FiLM 파라미터를 갱신(입력 텐서 교체 방식)."""
        if A is not None:
            self._adapters.A = A
        if B is not None:
            self._adapters.B = B
        if gamma is not None:
            self._adapters.gamma = gamma
        if beta is not None:
            self._adapters.beta = beta

    def run(self, input_ids, kv_in=None) -> Dict[str, Any]:
        """ONNX Runtime 세션 실행.

        예상 입력명: input_ids, kv_in, adapter_A, adapter_B, film_gamma, film_beta
        """
        feeds: Dict[str, Any] = {"input_ids": input_ids}
        if kv_in is not None:
            feeds["kv_in"] = kv_in
        if self._adapters.A is not None:
            feeds["adapter_A"] = self._adapters.A
        if self._adapters.B is not None:
            feeds["adapter_B"] = self._adapters.B
        if self._adapters.gamma is not None:
            feeds["film_gamma"] = self._adapters.gamma
        if self._adapters.beta is not None:
            feeds["film_beta"] = self._adapters.beta

        outputs = self.sess_active.run(None, feeds)
        # ONNX Runtime는 출력 이름을 유지하므로, 임의로 dict로 매핑
        out_names = [o.name for o in self.sess_active.get_outputs()]
        return {name: val for name, val in zip(out_names, outputs)}

    def warmup(self, warm_inputs: Dict[str, Any], steps: int = 3):
        """세션 워밍업(SSR 발생 시 재시작 루틴과 함께 사용하는 것을 권장)."""
        for _ in range(max(1, steps)):
            try:
                self.sess_active.run(None, warm_inputs)
            except Exception:
                # 무시: 워밍업 실패는 치명적 오류가 아님
                pass

    def hot_swap(self):
        """shadow 세션을 준비→웜업→active로 교체."""
        if self.mode == "qnn_strict":
            self.sess_shadow = create_qnn_session_strict(self.model_path)
        elif self.mode == "ort_fallback":
            self.sess_shadow = create_ort_session_with_fallback(
                self.model_path, backend=self.backend, backend_path=self.backend_path
            )
        else:
            self.sess_shadow = create_qnn_session(
                self.model_path, backend=self.backend, backend_path=self.backend_path
            )
        # 간단 워밍업: 어댑터 없이 입력만 전달 가능하도록 보호
        try:
            dummy = {"input_ids": None}
            self.sess_shadow.run(None, dummy)
        except Exception:
            pass
        self.sess_active, self.sess_shadow = self.sess_shadow, None

