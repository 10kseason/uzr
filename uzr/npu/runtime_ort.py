import os
from typing import Dict, List, Optional


def _import_ort():
    try:
        import onnxruntime as ort  # type: ignore
        return ort
    except Exception as e:
        raise RuntimeError(
            "onnxruntime가 필요합니다. 'pip install onnxruntime-qnn' 또는 호환 패키지를 설치하세요."
        ) from e


def _session_options(
    disable_cpu_fallback: bool = False,
    context_enable: bool = True,
    profiling_level: Optional[str] = None,
    profiling_file_path: Optional[str] = None,
):
    ort = _import_ort()
    opts = ort.SessionOptions()
    if disable_cpu_fallback:
        # 전체 모델이 QNN에 올라가는지 검증할 때 사용
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if context_enable:
        # 컨텍스트 캐시 활성화로 세션 생성 시간 단축
        opts.add_session_config_entry("ep.context_enable", "1")
    if profiling_level:
        try:
            opts.add_session_config_entry("profiling_level", str(profiling_level))
            if profiling_file_path:
                # ORT가 지원하는 경우 결과를 파일로 남김
                opts.add_session_config_entry("profiling_file_path", str(profiling_file_path))
        except Exception:
            # 프로파일 설정이 지원되지 않는 빌드일 수 있음
            pass
    return opts


def _provider_options(
    backend_type: Optional[str] = None,
    backend_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    pov: Dict[str, str] = {}
    if backend_path:
        # Windows: QnnHtp.dll / Linux/Android: libQnnHtp.so
        pov["backend_path"] = backend_path
    if backend_type:
        # 기본: htp (Qualcomm NPU)
        pov["backend_type"] = backend_type
    return [pov] if pov else [{}]


def create_qnn_session(
    model_path: str,
    *,
    backend: str = "htp",
    backend_path: Optional[str] = None,
    disable_cpu_fallback: bool = False,
    context_enable: bool = True,
    profiling_level: Optional[str] = None,
    profiling_file_path: Optional[str] = None,
):
    """QNN EP 단독 세션 생성 (권장 기본).

    - backend: 'htp' | 'cpu' | 'gpu'
    - backend_path: 명시 경로가 필요한 환경에서만 지정 (예: 'QnnHtp.dll')
    """
    ort = _import_ort()
    opts = _session_options(
        disable_cpu_fallback=disable_cpu_fallback,
        context_enable=context_enable,
        profiling_level=profiling_level,
        profiling_file_path=profiling_file_path,
    )
    pov = _provider_options(backend_type=backend, backend_path=backend_path)
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["QNNExecutionProvider"],
        provider_options=pov,
    )


def create_qnn_session_strict(model_path: str):
    """CPU 폴백을 완전히 막고 QNN 전량 오프로딩을 검증하는 세션."""
    # 백엔드 경로는 환경에 따라 자동 감지 또는 사용자가 설정
    backend_path = None
    if os.name == "nt":
        backend_path = "QnnHtp.dll"
    return create_qnn_session(
        model_path,
        backend="htp",
        backend_path=backend_path,
        disable_cpu_fallback=True,
        context_enable=True,
    )


def create_ort_session_with_fallback(
    model_path: str,
    *,
    backend: str = "htp",
    backend_path: Optional[str] = None,
    context_enable: bool = True,
):
    """QNN → DML → CPU 순서의 폴백 체인 세션 생성.

    QNN 실패 시 DML, 최종 CPU로 폴백합니다.
    """
    ort = _import_ort()
    opts = _session_options(
        disable_cpu_fallback=False,
        context_enable=context_enable,
    )
    pov = _provider_options(backend_type=backend, backend_path=backend_path)
    providers = ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=providers,
        provider_options=[pov, {}, {}],
    )

