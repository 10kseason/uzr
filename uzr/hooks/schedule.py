"""
Abstain 감쇠 스케줄 (Abstain Decay Schedule)

3000~15000 스텝에서 abstain_max를 1.0에서 0.2로 선형 감쇠
"""


def abstain_cap_schedule(step, t0=3000, t1=9000, a_start=1.0, a_end=0.15):
    """
    Abstain cap 스케줄 함수 (가속 버전)

    lacomi.txt 처방: 3k→9k로 압축, 0.15까지 감쇠
    - 커버리지 창을 앞당겨 분포 전환 시 엔트로피 스파이크를 학습 기회로 전환

    Args:
        step (int): 현재 글로벌 스텝
        t0 (int): 감쇠 시작 스텝 (기본값: 3000)
        t1 (int): 감쇠 종료 스텝 (기본값: 9000, 기존 15000에서 가속)
        a_start (float): 시작 abstain_max 값 (기본값: 1.0)
        a_end (float): 종료 abstain_max 값 (기본값: 0.15, 기존 0.2에서 강화)

    Returns:
        float: 현재 스텝의 abstain_max 값
    """
    # 스텝이 t0 이전이면 시작값 반환
    if step < t0:
        return a_start

    # 스텝이 t1 이후면 종료값 반환
    if step >= t1:
        return a_end

    # 선형 보간 (linear interpolation)
    progress = (step - t0) / (t1 - t0)
    a_current = a_start + (a_end - a_start) * progress

    # 안전 클램프 [0.0, 1.0]
    a_current = min(max(a_current, 0.0), 1.0)

    return a_current


def get_abstain_schedule_info(step):
    """
    현재 스텝의 스케줄 정보를 반환

    Args:
        step (int): 현재 글로벌 스텝

    Returns:
        dict: 스케줄 정보 (abstain_cap, phase, progress)
    """
    abstain_cap = abstain_cap_schedule(step)

    # 현재 단계 파악 (가속 버전에 맞춰 업데이트)
    if step < 3000:
        phase = "warmup"
        progress = 0.0
    elif step < 9000:
        phase = "decay"
        progress = (step - 3000) / (9000 - 3000)
    else:
        phase = "stable"
        progress = 1.0

    return {
        "abstain_cap": abstain_cap,
        "phase": phase,
        "progress": progress
    }
