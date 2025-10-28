"""
강제 쓰기 미니태스크 (Force Write Mini-task)

3~5% 배치에서 무조건 출력하게 만들어 균등분포 고착을 방지
"""

import torch
import random


class ForceWriteConfig:
    """강제 쓰기 설정 (lacomi.txt + rikomi.txt 처방 적용)"""
    prob = 0.10                # 10% 확률로 강제 쓰기 (기존 5%에서 상향)
    burst_every = 700          # 주기적 버스트: 매 N 스텝마다
    burst_len = 50             # 버스트 구간 길이 (스텝)
    burst_boost = 0.15         # 버스트 시 확률 (15%)
    min_len = 12               # 최소 토큰 길이
    loss_boost = 1.0           # 강제 구간 가중치 (필요시 1.2~1.5)
    len_penalty_coef = 0.03    # 길이 패널티 계수 (rikomi.txt: 0.02~0.05 권장, 기본 0.03)
    enable_after = 0           # 바로 활성화 (3000부터면 3000)


FW = ForceWriteConfig()


def should_force_write(step):
    """
    현재 스텝에서 강제 쓰기를 적용할지 결정 (주기적 버스트 포함)

    lacomi.txt 처방:
    - 기본 확률: 10% (기존 5%에서 상향)
    - 주기적 버스트: 매 700스텝마다 50스텝 구간 확률 15%
    - 목적: 모델이 회피할 때 등을 떠밀어 분기 표현 확장

    Args:
        step (int): 현재 글로벌 스텝

    Returns:
        bool: 강제 쓰기 적용 여부
    """
    if step < FW.enable_after:
        return False

    # 버스트 구간인지 확인 (매 burst_every 스텝마다 burst_len 스텝 동안)
    in_burst = (step % FW.burst_every) < FW.burst_len

    # 버스트 구간이면 확률 상승
    prob = FW.prob + (FW.burst_boost if in_burst else 0.0)

    # 확률 클램프 [0.0, 1.0]
    prob = min(max(prob, 0.0), 1.0)

    return random.random() < prob


def apply_force_write(batch, model_out, logits, targets, step):
    """
    강제 쓰기 손실 계산

    Args:
        batch: 입력 배치 패키지
        model_out: 모델 출력
        logits: [B, T, V] 로짓
        targets: [B, T] 타겟 토큰
        step: 현재 글로벌 스텝

    Returns:
        tuple: (extra_loss, info_dict)
    """
    # 1) 최소 길이 제약: 너무 짧으면 길이 패널티
    # ignore index=-100 기준으로 유효 토큰 마스크 계산
    valid_mask = (targets != -100).float()     # [B, T]
    seq_len = valid_mask.sum(dim=1)            # [B]

    # 최소 길이보다 짧으면 패널티 추가
    # rikomi.txt: len_penalty_coef를 0.02~0.05로 조정 가능하도록 (기본 0.03)
    len_penalty = torch.clamp(FW.min_len - seq_len, min=0).mean() * FW.len_penalty_coef

    # 2) 출력 손실(teacher forcing) 강제 집행
    # 표준 Cross Entropy 사용
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="mean"
    )

    # 최종 강제 쓰기 손실
    extra = FW.loss_boost * ce + len_penalty

    # 로깅용 정보
    info = {
        "fw_ce": ce.detach().item(),
        "fw_len_pen": len_penalty.detach().item(),
        "fw_avg_seq_len": seq_len.mean().detach().item()
    }

    return extra, info


def update_force_write_config(prob=None, min_len=None, loss_boost=None, enable_after=None,
                              burst_every=None, burst_len=None, burst_boost=None, len_penalty_coef=None):
    """
    강제 쓰기 설정 업데이트

    Args:
        prob (float, optional): 강제 쓰기 기본 확률
        min_len (int, optional): 최소 토큰 길이
        loss_boost (float, optional): 손실 가중치
        enable_after (int, optional): 활성화 시작 스텝
        burst_every (int, optional): 버스트 주기
        burst_len (int, optional): 버스트 구간 길이
        burst_boost (float, optional): 버스트 시 추가 확률
        len_penalty_coef (float, optional): 길이 패널티 계수 (rikomi.txt: 0.02~0.05 권장)
    """
    if prob is not None:
        FW.prob = prob
    if min_len is not None:
        FW.min_len = min_len
    if loss_boost is not None:
        FW.loss_boost = loss_boost
    if enable_after is not None:
        FW.enable_after = enable_after
    if burst_every is not None:
        FW.burst_every = burst_every
    if burst_len is not None:
        FW.burst_len = burst_len
    if burst_boost is not None:
        FW.burst_boost = burst_boost
    if len_penalty_coef is not None:
        FW.len_penalty_coef = len_penalty_coef


def get_force_write_stats():
    """
    현재 강제 쓰기 설정 정보 반환

    Returns:
        dict: 설정 정보
    """
    return {
        "prob": FW.prob,
        "min_len": FW.min_len,
        "loss_boost": FW.loss_boost,
        "enable_after": FW.enable_after,
        "burst_every": FW.burst_every,
        "burst_len": FW.burst_len,
        "burst_boost": FW.burst_boost,
        "len_penalty_coef": FW.len_penalty_coef
    }
