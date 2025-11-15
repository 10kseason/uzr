#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uzr_ckpt_best.pt 체크포인트의 차원을 상세하게 조사하는 스크립트
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, List


def analyze_tensor(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    """텐서의 상세 정보 분석"""
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "requires_grad": tensor.requires_grad,
        "is_sparse": tensor.is_sparse,
        "ndim": tensor.ndim,
    }


def analyze_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str = "") -> List[Dict[str, Any]]:
    """state_dict의 모든 텐서 분석"""
    results = []

    for key, value in state_dict.items():
        full_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, torch.Tensor):
            results.append(analyze_tensor(full_name, value))
        elif isinstance(value, dict):
            results.extend(analyze_state_dict(value, prefix=f"{full_name}."))

    return results


def analyze_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """체크포인트 파일 전체 분석"""
    print(f"체크포인트 로딩 중: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": f"체크포인트 로딩 실패: {str(e)}"}

    print(f"체크포인트 로딩 완료!\n")

    analysis = {
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "top_level_keys": list(checkpoint.keys()),
        "sections": {}
    }

    # 최상위 키 정보
    print("=" * 80)
    print("최상위 키:")
    print("=" * 80)
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor{list(value.shape)}")
        elif isinstance(value, dict):
            print(f"  {key}: Dict (키 개수: {len(value)})")
        else:
            print(f"  {key}: {type(value).__name__}")
    print()

    # 각 섹션 상세 분석
    for section_name, section_data in checkpoint.items():
        print("=" * 80)
        print(f"섹션: {section_name}")
        print("=" * 80)

        if isinstance(section_data, dict):
            # state_dict인 경우
            if all(isinstance(v, torch.Tensor) for v in section_data.values()):
                tensor_info = analyze_state_dict(section_data)
                analysis["sections"][section_name] = {
                    "type": "state_dict",
                    "num_parameters": len(tensor_info),
                    "total_numel": sum(info["numel"] for info in tensor_info),
                    "tensors": tensor_info
                }

                print(f"타입: state_dict")
                print(f"파라미터 개수: {len(tensor_info)}")
                print(f"총 원소 개수: {sum(info['numel'] for info in tensor_info):,}")
                print(f"\n상세 차원 정보:")

                for info in tensor_info:
                    shape_str = "×".join(map(str, info["shape"]))
                    print(f"  {info['name']:50s} | 차원: {shape_str:20s} | 원소: {info['numel']:>12,} | dtype: {info['dtype']}")

            # 일반 dict인 경우 (args 등)
            else:
                analysis["sections"][section_name] = {
                    "type": "dict",
                    "keys": list(section_data.keys()),
                    "content": {k: str(v) if not isinstance(v, (dict, list)) else type(v).__name__
                               for k, v in section_data.items()}
                }

                print(f"타입: dict")
                print(f"키 목록:")
                for key, value in section_data.items():
                    if isinstance(value, (int, float, str, bool)):
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {type(value).__name__}")

        elif isinstance(section_data, torch.Tensor):
            info = analyze_tensor(section_name, section_data)
            analysis["sections"][section_name] = {
                "type": "tensor",
                "info": info
            }

            shape_str = "×".join(map(str, info["shape"]))
            print(f"타입: Tensor")
            print(f"차원: {shape_str}")
            print(f"원소 개수: {info['numel']:,}")
            print(f"dtype: {info['dtype']}")

        else:
            analysis["sections"][section_name] = {
                "type": type(section_data).__name__,
                "value": str(section_data) if not isinstance(section_data, (dict, list)) else f"{type(section_data).__name__} with {len(section_data)} items"
            }
            print(f"타입: {type(section_data).__name__}")

        print()

    return analysis


def main():
    import argparse

    parser = argparse.ArgumentParser(description="uzr_ckpt_best.pt 차원 분석 도구")
    parser.add_argument("--ckpt", default="uzr_ckpt_best.pt", help="체크포인트 파일 경로")
    parser.add_argument("--output", default="checkpoint_dimensions.json", help="결과 저장 파일명")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    args = parser.parse_args()

    # 체크포인트 파일 존재 확인
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        print(f"현재 작업 디렉토리: {Path.cwd()}")
        return

    # 분석 실행
    analysis = analyze_checkpoint(str(ckpt_path))

    # 결과 저장
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"분석 완료! 결과가 저장되었습니다: {output_path.resolve()}")
    print("=" * 80)

    # 요약 통계
    if "sections" in analysis:
        total_params = 0
        total_numel = 0

        for section_name, section_info in analysis["sections"].items():
            if section_info.get("type") == "state_dict":
                total_params += section_info["num_parameters"]
                total_numel += section_info["total_numel"]

        if total_params > 0:
            print(f"\n총 파라미터 수: {total_params:,}")
            print(f"총 원소 개수: {total_numel:,}")
            print(f"메모리 (float32 기준): {total_numel * 4 / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
