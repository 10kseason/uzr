"""
UZR — 한글 기반 실시간 코드북 (VQ/PQ 하이브리드) 표상 시스템

T-Codebook: 한글 자모 기반 Product Quantization
V-Codebook: 의미 임베딩 기반 Vector Quantization (VQ-VAE)
"""

import unicodedata
import hashlib
import random
import time
from typing import List, Dict, Optional, Tuple, Any
from copy import deepcopy
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 자모 분해 및 n-gram 해시 유틸리티
# ============================================================

# 한글 유니코드 범위: U+AC00 ~ U+D7A3 (가 ~ 힣)
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
JAMO_INITIAL_BASE = 0x1100  # 초성
JAMO_MEDIAL_BASE = 0x1161   # 중성
JAMO_FINAL_BASE = 0x11A8    # 종성

# 초성 19개
CHOSEONG = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
    'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]
# 중성 21개
JUNGSEONG = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
# 종성 28개 (0번은 받침 없음)
JONGSEONG = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]


def decompose_jamo(text: str) -> str:
    """한글 음절을 자모로 분해 (조합형 → 자모 문자열)"""
    result = []
    for ch in text:
        code = ord(ch)
        if HANGUL_BASE <= code <= HANGUL_END:
            # 조합형 한글
            offset = code - HANGUL_BASE
            cho_idx = offset // (21 * 28)
            jung_idx = (offset % (21 * 28)) // 28
            jong_idx = offset % 28

            result.append(CHOSEONG[cho_idx])
            result.append(JUNGSEONG[jung_idx])
            if jong_idx > 0:
                result.append(JONGSEONG[jong_idx])
        else:
            # 한글 아니면 그대로
            result.append(ch)
    return ''.join(result)


def normalize_text(text: str) -> str:
    """텍스트 정규화: NFKC + 연속 공백 제거 + 제어문자 제거"""
    text = unicodedata.normalize('NFKC', text)
    # 제어문자 제거
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('C') or ch in '\n\t ')
    # 연속 공백을 하나로
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_ngrams(text: str, n_min: int = 2, n_max: int = 3) -> List[str]:
    """자모 문자열에서 n-gram 추출"""
    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            if gram.strip():  # 공백만 있는 n-gram 제외
                ngrams.append(gram)
    return ngrams


def murmur_hash(text: str, seed: int = 42) -> int:
    """MurmurHash3 대신 간단한 해시 (파이썬 기본 hash + seed)"""
    # 실제로는 mmh3 라이브러리 사용 권장
    # pip install mmh3
    # import mmh3
    # return mmh3.hash(text, seed) & 0xFFFFFFFF

    # 여기서는 간단한 대체 구현
    h = hashlib.md5(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(h[:4], 'little')


def hash_ngrams_to_sparse(ngrams: List[str], m: int = 131072, seed: int = 42) -> torch.Tensor:
    """n-gram을 해시하여 희소 벡터 생성 (m차원)"""
    vec = torch.zeros(m, dtype=torch.float32)
    for gram in ngrams:
        idx = murmur_hash(gram, seed) % m
        vec[idx] += 1.0
    return vec


# ============================================================
# 2. TCodebook (한글 자모 기반 PQ)
# ============================================================

class TCodebook:
    """Text Codebook: 한글 자모 기반 Product Quantization"""

    def __init__(
        self,
        Gt: int = 6,           # 서브스페이스 개수
        Kt: int = 128,         # 코드북 크기 (각 서브스페이스)
        dt: int = 384,         # 투영 차원
        m: int = 131072,       # 해시 공간 크기
        ema_decay: float = 0.995,
        seed: int = 42,
    ):
        self.Gt = Gt
        self.Kt = Kt
        self.dt = dt
        self.m = m
        self.ema_decay = ema_decay
        self.seed = seed

        # 랜덤 투영 행렬 U: (dt, m)
        torch.manual_seed(seed)
        U_raw = torch.randn(dt, m)
        # 정규직교화 (QR 분해)
        self.U, _ = torch.qr(U_raw.T)
        self.U = self.U.T  # (dt, m)

        # 서브스페이스별 차원
        self.d_sub = dt // Gt

        # 코드북: Gt개 서브스페이스, 각각 Kt개 코드 (d_sub 차원)
        self.centroids = []
        for g in range(Gt):
            # 초기 랜덤 초기화 (나중에 k-means로 초기화 권장)
            C_g = torch.randn(Kt, self.d_sub) * 0.1
            self.centroids.append(C_g)

        # 사용 통계
        self.usage_counts = [torch.zeros(Kt) for _ in range(Gt)]

    def encode(self, text: str) -> List[str]:
        """텍스트 → T 토큰 리스트 ['KA07', 'KB33', ...]"""
        if not text:
            return []

        # 정규화 + 자모 분해
        norm_text = normalize_text(text)
        jamo_text = decompose_jamo(norm_text)

        # n-gram 추출 및 해시
        ngrams = extract_ngrams(jamo_text, n_min=2, n_max=3)
        if not ngrams:
            return []

        sparse_vec = hash_ngrams_to_sparse(ngrams, m=self.m, seed=self.seed)

        # 투영: U @ sparse_vec -> (dt,)
        feats = self.U @ sparse_vec  # (dt,)

        # PQ: 서브스페이스로 분할 및 양자화
        tokens = []
        for g in range(self.Gt):
            start = g * self.d_sub
            end = start + self.d_sub
            feat_g = feats[start:end]  # (d_sub,)

            # 최근접 코드 찾기
            C_g = self.centroids[g]  # (Kt, d_sub)
            dists = torch.cdist(feat_g.unsqueeze(0), C_g, p=2).squeeze(0)  # (Kt,)
            idx = torch.argmin(dists).item()

            # 토큰 생성: K[A-Z][0-9]{2}
            prefix = chr(ord('A') + g % 26)
            token = f"K{prefix}{idx:02d}"
            tokens.append(token)

            # 사용 통계
            self.usage_counts[g][idx] += 1

        return tokens

    def update_ema(self, text: str, tokens: List[str]):
        """EMA 업데이트 (온라인)"""
        if not text or not tokens:
            return

        # 특징 재추출
        norm_text = normalize_text(text)
        jamo_text = decompose_jamo(norm_text)
        ngrams = extract_ngrams(jamo_text, n_min=2, n_max=3)
        if not ngrams:
            return

        sparse_vec = hash_ngrams_to_sparse(ngrams, m=self.m, seed=self.seed)
        feats = self.U @ sparse_vec

        # 각 서브스페이스 업데이트
        for g in range(min(self.Gt, len(tokens))):
            start = g * self.d_sub
            end = start + self.d_sub
            feat_g = feats[start:end]

            # 토큰에서 인덱스 추출
            token = tokens[g]
            if not token.startswith('K'):
                continue
            try:
                idx = int(token[-2:])
            except:
                continue

            # EMA 업데이트
            alpha = self.ema_decay
            self.centroids[g][idx] = alpha * self.centroids[g][idx] + (1 - alpha) * feat_g

    def get_entropy(self) -> float:
        """코드 사용 엔트로피 계산"""
        total_usage = sum(counts.sum().item() for counts in self.usage_counts)
        if total_usage == 0:
            return 0.0

        entropies = []
        for counts in self.usage_counts:
            probs = counts / (counts.sum() + 1e-9)
            probs = probs.clamp(min=1e-9)
            ent = -(probs * probs.log()).sum().item()
            entropies.append(ent)

        return sum(entropies) / len(entropies)

    def get_dead_ratio(self) -> float:
        """사용되지 않는 코드 비율"""
        total_codes = self.Gt * self.Kt
        dead_count = sum((counts == 0).sum().item() for counts in self.usage_counts)
        return dead_count / total_codes


# ============================================================
# 3. VCodebook (의미 임베딩 기반 VQ-VAE)
# ============================================================

class VCodebook(nn.Module):
    """Vector Codebook: VQ-VAE with EMA"""

    def __init__(
        self,
        d: int = 768,           # 입력 차원
        G: int = 6,             # 서브스페이스 개수
        K: int = 128,           # 코드북 크기 (각 서브스페이스)
        beta: float = 0.25,     # commitment loss 가중치
        ema_decay: float = 0.995,
        seed: int = 42,
    ):
        super().__init__()
        self.d = d
        self.G = G
        self.K = K
        self.beta = beta
        self.ema_decay = ema_decay

        # 서브스페이스별 차원 (균등 분할)
        self.d_g = d // G

        # 투영 행렬: 각 서브스페이스로 투영
        torch.manual_seed(seed)
        self.projections = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_g, d) * 0.02) for _ in range(G)
        ])

        # 코드북: G개 서브스페이스, 각각 K개 코드 (d_g 차원)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(K, self.d_g) * 0.1) for _ in range(G)
        ])

        # EMA 통계
        self.register_buffer('ema_cluster_size', torch.zeros(G, K))
        self.register_buffer('ema_w', torch.zeros(G, K, self.d_g))

        # 사용 통계
        self.register_buffer('usage_counts', torch.zeros(G, K))

    def encode(self, h: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        h: [d] 또는 [B, d]
        return: (토큰 리스트, 양자화된 벡터)
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)  # [1, d]

        B = h.size(0)
        tokens = []
        quantized_parts = []

        for g in range(self.G):
            # 투영: W_g @ h -> (B, d_g)
            z_g = F.linear(h, self.projections[g])  # (B, d_g)

            # 최근접 코드 찾기
            C_g = self.codebooks[g]  # (K, d_g)
            dists = torch.cdist(z_g, C_g, p=2)  # (B, K)
            indices = torch.argmin(dists, dim=1)  # (B,)

            # 양자화
            quantized_g = F.embedding(indices, C_g)  # (B, d_g)
            quantized_parts.append(quantized_g)

            # 토큰 생성: [A-Z][0-9]{2}
            for b in range(B):
                idx = indices[b].item()
                prefix = chr(ord('A') + g % 26)
                token = f"{prefix}{idx:02d}"
                if b == 0:  # 배치 첫 샘플만
                    tokens.append(token)

                # 사용 통계
                self.usage_counts[g, idx] += 1

        # 결합
        quantized = torch.cat(quantized_parts, dim=1)  # (B, d)

        return tokens, quantized

    def decode(self, tokens: List[str]) -> torch.Tensor:
        """토큰 리스트 → 벡터 복원"""
        parts = []
        for g, token in enumerate(tokens[:self.G]):
            # 토큰에서 인덱스 추출
            try:
                idx = int(token[-2:])
            except:
                idx = 0

            C_g = self.codebooks[g]
            vec_g = C_g[idx]  # (d_g,)
            parts.append(vec_g)

        # 결합 후 정규화
        h_hat = torch.cat(parts, dim=0)  # (d,)
        h_hat = F.normalize(h_hat, p=2, dim=0)
        return h_hat

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VQ-VAE forward pass
        return: (quantized, vq_loss)
        """
        tokens, quantized = self.encode(h)

        # VQ loss: ||sg[z] - e||^2 + beta * ||z - sg[e]||^2
        loss_vq = F.mse_loss(quantized.detach(), h) + self.beta * F.mse_loss(quantized, h.detach())

        # Straight-through estimator
        quantized = h + (quantized - h).detach()

        return quantized, loss_vq

    def update_ema(self, h: torch.Tensor):
        """EMA 코드북 업데이트"""
        if h.dim() == 1:
            h = h.unsqueeze(0)

        with torch.no_grad():
            for g in range(self.G):
                z_g = F.linear(h, self.projections[g])  # (B, d_g)
                C_g = self.codebooks[g]

                dists = torch.cdist(z_g, C_g, p=2)
                indices = torch.argmin(dists, dim=1)

                # EMA 업데이트
                for i in range(self.K):
                    mask = (indices == i)
                    count = mask.sum().item()
                    if count > 0:
                        cluster_vecs = z_g[mask].mean(dim=0)

                        self.ema_cluster_size[g, i] = self.ema_decay * self.ema_cluster_size[g, i] + (1 - self.ema_decay) * count
                        self.ema_w[g, i] = self.ema_decay * self.ema_w[g, i] + (1 - self.ema_decay) * cluster_vecs

                        # 코드북 갱신
                        n = self.ema_cluster_size[g, i]
                        if n > 0:
                            self.codebooks[g].data[i] = self.ema_w[g, i] / n

    def get_entropy(self) -> float:
        """코드 사용 엔트로피"""
        total = self.usage_counts.sum().item()
        if total == 0:
            return 0.0

        entropies = []
        for g in range(self.G):
            counts_g = self.usage_counts[g]
            probs = counts_g / (counts_g.sum() + 1e-9)
            probs = probs.clamp(min=1e-9)
            ent = -(probs * probs.log()).sum().item()
            entropies.append(ent)

        return sum(entropies) / len(entropies)

    def get_dead_ratio(self) -> float:
        """사용되지 않는 코드 비율"""
        total_codes = self.G * self.K
        dead_count = (self.usage_counts == 0).sum().item()
        return dead_count / total_codes


# ============================================================
# 4. CodebookManager (통합 관리자)
# ============================================================

@dataclass
class CodebookStats:
    """코드북 통계"""
    t_entropy: float
    v_entropy: float
    t_dead_ratio: float
    v_dead_ratio: float
    recon_error_mean: float = 0.0
    recon_error_p95: float = 0.0


class CodebookManager:
    """T-Codebook + V-Codebook 통합 관리"""

    def __init__(
        self,
        d: int = 768,
        t_cfg: Optional[Dict] = None,
        v_cfg: Optional[Dict] = None,
        seed: int = 42,
        commit_steps: int = 1000,
        device: str = "cpu",
    ):
        self.d = d
        self.seed = seed
        self.commit_steps = commit_steps
        self.device = device

        # T-Codebook 설정
        if t_cfg is None:
            t_cfg = dict(Gt=6, Kt=128, dt=384, m=131072, ema_decay=0.995)
        self.t = TCodebook(**t_cfg, seed=seed)

        # V-Codebook 설정
        if v_cfg is None:
            v_cfg = dict(G=6, K=128, beta=0.25, ema_decay=0.995)
        self.v = VCodebook(d=d, **v_cfg, seed=seed).to(device)

        # Shadow codebook (온라인 업데이트용)
        self.shadow = None
        self._init_shadow()

        # 버전 관리
        self.active_id = self._generate_cb_id()
        self.step_count = 0

        # 재구성 오차 추적
        self.recon_errors = []

    def _init_shadow(self):
        """Shadow codebook 초기화"""
        self.shadow = {
            't': deepcopy(self.t),
            'v': deepcopy(self.v),
        }

    def _generate_cb_id(self) -> str:
        """코드북 ID 생성: uzr-<model>-<date>-<hash8>"""
        timestamp = time.strftime("%Y%m%d")
        random_hash = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:8]
        return f"uzr-3brains-{timestamp}-{random_hash}"

    def encode(
        self,
        text: Optional[str] = None,
        h: Optional[torch.Tensor] = None,
        return_parts: bool = False,
    ) -> Dict[str, List[str]]:
        """
        텍스트 및/또는 임베딩을 코드북 토큰으로 인코딩

        Args:
            text: 입력 텍스트 (None이면 T 생략)
            h: 임베딩 벡터 [d] (None이면 V 생략)
            return_parts: True면 {"T": [...], "V": [...]} 반환

        Returns:
            {"T": [...], "V": [...]}
        """
        tokens_T = []
        tokens_V = []

        if text is not None:
            tokens_T = self.t.encode(text)

        if h is not None:
            if not isinstance(h, torch.Tensor):
                h = torch.tensor(h, dtype=torch.float32)
            h = h.to(self.device)
            tokens_V, _ = self.v.encode(h)

        if return_parts:
            return {"T": tokens_T, "V": tokens_V}
        else:
            # 문자열 포맷
            return {"T": tokens_T, "V": tokens_V}

    def decode(self, tokens_V: List[str]) -> torch.Tensor:
        """V 토큰으로부터 벡터 복원"""
        return self.v.decode(tokens_V)

    def accumulate_update(self, text: Optional[str] = None, h: Optional[torch.Tensor] = None):
        """Shadow codebook에 EMA 업데이트 누적"""
        if text is not None:
            tokens_T = self.shadow['t'].encode(text)
            self.shadow['t'].update_ema(text, tokens_T)

        if h is not None:
            if not isinstance(h, torch.Tensor):
                h = torch.tensor(h, dtype=torch.float32)
            h = h.to(self.device)
            self.shadow['v'].update_ema(h)

        self.step_count += 1

    def passes_guardrails(self) -> bool:
        """Shadow codebook이 guardrail 기준을 통과하는지 확인"""
        # 엔트로피 하한: 균등 분포의 75% 이상
        t_entropy = self.shadow['t'].get_entropy()
        v_entropy = self.shadow['v'].get_entropy()

        t_uniform_ent = torch.log(torch.tensor(self.shadow['t'].Kt, dtype=torch.float32)).item()
        v_uniform_ent = torch.log(torch.tensor(self.shadow['v'].K, dtype=torch.float32)).item()

        if t_entropy < 0.75 * t_uniform_ent:
            return False
        if v_entropy < 0.75 * v_uniform_ent:
            return False

        # Dead code 비율: 10% 미만
        t_dead = self.shadow['t'].get_dead_ratio()
        v_dead = self.shadow['v'].get_dead_ratio()

        if t_dead > 0.10 or v_dead > 0.10:
            return False

        return True

    def maybe_commit(self) -> bool:
        """
        주기적으로 shadow를 active로 스왑

        Returns:
            True if swapped
        """
        if self.step_count < self.commit_steps:
            return False

        if not self.passes_guardrails():
            return False

        # Atomic swap
        self.t = deepcopy(self.shadow['t'])
        self.v = deepcopy(self.shadow['v'])

        # 새 ID 생성
        self.active_id = self._generate_cb_id()

        # Shadow 재초기화
        self._init_shadow()
        self.step_count = 0

        return True

    def stats(self) -> CodebookStats:
        """통계 정보 반환"""
        t_ent = self.t.get_entropy()
        v_ent = self.v.get_entropy()
        t_dead = self.t.get_dead_ratio()
        v_dead = self.v.get_dead_ratio()

        # 재구성 오차 통계
        if self.recon_errors:
            errors_tensor = torch.tensor(self.recon_errors[-1000:])  # 최근 1000개
            recon_mean = errors_tensor.mean().item()
            recon_p95 = torch.quantile(errors_tensor, 0.95).item()
        else:
            recon_mean = 0.0
            recon_p95 = 0.0

        return CodebookStats(
            t_entropy=t_ent,
            v_entropy=v_ent,
            t_dead_ratio=t_dead,
            v_dead_ratio=v_dead,
            recon_error_mean=recon_mean,
            recon_error_p95=recon_p95,
        )

    def format_zc(self, tokens: Dict[str, List[str]]) -> str:
        """ZC 필드 포맷: T:KA07 KB33 | V:A03 C17"""
        t_str = ' '.join(tokens.get('T', []))
        v_str = ' '.join(tokens.get('V', []))

        parts = []
        if t_str:
            parts.append(f"T:{t_str}")
        if v_str:
            parts.append(f"V:{v_str}")

        return ' | '.join(parts)

    @staticmethod
    def init(
        d: int = 768,
        t_cfg: Optional[Dict] = None,
        v_cfg: Optional[Dict] = None,
        seed: int = 42,
        commit_steps: int = 1000,
        device: str = "cpu",
    ) -> 'CodebookManager':
        """팩토리 메서드"""
        return CodebookManager(d=d, t_cfg=t_cfg, v_cfg=v_cfg, seed=seed, commit_steps=commit_steps, device=device)
