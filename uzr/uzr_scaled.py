#!/usr/bin/env python3
"""
UZR Scaled: Qwen 0.5B + z-conditioning + CompressedMemory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Dict, Any
import math

# ==================== FiLM Conditioning ====================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for z conditioning"""
    def __init__(self, z_dim: int, d_model: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, d_model * 2)
        # 작게 초기화
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, h: torch.Tensor, z: torch.Tensor):
        """
        h: [B, T, D] hidden states
        z: [B, z_dim] or [z_dim] conditioning vector
        """
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(h.size(0), -1)
        
        gamma, beta = self.fc(z).chunk(2, dim=-1)  # [B, D] each
        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)    # [B, 1, D]
        
        # FiLM: h * (1 + tanh(gamma)) + beta
        return h * (1 + torch.tanh(gamma)) + beta


# ==================== UZR Scaled Model ====================

class UZRScaled(nn.Module):
    """
    Qwen 0.5B as frozen base + z-conditioning via FiLM
    """
    def __init__(
        self, 
        base_model_name: str = "Qwen/Qwen2.5-0.5B",
        z_dim: int = 512,
        freeze_base: bool = True,
        num_film_layers: int = 4,  # FiLM을 몇 개 레이어에 넣을지
    ):
        super().__init__()
        
        print(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=None,  # 수동 배치
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # padding token 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # base model freeze
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("Base model frozen")
        
        # 모델 구조 파악
        self.d_model = self.base_model.config.hidden_size  # 896 for Qwen 0.5B
        self.vocab_size = self.base_model.config.vocab_size
        self.z_dim = z_dim
        
        # FiLM layers (transformer의 중간 레이어들에 적용)
        self.num_layers = self.base_model.config.num_hidden_layers
        self.film_layers = nn.ModuleList([
            FiLMLayer(z_dim, self.d_model) 
            for _ in range(min(num_film_layers, self.num_layers))
        ])
        self.film_indices = [
            i * (self.num_layers // num_film_layers) 
            for i in range(num_film_layers)
        ]
        
        # z initialization (learnable)
        self.z0 = nn.Parameter(torch.zeros(z_dim))
        
        print(f"UZR Scaled initialized:")
        print(f"  - Base: {base_model_name}")
        print(f"  - d_model: {self.d_model}")
        print(f"  - z_dim: {z_dim}")
        print(f"  - FiLM layers: {num_film_layers} at indices {self.film_indices}")
        print(f"  - Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.1f}M")
    
    def init_z(self, batch_size: int = 1):
        """Initialize z from learned prior"""
        return self.z0.unsqueeze(0).expand(batch_size, -1).clone()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        z: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        input_ids: [B, T]
        z: [B, z_dim] or [z_dim]
        returns: logits [B, T, V]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(input_ids.size(0), -1)
        
        # Get base model outputs with hooks for FiLM
        outputs = self._forward_with_film(input_ids, z, attention_mask)
        return outputs.logits
    
    def _forward_with_film(self, input_ids, z, attention_mask):
        """
        Forward pass with FiLM conditioning applied at specific layers
        """
        # 간단한 구현: 전체 forward 후 마지막에만 FiLM 적용
        # 더 정교한 구현은 hook 사용 필요
        
        with torch.no_grad() if self.base_model.training == False else torch.enable_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        # 마지막 hidden state에 FiLM 적용 (간단 버전)
        hidden_states = outputs.hidden_states[-1]  # [B, T, D]
        
        # FiLM 적용
        for film in self.film_layers:
            hidden_states = film(hidden_states, z)
        
        # LM head 통과 (base model의 lm_head 사용)
        logits = self.base_model.lm_head(hidden_states)
        
        # outputs 객체 생성 (namedtuple 형태 유지)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def generate_text(self, prompt: str, z: torch.Tensor, max_length: int = 50):
        """간단한 생성 함수"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(z.device)
        
        with torch.no_grad():
            logits = self.forward(input_ids, z)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            
        return self.tokenizer.decode([next_token.item()])


# ==================== Loss & Optimization Helpers ====================

def seq_ce_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100):
    """
    logits: [B, T, V]
    target: [B, T]
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), 
        target.reshape(-1), 
        ignore_index=ignore_index
    )

def soft_threshold(z: torch.Tensor, lam: float):
    """Proximal operator for L1 regularization"""
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)

def confidence_from_logits(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100):
    """
    Compute confidence score from logits and target
    Returns: [B] confidence scores in [0, 1]
    """
    with torch.no_grad():
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=ignore_index,
            reduction='none'
        ).view(target.shape)
        
        mask = (target != ignore_index).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        avg_ce = (ce * mask).sum(dim=1) / denom
        
        # 0-1로 스케일 (낮은 CE = 높은 confidence)
        conf = torch.sigmoid(-avg_ce / 2.0)
        return conf


# ==================== Inner Loop Adaptation ====================

def inner_adapt_z(
    model: UZRScaled,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    z_init: torch.Tensor,
    steps: int = 5,
    lr: float = 0.5,
    lam: float = 1e-3,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Inner loop: adapt z on context examples
    
    Returns: adapted z
    """
    z = z_init.clone().detach().requires_grad_(True)
    
    for _ in range(steps):
        logits = model(input_ids, z, attention_mask)
        loss = seq_ce_loss(logits, target_ids, ignore_index=-100)
        loss = loss + lam * torch.sum(torch.abs(z))
        
        # gradient descent
        grad = torch.autograd.grad(loss, z, retain_graph=False)[0]
        
        # adaptive step size based on confidence
        with torch.no_grad():
            conf = confidence_from_logits(logits, target_ids).mean()
            step_size = lr * (0.4 + 0.6 * conf.item())
        
        z = z - step_size * grad
    
    # soft threshold (L1 proximal)
    with torch.no_grad():
        z = soft_threshold(z, lam * 0.5)
    
    return z.detach()


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 60)
    print("UZR Scaled Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Initialize model
    model = UZRScaled(
        base_model_name="Qwen/Qwen2.5-0.5B",
        z_dim=512,
        freeze_base=True,
        num_film_layers=4,
    ).to(device)
    
    # Test tokenization
    tokenizer = model.tokenizer
    text = "Hello, world!"
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    print(f"Input text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Shape: {tokens.shape}\n")
    
    # Test forward pass
    z = model.init_z(batch_size=1).to(device)
    print(f"z shape: {z.shape}")
    print(f"z L1 norm: {torch.sum(torch.abs(z)).item():.4f}\n")
    
    with torch.no_grad():
        logits = model(tokens, z)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]\n")
    
    # Test inner adaptation
    print("Testing inner adaptation...")
    context = ["The capital of France is", "Paris"]
    input_text = tokenizer(context[0], return_tensors="pt").input_ids.to(device)
    target_text = tokenizer(context[1], return_tensors="pt").input_ids.to(device)
    
    # Pad to same length
    max_len = max(input_text.size(1), target_text.size(1))
    input_text = F.pad(input_text, (0, max_len - input_text.size(1)), value=tokenizer.pad_token_id)
    target_text = F.pad(target_text, (0, max_len - target_text.size(1)), value=-100)
    
    z_adapted = inner_adapt_z(
        model, input_text, target_text, z,
        steps=3, lr=0.5, lam=1e-3
    )
    
    print(f"Adapted z L1 norm: {torch.sum(torch.abs(z_adapted)).item():.4f}")
    print(f"Change: {torch.sum(torch.abs(z_adapted - z)).item():.4f}\n")
    
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)
