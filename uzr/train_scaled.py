#!/usr/bin/env python3
"""
Train UZR Scaled (Qwen 0.5B + z-conditioning)
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from pathlib import Path

# 기존 UZR 모듈들
import sys
sys.path.append(str(Path(__file__).parent))

from uzr_scaled import UZRScaled, seq_ce_loss, inner_adapt_z, soft_threshold
from uzr.tasks import sample_task  # 기존 task generator 재사용
from uzr.memory import CompressedMemory, make_sketch

def prepare_batch(pairs, tokenizer, device, max_length=128):
    """
    Convert (input, output) pairs to tokenized batch
    """
    texts_in = [x for x, _ in pairs]
    texts_out = [y for _, y in pairs]
    
    # Tokenize
    inputs = tokenizer(
        texts_in,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    targets = tokenizer(
        texts_out,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    # Target에서 padding을 -100으로 (loss 계산시 무시)
    target_ids = targets.input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = -100
    
    return inputs.input_ids, target_ids, inputs.attention_mask


def avg_embed(model, input_ids, attention_mask=None):
    """Get average embedding from encoder"""
    with torch.no_grad():
        # Base model의 embedding layer 사용
        embeds = model.base_model.get_input_embeddings()(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            avg = (embeds * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            avg = embeds.mean(dim=1)
        return avg.mean(dim=0)  # batch average


def train_step(model, context_pairs, query_pairs, tokenizer, device, args, mem=None, z_slow=None):
    """
    Single meta-learning step
    """
    # Prepare context batch
    Xc, Yc, mask_c = prepare_batch(context_pairs, tokenizer, device)
    
    # Inner adaptation on context
    if z_slow is None:
        z_init = model.init_z(batch_size=1)[0].to(device)
    else:
        z_init = z_slow.clone()
    
    # Retrieve from memory if available
    if mem is not None:
        enc_avg = avg_embed(model, Xc, mask_c)
        items = mem.retrieve(enc_avg, topk=4)
        if items:
            z_retrieved = torch.stack([it.val["z_slow"] for it in items], dim=0).mean(dim=0)
            z_init = z_init + 0.3 * z_retrieved
    
    z_adapted = inner_adapt_z(
        model, Xc, Yc, z_init,
        steps=args.inner_steps,
        lr=args.inner_lr,
        lam=args.lam,
        attention_mask=mask_c
    )
    
    # Query loss (outer loop)
    Xq, Yq, mask_q = prepare_batch(query_pairs, tokenizer, device)
    logits_q = model(Xq, z_adapted, mask_q)
    loss = seq_ce_loss(logits_q, Yq, ignore_index=-100)
    
    return loss, z_adapted, enc_avg if mem else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--z_dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--inner_lr", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.3)  # z_slow update rate
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--use_memory", action="store_true")
    parser.add_argument("--max_memory", type=int, default=2048)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    print("Initializing model...")
    model = UZRScaled(
        base_model_name=args.model_name,
        z_dim=args.z_dim,
        freeze_base=True,
        num_film_layers=4
    ).to(device)
    
    tokenizer = model.tokenizer
    
    # Optimizer (only FiLM layers + z0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Memory (optional)
    mem = None
    z_slow = None
    if args.use_memory:
        mem = CompressedMemory(max_items=args.max_memory, device=device)
        z_slow = model.init_z(batch_size=1)[0].to(device) * 0.0
    
    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    print(f"Memory: {'enabled' if args.use_memory else 'disabled'}")
    print()
    
    pbar = tqdm(range(args.steps), desc="Training")
    ema_loss = None
    
    for step in pbar:
        # Sample task
        context_pairs, query_pairs, desc = sample_task(
            n_context=6,
            n_query=4,
            n_tokens=5
        )
        
        # Forward
        loss, z_adapted, enc_avg = train_step(
            model, context_pairs, query_pairs, tokenizer, device, args,
            mem=mem, z_slow=z_slow
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        # Update z_slow (if using memory)
        if args.use_memory and z_slow is not None:
            with torch.no_grad():
                z_fast = z_adapted - z_slow
                z_slow = z_slow + args.alpha * z_fast
                z_slow = soft_threshold(z_slow, args.lam * 0.5)
                
                # Add to memory
                key, val = make_sketch(enc_avg, z_slow, meta={"desc": desc})
                mem.add(key, val, step=step)
        
        # Logging
        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val
        
        pbar.set_postfix({
            "loss": f"{loss_val:.3f}",
            "ema": f"{ema_loss:.3f}",
            "task": desc[:30],
            "mem": len(mem.items) if mem else 0
        })
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"uzr_scaled_step_{step+1}.pt")
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "args": vars(args),
                "ema_loss": ema_loss,
            }
            if args.use_memory:
                save_dict["z_slow"] = z_slow
                # Memory는 너무 크니까 저장 안 함 (필요하면 추가)
            
            torch.save(save_dict, ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")
    
    # Final save
    final_path = os.path.join(args.save_dir, "uzr_scaled_final.pt")
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final checkpoint: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
