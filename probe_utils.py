import torch

def probe_grad_norm_fix(model, Xc, Yc, Mc, z_slow, z_fast, lam=1e-3, seq_ce_loss_fn=None, cg_mark_step_fn=None, use_mark_step=False):
    """
    Robust gradient probe for z_fast:
    - Uses a detached+cloned leaf tensor to ensure requires_grad=True even if the original
      z_fast was produced under a no_grad block (e.g., after soft-threshold).
    - Accepts the CE loss function and optional cudagraph mark_step helper to match caller.
    """
    if seq_ce_loss_fn is None:
        # Fallback to basic CE if not provided
        def seq_ce_loss_fn(logits, targets, ignore_index=-100):
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )

    z_fast_var = z_fast.detach().clone().requires_grad_(True)

    z_comb = z_slow + z_fast_var
    z_comb = z_comb if z_comb.dim()==2 else z_comb.unsqueeze(0).expand(Xc.size(0), -1)

    if cg_mark_step_fn is not None:
        cg_mark_step_fn(bool(use_mark_step))

    logits_c = model(Xc, z_comb, Mc)
    loss_c = seq_ce_loss_fn(logits_c, Yc, ignore_index=-100) + lam * z_fast_var.abs().mean()
    (g,) = torch.autograd.grad(loss_c, z_fast_var, retain_graph=False, allow_unused=False)
    print(f"[probe] grad_norm(z_fast): {g.norm().item():.6e}")
    return g

