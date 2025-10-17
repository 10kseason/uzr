
import argparse, sys, torch, readline
from uzr.model import UZRModel, ByteTokenizer, soft_threshold, seq_ce_loss
from uzr.memory import CompressedMemory, make_sketch
from uzr.infer_longrun import avg_embed, init_from_retrieval

PROMPT_KO = "사용자> "
PROMPT_BOT = "루리아> "

def generate(model, tok, text, z, device):
    # Teacher-forcing style transform: model predicts token-wise output given input
    X = torch.stack([tok.encode(text)], dim=0).to(device)
    with torch.no_grad():
        logits = model(X, z)
        ids = logits.argmax(dim=-1)[0].tolist()
        return tok.decode(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--max_items", type=int, default=20000)
    ap.add_argument("--identity", default="루리아")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--prox", type=float, default=5e-4)
    ap.add_argument("--inner_steps", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1e-3)
    args = ap.parse_args()

    data = torch.load(args.ckpt, map_location="cpu")
    cfg = data["args"]
    tok = ByteTokenizer(max_len=cfg["max_len"])
    model = UZRModel(tok.vocab_size, d_model=cfg["d_model"], z_dim=cfg["z_dim"], max_len=cfg["max_len"])
    model.load_state_dict(data["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    mem = CompressedMemory(max_items=args.max_items, device=device)
    # seed identity
    from uzr.infer_longrun import seed_identity
    seed_identity(mem, model, tok, device, identity=args.identity)

    z_slow = model.init_z(batch_size=1).to(device)[0] * 0.0

    print("인터랙티브 모드 시작. (Ctrl+C로 종료)")
    while True:
        try:
            user = input(PROMPT_KO)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

        # fast init from retrieval
        enc_avg = avg_embed(model, torch.stack([tok.encode(user)], dim=0).to(device))
        z_fast = init_from_retrieval(mem, enc_avg, z_slow).clone().detach().requires_grad_(True)

        # inner adapt on single-turn "context": try to map identity QAs if detected
        # Here, we use a pseudo-target: identity echo if user asks, else self reconstruction (identity bias)
        target = None
        lower = user.lower()
        if any(q in lower for q in ["who are you", "your name"]) or "누구" in user or "이름" in user or "だれ" in user or "なまえ" in user:
            if "누구" in user or "이름" in user:
                target = f"나는 {args.identity}입니다."
            elif "だれ" in user or "なまえ" in user:
                target = f"わたしは{args.identity}です。"
            else:
                target = f"I am {args.identity}."
        else:
            target = user  # identity-stable echo as a weak prior

        Xc = torch.stack([tok.encode(user)], dim=0).to(device)
        Yc = torch.stack([tok.encode(target)], dim=0).to(device)
        for _ in range(args.inner_steps):
            logits_c = model(Xc, z_slow + z_fast)
            loss_c = seq_ce_loss(logits_c, Yc) + args.lam * torch.mean(torch.abs(z_slow + z_fast))
            g = torch.autograd.grad(loss_c, z_fast, retain_graph=False)[0]
            # fixed step for CLI
            z_fast = z_fast - 0.6 * g

        with torch.no_grad():
            z_fast = soft_threshold(z_fast, args.lam * 0.5)

        # respond
        z = z_slow + z_fast
        out = generate(model, tok, user, z, device)
        print(PROMPT_BOT + out)

        # update slow state and memory
        with torch.no_grad():
            z_slow = z_slow + args.alpha * (z_fast - z_slow)
            z_slow = soft_threshold(z_slow, args.prox)
            key, val = make_sketch(enc_avg, z_slow, meta={"last_user": user, "last_bot": out})
            mem.add(key, val, step=0)

if __name__ == "__main__":
    main()
