
# UZR — Universal Z‑Rule Runner (Prototype)

**Goal:** A single network `f_θ(x, z)` that handles language-style rules (formats/grammar/transformations) by **adapting only a small latent `z`** from a few context pairs, then answering queries with that `z`. It supports **long multi‑turn** sessions via split state (`z_slow + z_fast`), **proximal (L1) regularization**, **confidence‑gated updates**, and a **compressed replay memory**.

## Core equations

**Rule estimation (few steps, inference-time):**
\(
z^* = \arg\min_z \sum_{(x,y)\in C} \ell(f_\theta(x,z), y) + \lambda \mathcal{R}(z)
\)

**Prediction:** \(\hat y = f_\theta(x_q, z^*)\)

**Long-run stability:** split \(z = z_{\text{slow}} + z_{\text{fast}}\), proximal updates with soft‑threshold, confidence‑gating, periodic replay.

---

## Files
- `uzr/model.py` — Tiny Transformer encoder with **FiLM** conditioning by `z`, readouts for character tokens.
- `uzr/memory.py` — Compressed memory (sketch), cosine‑similarity retrieval, replay sampler, soft‑threshold prox.
- `uzr/tasks.py` — Synthetic **language‑like rule families** (prefix/suffix, case rules, delimiter formatting, mirror, Caesar shifts, digit masking, template fill, etc.).
- `uzr/train_meta.py` — Meta‑learning loop: sample a task → build few‑shot context `C` → do **z‑only inner updates** → optimize θ by query loss.
- `uzr/infer_longrun.py` — Demonstrates **10k‑turn** style loop (shortened in code), maintaining `z_slow`, per‑turn `z_fast`, memory replay, branching heuristic.

> This is a compact educational prototype (few hundred lines). It runs on CPU; to speed up, use a GPU with PyTorch installed.

## Quick start

```bash
# (Recommended) Python 3.10+
pip install -r requirements.txt
# Meta-train θ (toy)
python -m uzr.train_meta --steps 500 --device cpu

# Long-run inference demo (uses saved θ if available)
python -m uzr.infer_longrun --turns 200 --device cpu
```

## Notes
- The model is **small** (default hidden=256, z_dim=128). Adjust with CLI flags.
- For clarity, we keep tokenization simple (byte level) and **teacher forcing** losses.
- Extend `tasks.py` with your own rule families to move closer to real dialogues.
- The **proximal L1** on `z` enforces **rule sparsity/cleanliness**.
- Confidence metric uses negative loss / entropy; plug in your preferred estimator.
- Memory stores a **sketch**: averaged token embeddings + `z_slow` snapshot for retrieval.
