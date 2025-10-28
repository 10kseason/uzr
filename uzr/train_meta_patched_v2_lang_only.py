import random
import sys

try:
    # Import the base trainer
    import train_meta_patched_v2_more_logical as base
except Exception:
    # Fallback in case module path differs
    from . import train_meta_patched_v2_more_logical as base  # type: ignore

try:
    # Access task factories/generators
    import tasks as T
except Exception:
    from uzr import tasks as T  # type: ignore


def sample_task_language_only(n_context=6, n_query=8, n_tokens=5):
    """Sample only language-specific tasks (EN/KO). No BASE or MATH.

    - Picks a language bucket from {en, ko}
    - Uses only that language's rule factories (no BASE/MATH)
    - Keeps the advanced/composition behavior similar to the base sampler
    """
    lang = random.choice(["en", "ko"])  # JA is not enabled in current tasks

    if lang == "en":
        pool = T.RULE_FACTORIES_EN
        gen_default = T.sample_en
    else:
        pool = T.RULE_FACTORIES_KO
        gen_default = T.sample_ko

    advanced_pick = random.random() < 0.55
    gen = gen_default
    desc = ""

    if advanced_pick:
        f1, d1, g1 = random.choice(pool)()
        # mild composition from the SAME language set only
        if random.random() < 0.35 and len(pool) > 1:
            f2, d2, _ = random.choice(pool)()
            f = T.compose([f1, f2])
            desc = f"{d1} + {d2}"
        else:
            f = f1
            desc = d1
        if g1 is not None:
            gen = g1
    else:
        k = random.choice([2, 3])
        picks = [random.choice(pool)() for _ in range(k)]
        fs = [p[0] for p in picks]
        f = T.compose(fs)
        desc = " + ".join(p[1] for p in picks)

    C, Q = [], []
    for _ in range(n_context):
        s = gen(n_tokens=n_tokens)
        C.append((s, f(s)))
    for _ in range(n_query):
        s = gen(n_tokens=n_tokens)
        Q.append((s, f(s)))

    return C, Q, f"{lang}: {desc}"


def main():
    # Monkeypatch the sampler in the base trainer
    base.sample_task = sample_task_language_only  # type: ignore
    base.main()


if __name__ == "__main__":
    main()

