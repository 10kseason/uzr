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


def sample_task_logic_only(n_context=6, n_query=8, n_tokens=5):
    """Sample only logic (BASE/MATH) tasks. No language-specific rules.

    - Uses BASE rule factories and MATH only
    - Composition draws exclusively from BASE rules
    - Generator defaults to generic token sequence unless a rule overrides it
    """
    pool_logic = T.RULE_FACTORIES_BASE + [T.FACTORY_MATH]
    gen_default = T.make_sequence

    advanced_pick = random.random() < 0.55
    gen = gen_default
    desc = ""

    if advanced_pick:
        f1, d1, g1 = random.choice(pool_logic)()
        # mild composition with one safe post-step from BASE only
        if random.random() < 0.35:
            f2, d2, _ = random.choice(T.RULE_FACTORIES_BASE)()
            f = T.compose([f1, f2])
            desc = f"{d1} + {d2}"
        else:
            f = f1
            desc = d1
        if g1 is not None:
            gen = g1
    else:
        k = random.choice([2, 3])
        picks = [random.choice(T.RULE_FACTORIES_BASE)() for _ in range(k)]
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

    return C, Q, f"base: {desc}"


def main():
    # Force logic-only sampling
    base.sample_task = sample_task_logic_only  # type: ignore
    # Force language to be treated as BASE inside the trainer
    base.detect_lang_from_text = lambda _s: "base"  # type: ignore

    # Disable language-specific auxiliary losses by default if not provided
    # (identity QA weight and KO/EN alignment batches)
    argv = list(sys.argv)
    if "--id_weight" not in argv:
        argv += ["--id_weight", "0.0"]
    if "--aux_baux" not in argv:
        argv += ["--aux_baux", "0"]

    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()

