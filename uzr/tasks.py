#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KO/EN task generators and rule library (JA optional, commented).

Provides:
- Rich sample generators for Korean/English
- Rule functions returning (callable, description)
- Rule factories for composing harder tasks
- sample_task() for few-shot generation (contexts + queries + description)
"""

import random
import re
import unicodedata
from typing import List, Tuple, Callable, Optional

# ===== Utilities / Alphabets =====
ALPH = "abcdefghijklmnopqrstuvwxyz"
ALPH_UP = ALPH.upper()
DIG = "0123456789"

KO_JOSA = [
    "은","는","이","가","을","를","과","와","에서","으로","로","에게","께서","도","만","까지","부터","에","의"
]
EN_STOP = {"the","a","an","and","or","but","to","of","in","on","for","with"}


def compose(funcs: List[Callable[[str], str]]) -> Callable[[str], str]:
    def f(x: str) -> str:
        for g in funcs:
            x = g(x)
        return x
    return f


def wrap2(rule_ret: Tuple[Callable[[str], str], str]) -> Tuple[Callable[[str], str], str, Optional[Callable]]:
    f, d = rule_ret
    return f, d, None


def rand_word(min_len=3, max_len=8, alph=ALPH):
    L = random.randint(min_len, max_len)
    return "".join(random.choice(alph) for _ in range(L))


def rand_token():
    r = random.random()
    if r < 0.5:
        return rand_word()
    elif r < 0.8:
        return "".join(random.choice(DIG) for _ in range(random.randint(1, 5)))
    else:
        return random.choice(["-", "_", ":", "/", ".", ","])


def make_sequence(n_tokens=5):
    return " ".join(rand_token() for _ in range(n_tokens))


# ===== Base Generators =====
def sample_en(n_tokens=5):
    words = [
        "time","data","system","model","agent","rule","space","music","river","node",
        "graph","token","future","night","dream","light","plan","verify","deploy","iterate",
        "merge","align","prompt","repair","secure","channel","process","queue","cache","stream",
        "the","a","an","and","of","in","on","to","with",
    ]
    return " ".join(random.choice(words) for _ in range(n_tokens))


def sample_ko(n_tokens=5):
    chunks = [
        "오늘","회의","준비","완료","모델","규칙","공간","로그","분석","결과",
        "배포","계획","확인","추론","파이프라인","정리","데이터","전처리","품질","지표",
        "함께","검토","예정","진행","공유","업데이트","점검","완료"
    ]
    return " ".join(random.choice(chunks) for _ in range(n_tokens))


# ===== Structured Generators =====
MONTHS_EN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def sample_dates_en(n=1):
    outs = []
    for _ in range(n):
        y = random.randint(2018, 2031)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        f = random.choice([
            f"{m}/{d}/{y}",  # US
            f"{y}-{m:02d}-{d:02d}",  # ISO
            f"{MONTHS_EN[m-1]} {d}, {y}",  # Mon D, YYYY
        ])
        outs.append(f"meet at {f} with {rand_word()}")
    return " ".join(outs)


def sample_dates_ko(n=1):
    outs = []
    for _ in range(n):
        y = random.randint(2018, 2031)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        f = random.choice([
            f"{y}년 {m}월 {d}일",
            f"{y}.{m:02d}.{d:02d}",
            f"{m}월 {d}일 {y}년",
        ])
        outs.append(f"일정 {f} 확인")
    return " ".join(outs)


def sample_expr(n_terms=3):
    ops = ["+", "-", "*"]
    nums = [str(random.randint(0, 99)) for _ in range(n_terms)]
    expr = nums[0]
    for i in range(1, n_terms):
        expr += random.choice(ops) + nums[i]
    if random.random() < 0.6:
        expr = f"calc({expr})"
    return expr


def sample_codeswitch_ko_en(n_tokens=6):
    ko = ["오늘","회의","모델","규칙","데이터","결과","배포"]
    en = ["token","rule","graph","system","light","music","node"]
    seq = []
    for _ in range(n_tokens):
        seq.append(random.choice(ko if random.random() < 0.5 else en))
    return " ".join(seq)


# ===== Language-agnostic Rules =====
def rule_prefix_suffix(prefix="<<", suffix=">>"):
    def f(x: str):
        return f"{prefix}{x}{suffix}"
    return f, f"prefix='{prefix}' suffix='{suffix}'"


def rule_uppercase_every_k(k=2):
    def f(x: str):
        toks = x.split()
        for i in range(0, len(toks), k):
            toks[i] = toks[i].upper()
        return " ".join(toks)
    return f, f"uppercase every {k}th token"


def rule_reverse_tokens():
    def f(x: str):
        return " ".join(reversed(x.split()))
    return f, "reverse token order"


def rule_surround_numbers(l="[", r="]"):
    num_pat = re.compile(r"\d+")
    def f(x: str):
        return num_pat.sub(lambda m: f"{l}{m.group(0)}{r}", x)
    return f, f"surround digits with {l}{r}"


def rule_toggle_case_ascii(step=2):
    def tog(s: str):
        out = []
        for i, ch in enumerate(s):
            if not ch.isascii() or not ch.isalpha():
                out.append(ch)
                continue
            if (i // step) % 2 == 0:
                out.append(ch.swapcase())
            else:
                out.append(ch)
        return "".join(out)
    def f(x: str):
        return " ".join(tog(t) for t in x.split())
    return f, f"toggle ASCII case every {step}"


def rule_sort_numeric_tokens(ascending=True):
    def f(x: str):
        toks = x.split()
        nums = [int(t) for t in toks if t.isdigit()]
        others = [t for t in toks if not t.isdigit()]
        nums.sort(reverse=not ascending)
        return " ".join(others + [str(n) for n in nums])
    return f, ("sort numbers asc" if ascending else "sort numbers desc")


def rule_dedupe_preserve_order():
    def f(x: str):
        out = []
        prev = None
        for tok in x.split():
            if tok != prev:
                out.append(tok)
            prev = tok
        return " ".join(out)
    return f, "dedupe consecutive tokens"


def rule_caesar(shift=1):
    def shift_char(c: str) -> str:
        if c in ALPH:
            i = (ALPH.index(c) + shift) % 26
            return ALPH[i]
        if c in ALPH_UP:
            i = (ALPH_UP.index(c) + shift) % 26
            return ALPH_UP[i]
        return c
    def f(x: str):
        return "".join(shift_char(c) for c in x)
    return f, f"caesar shift {shift}"


# ===== Korean Rules =====
def rule_ko_emphasize_josa():
    pat = re.compile(r"\b(" + "|".join(map(re.escape, KO_JOSA)) + r")\b")
    def f(x: str):
        return pat.sub(lambda m: f"«{m.group(1)}»", x)
    return f, "ko: emphasize josa"


def rule_ko_append_yo():
    def f(x: str):
        s = x.strip()
        return s if s.endswith("요") else s + " 요"
    return f, "ko: append 요"


def rule_ko_kdigit_box(l="[", r="]"):
    num_pat = re.compile(r"\d+")
    def f(x: str):
        return num_pat.sub(lambda m: f"{l}{m.group(0)}{r}", x)
    return f, "ko: wrap digits with brackets"


def rule_ko_date_to_iso():
    re_dot = re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})")
    re_kr = re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")
    def f(x: str):
        x = re_dot.sub(lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", x)
        x = re_kr.sub(lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", x)
        return x
    return f, "ko: date → ISO"


_CHO = [chr(c) for c in range(0x1100, 0x1113)]
_JUNG = [chr(c) for c in range(0x1161, 0x1176)]
_JONG = ["\0"] + [chr(c) for c in range(0x11A8, 0x11C3)]


def has_jongseong(syll: str) -> bool:
    code = ord(syll)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return jong != 0
    return False


def rule_ko_decompose_jamo(sep="/"):
    def decompose(ch: str) -> str:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            s_index = code - 0xAC00
            cho = _CHO[s_index // 588]
            jung = _JUNG[(s_index % 588) // 28]
            jong_idx = s_index % 28
            jong = _JONG[jong_idx] if jong_idx else ""
            parts = [cho, jung] + ([jong] if jong else [])
            return sep.join(parts)
        return ch
    def f(x: str):
        return " ".join("".join(decompose(ch) for ch in token) for token in x.split())
    return f, "ko: decompose jamo"


def rule_ko_compose_jamo(sep="/"):
    def compose_token(tok: str) -> str:
        parts = tok.split(sep)
        if 2 <= len(parts) <= 3 and all(parts):
            try:
                cho = _CHO.index(parts[0]); jung = _JUNG.index(parts[1])
                jong = _JONG.index(parts[2]) if len(parts) == 3 else 0
                code = 0xAC00 + (cho * 588) + (jung * 28) + jong
                return chr(code)
            except ValueError:
                return tok
        return tok
    def f(x: str):
        return " ".join(compose_token(t) for t in x.split())
    return f, "ko: compose jamo"


def rule_ko_fix_josa():
    pat = re.compile(r"([가-힣])\((은/는|이/가|을/를)\)")
    def pick(ch: str, pair: str) -> str:
        jong = has_jongseong(ch)
        if pair == "은/는":
            return "은" if jong else "는"
        if pair == "이/가":
            return "이" if jong else "가"
        if pair == "을/를":
            return "을" if jong else "를"
        return pair
    def f(x: str):
        return pat.sub(lambda m: m.group(1) + pick(m.group(1), m.group(2)), x)
    return f, "ko: josa auto-correct"


def sample_ko_with_josa(n=2):
    nouns = ["사과","바나나","학생","시간","결과","모델","규칙","예전","음악"]
    templates = [
        "{n}(은/는) 준비 완료",
        "{n}(이/가) 필요해요",
        "{n}(을/를) 확인",
    ]
    return " ".join(random.choice(templates).format(n=random.choice(nouns)) for _ in range(n))


# ===== English Rules (richer) =====
def rule_en_title_case_skip_stops():
    def title_word(w):
        if w.lower() in EN_STOP:
            return w.lower()
        return w[:1].upper() + w[1:]
    def f(x: str):
        toks = x.split()
        return " ".join(title_word(w) for w in toks)
    return f, "en: title case (skip stops)"


def rule_en_replace_and_amp():
    def f(x: str):
        return x.replace(" and ", " & ")
    return f, "en: replace 'and' with '&'"


def rule_en_date_to_iso():
    mon_map = {m: i + 1 for i, m in enumerate(MONTHS_EN)}
    re_us = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
    re_mon = re.compile(r"\b([A-Z][a-z]{2}) (\d{1,2}), (\d{4})\b")
    def f(x: str):
        x = re_us.sub(lambda m: f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}", x)
        x = re_mon.sub(lambda m: f"{int(m.group(3)):04d}-{mon_map[m.group(1)]:02d}-{int(m.group(2)):02d}", x)
        return x
    return f, "en: date → ISO"


def rule_en_pluralize_simple():
    def pluralize(w):
        if not w.isalpha():
            return w
        wl = w.lower()
        if re.search(r"(s|x|z|ch|sh)$", wl):
            return w + "es"
        if re.search(r"[^aeiou]y$", wl):
            return w[:-1] + "ies"
        return w + "s"
    def f(x: str):
        return " ".join(pluralize(t) for t in x.split())
    return f, "en: pluralize (simple)"


def rule_en_past_tense_simple():
    def past(w):
        if not w.isalpha():
            return w
        wl = w.lower()
        if wl.endswith("e"):
            return w + "d"
        if re.search(r"[^aeiou]y$", wl):
            return w[:-1] + "ied"
        return w + "ed"
    def f(x: str):
        return " ".join(past(t) for t in x.split())
    return f, "en: past tense (simple)"


def rule_en_hyphenate_vcv():
    vowels = set("aeiouAEIOU")
    def hyph(w):
        if not w.isalpha() or len(w) < 4:
            return w
        for i in range(1, len(w) - 1):
            if (w[i - 1] not in vowels) and (w[i] in vowels) and (w[i + 1] not in vowels):
                return w[: i + 1] + "-" + w[i + 1 :]
        return w
    def f(x: str):
        return " ".join(hyph(t) for t in x.split())
    return f, "en: hyphenate VCV"


def rule_en_american_to_british():
    MAP = {"color": "colour", "favorite": "favourite", "organize": "organise", "analyze": "analyse"}
    pat = re.compile(r"\b(" + "|".join(MAP.keys()) + r")\b", re.I)
    def f(x: str):
        def repl(m):
            w = m.group(1)
            rep = MAP.get(w.lower(), w)
            if w.istitle():
                return rep.title()
            if w.isupper():
                return rep.upper()
            return rep
        return pat.sub(repl, x)
    return f, "en: American→British (demo)"


# ===== Japanese Rules (optional) =====
def rule_eval_simple_math():
    # Evaluate simple expressions optionally wrapped in calc(...)
    expr_pat = re.compile(r"calc\(([^)]+)\)|([0-9+\- *]+)")
    def f(x: str):
        def _eval(e: str) -> str:
            e = re.sub(r"[^0-9+\- *]", "", e)
            try:
                return str(int(eval(e)))  # no variables
            except Exception:
                return e
        def repl(m):
            s = m.group(1) or m.group(2) or ""
            return _eval(s)
        return expr_pat.sub(repl, x)
    return f, "math: eval"


# ===== Rule Factories for sample_task =====
def FACTORY_MATH():
    return (rule_eval_simple_math()[0], "math: eval", lambda n_tokens=6: sample_expr(n_terms=random.randint(2, 4)))


RULE_FACTORIES_BASE = [
    lambda: wrap2(rule_prefix_suffix(prefix=random.choice(["<<","[[","(" ]), suffix=random.choice([">>","]]",")"]))),
    lambda: wrap2(rule_uppercase_every_k(k=random.randint(2, 3))),
    lambda: wrap2(rule_reverse_tokens()),
    lambda: wrap2(rule_surround_numbers(l=random.choice(["{","[","<"]), r=random.choice(["}","]",">"]))),
    lambda: wrap2(rule_toggle_case_ascii(step=random.randint(2, 3))),
    lambda: wrap2(rule_sort_numeric_tokens(ascending=random.random() < 0.5)),
    lambda: wrap2(rule_dedupe_preserve_order()),
    lambda: wrap2(rule_caesar(shift=random.randint(1, 3))),
]


RULE_FACTORIES_KO = [
    lambda: wrap2(rule_ko_emphasize_josa()),
    lambda: wrap2(rule_ko_append_yo()),
    lambda: wrap2(rule_ko_kdigit_box()),
    lambda: (rule_ko_date_to_iso()[0], "ko: date → ISO", lambda n_tokens=6: sample_dates_ko(n=2)),
    lambda: wrap2(rule_ko_decompose_jamo(sep="/")),
    lambda: wrap2(rule_ko_compose_jamo(sep="/")),
    lambda: (rule_ko_fix_josa()[0], "ko: josa auto-correct", lambda n_tokens=6: sample_ko_with_josa(n=2)),
]


RULE_FACTORIES_EN = [
    lambda: wrap2(rule_en_title_case_skip_stops()),
    lambda: wrap2(rule_en_replace_and_amp()),
    lambda: (rule_en_date_to_iso()[0], "en: date → ISO", lambda n_tokens=6: sample_dates_en(n=2)),
    lambda: wrap2(rule_en_pluralize_simple()),
    lambda: wrap2(rule_en_past_tense_simple()),
    lambda: wrap2(rule_en_hyphenate_vcv()),
    lambda: wrap2(rule_en_american_to_british()),
]


# ===== Sample Task =====
def sample_task(n_context=6, n_query=8, n_tokens=5) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """Build few-shot contexts and queries with description."""
    lang = random.choice(["en", "ko", "base"])  # JA deferred
    if lang == "en":
        pool = RULE_FACTORIES_EN + RULE_FACTORIES_BASE[:2] + [FACTORY_MATH]
        gen_default = sample_en
    elif lang == "ko":
        pool = RULE_FACTORIES_KO + RULE_FACTORIES_BASE[:2] + [FACTORY_MATH]
        gen_default = sample_ko
    else:
        pool = RULE_FACTORIES_BASE + [FACTORY_MATH]
        gen_default = make_sequence

    advanced_pick = random.random() < 0.55
    gen = gen_default
    desc = ""

    if advanced_pick:
        f1, d1, g1 = random.choice(pool)()
        if random.random() < 0.35:
            f2, d2, _ = random.choice(RULE_FACTORIES_BASE)()
            f = compose([f1, f2])
            desc = f"{d1} → {d2}"
        else:
            f = f1; desc = d1
        if g1 is not None:
            gen = g1
    else:
        k = random.choice([2, 3])
        picks = [random.choice(RULE_FACTORIES_BASE)() for _ in range(k)]
        fs = [p[0] for p in picks]
        f = compose(fs)
        desc = " → ".join(p[1] for p in picks)

    C, Q = [], []
    for _ in range(n_context):
        s = gen(n_tokens=n_tokens)
        C.append((s, f(s)))
    for _ in range(n_query):
        s = gen(n_tokens=n_tokens)
        Q.append((s, f(s)))

    return C, Q, f"{lang}: {desc}"

