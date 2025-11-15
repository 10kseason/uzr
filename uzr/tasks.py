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

import csv
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ===== Utilities / Alphabets =====
ALPH = "abcdefghijklmnopqrstuvwxyz"
ALPH_UP = ALPH.upper()
DIG = "0123456789"

KO_JOSA = [
    "은","는","이","가","을","를","과","와","에서","으로","로","에게","께서","도","만","까지","부터","에","의"
]
EN_STOP = {"the","a","an","and","or","but","to","of","in","on","for","with"}
VOWELS = set("aeiouAEIOU")

PREFIX_VARIANTS = [
    "<<","[[","{{","((","||","##","**","__","~~","==","++","--","//","\\\\","!!","??","::","%%","@@","<@","::=",
]
SUFFIX_VARIANTS = [
    ">>","]]", "}}", "))","||","##","**","__","~~","==","++","--","//","\\\\","!!","??","::","%%","@@","@>","=::",
]
MARKER_VARIANTS = ["|","/","-","~","+","=","::","**","##","//","..",":::"]
INSERT_MARKER_STEPS = [2, 3, 4]
REPEAT_COUNTS = [2, 3, 4]
ROTATE_STEPS = [1, 2, 3, 4]
VOWEL_REPLACE_CHARS = ["*","#","_","~","^"]
EN_VOWEL_ORDER = "aeiou"
DIGIT_WORDS_EN = ["zero","one","two","three","four","five","six","seven","eight","nine"]
DIGIT_WORD_TRANSFORMS = [
    ("en lower", lambda s: s.lower()),
    ("en upper", lambda s: s.upper()),
    ("en title", lambda s: s.title()),
]
KO_DIGITS_SINO = ["영","일","이","삼","사","오","육","칠","팔","구"]


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


def rule_insert_marker_every_k(marker="|", k=2):
    def f(x: str):
        toks = x.split()
        if not toks:
            return x
        out = []
        for idx, tok in enumerate(toks, 1):
            out.append(tok)
            if idx % k == 0 and idx != len(toks):
                out.append(marker)
        return " ".join(out)
    return f, f"insert '{marker}' every {k} tokens"


def rule_repeat_tokens(times=2):
    def f(x: str):
        toks = x.split()
        out = []
        for tok in toks:
            out.extend([tok] * max(1, times))
        return " ".join(out)
    return f, f"repeat tokens ×{times}"


def rule_rotate_tokens(k=1):
    def f(x: str):
        toks = x.split()
        if not toks:
            return x
        shift = k % len(toks)
        if shift == 0:
            return x
        return " ".join(toks[shift:] + toks[:shift])
    return f, f"rotate tokens by {k}"


def rule_sort_by_length(desc=False):
    def f(x: str):
        toks = x.split()
        toks.sort(key=lambda tok: (len(tok), tok), reverse=desc)
        return " ".join(toks)
    return f, ("sort by length desc" if desc else "sort by length asc")


def rule_replace_vowels(rep_char="*"):
    def sub(ch: str) -> str:
        return rep_char if ch in VOWELS else ch
    def f(x: str):
        return "".join(sub(c) for c in x)
    return f, f"replace vowels with '{rep_char}'"


def rule_digit_to_word(words=None, transform=None, label="en words"):
    table = words if words is not None else DIGIT_WORDS_EN
    xf = transform if transform is not None else (lambda s: s)
    num_pat = re.compile(r"\d")
    def repl(match: re.Match) -> str:
        digit = int(match.group(0))
        return xf(table[digit % len(table)])
    def f(x: str):
        return num_pat.sub(repl, x)
    return f, f"digits → {label}"


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


def rule_ko_digits_to_sino(delimiter=""):
    def f(x: str):
        out = []
        total = len(x)
        for idx, ch in enumerate(x):
            if ch.isdigit():
                word = KO_DIGITS_SINO[int(ch)]
                out.append(word)
                if delimiter and idx + 1 < total and x[idx + 1].isdigit():
                    out.append(delimiter)
            else:
                out.append(ch)
        return "".join(out)
    desc = "ko: digits → sino" if not delimiter else f"ko: digits → sino delim '{delimiter}'"
    return f, desc


def rule_ko_mark_verbs(prefix="[동]", suffix="[/동]"):
    pat = re.compile(r"([가-힣]+(?:다|요))")
    def f(x: str):
        return pat.sub(lambda m: f"{prefix}{m.group(1)}{suffix}", x)
    return f, f"ko: mark verbs via {prefix}…{suffix}"


def rule_ko_mark_batchim(prefix="[받]", suffix="[/받]"):
    def f(x: str):
        toks = x.split()
        out = []
        for tok in toks:
            last = tok[-1] if tok else ""
            if last and "\uac00" <= last <= "\ud7a3" and has_jongseong(last):
                out.append(f"{prefix}{tok}{suffix}")
            else:
                out.append(tok)
        return " ".join(out)
    return f, "ko: highlight batchim syllables"


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


def rule_en_mark_ing_words(prefix="[ing]", suffix="[/ing]"):
    pat = re.compile(r"\b([A-Za-z]+ing)\b")
    def f(x: str):
        return pat.sub(lambda m: f"{prefix}{m.group(1)}{suffix}", x)
    return f, f"en: mark -ing words via {prefix}…{suffix}"


def rule_en_vowel_shift(shift=1):
    def shift_char(ch: str) -> str:
        lower = ch.lower()
        if lower in EN_VOWEL_ORDER:
            idx = EN_VOWEL_ORDER.index(lower)
            repl = EN_VOWEL_ORDER[(idx + shift) % len(EN_VOWEL_ORDER)]
            return repl.upper() if ch.isupper() else repl
        return ch
    def f(x: str):
        return "".join(shift_char(c) for c in x)
    return f, f"en: vowel shift +{shift}"


def rule_en_uppercase_short_words(max_len=3):
    def f(x: str):
        toks = x.split()
        return " ".join(tok.upper() if tok.isalpha() and len(tok) <= max_len else tok for tok in toks)
    return f, f"en: short tokens≤{max_len} upper"


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


PREFIX_SUFFIX_FACTORIES = [
    (lambda prefix=prefix, suffix=suffix: wrap2(rule_prefix_suffix(prefix=prefix, suffix=suffix)))
    for prefix in PREFIX_VARIANTS
    for suffix in SUFFIX_VARIANTS
]
MARKER_INSERT_FACTORIES = [
    (lambda marker=marker, step=step: wrap2(rule_insert_marker_every_k(marker=marker, k=step)))
    for marker in MARKER_VARIANTS
    for step in INSERT_MARKER_STEPS
]
REPEAT_FACTORIES = [
    (lambda times=times: wrap2(rule_repeat_tokens(times=times)))
    for times in REPEAT_COUNTS
]
ROTATE_FACTORIES = [
    (lambda shift=shift: wrap2(rule_rotate_tokens(k=shift)))
    for shift in ROTATE_STEPS
]
LENGTH_SORT_FACTORIES = [
    (lambda desc=desc: wrap2(rule_sort_by_length(desc=desc)))
    for desc in (False, True)
]
VOWEL_FACTORIES = [
    (lambda rep=rep: wrap2(rule_replace_vowels(rep_char=rep)))
    for rep in VOWEL_REPLACE_CHARS
]
DIGIT_FACTORIES = [
    (lambda label=label, xf=xf: wrap2(rule_digit_to_word(words=DIGIT_WORDS_EN, transform=xf, label=label)))
    for label, xf in DIGIT_WORD_TRANSFORMS
]

RULE_FACTORIES_BASE_CORE = [
    lambda: wrap2(rule_prefix_suffix(prefix=random.choice(["<<","[[","(" ]), suffix=random.choice([">>","]]",")"]))),
    lambda: wrap2(rule_uppercase_every_k(k=random.randint(2, 3))),
    lambda: wrap2(rule_reverse_tokens()),
    lambda: wrap2(rule_surround_numbers(l=random.choice(["{","[","<"]), r=random.choice(["}","]",">"]))),
    lambda: wrap2(rule_toggle_case_ascii(step=random.randint(2, 3))),
    lambda: wrap2(rule_sort_numeric_tokens(ascending=random.random() < 0.5)),
    lambda: wrap2(rule_dedupe_preserve_order()),
    lambda: wrap2(rule_caesar(shift=random.randint(1, 3))),
]

RULE_FACTORIES_BASE = (
    RULE_FACTORIES_BASE_CORE
    + PREFIX_SUFFIX_FACTORIES
    + MARKER_INSERT_FACTORIES
    + REPEAT_FACTORIES
    + ROTATE_FACTORIES
    + LENGTH_SORT_FACTORIES
    + VOWEL_FACTORIES
    + DIGIT_FACTORIES
)


KO_VERB_MARKERS = [
    ("[동]","[/동]"),
    ("<verb>","</verb>"),
    ("{V}","{/V}"),
]
KO_BATCHIM_MARKERS = [
    ("[받]","[/받]"),
    ("<jong>","</jong>"),
    ("{J}","{/J}"),
]
KO_DIGIT_DELIMS = ["", " ", "/"]

RULE_FACTORIES_KO = [
    lambda: wrap2(rule_ko_emphasize_josa()),
    lambda: wrap2(rule_ko_append_yo()),
    lambda: wrap2(rule_ko_kdigit_box()),
    lambda: (rule_ko_date_to_iso()[0], "ko: date → ISO", lambda n_tokens=6: sample_dates_ko(n=2)),
    lambda: wrap2(rule_ko_decompose_jamo(sep="/")),
    lambda: wrap2(rule_ko_compose_jamo(sep="/")),
    lambda: (rule_ko_fix_josa()[0], "ko: josa auto-correct", lambda n_tokens=6: sample_ko_with_josa(n=2)),
] + [
    (lambda prefix=prefix, suffix=suffix: wrap2(rule_ko_mark_verbs(prefix=prefix, suffix=suffix)))
    for prefix, suffix in KO_VERB_MARKERS
] + [
    (lambda prefix=prefix, suffix=suffix: wrap2(rule_ko_mark_batchim(prefix=prefix, suffix=suffix)))
    for prefix, suffix in KO_BATCHIM_MARKERS
] + [
    (lambda delim=delim: wrap2(rule_ko_digits_to_sino(delimiter=delim)))
    for delim in KO_DIGIT_DELIMS
]


EN_ING_MARKERS = [
    ("[ing]","[/ing]"),
    ("<ING>","</ING>"),
    ("{ing}","{/ing}"),
]
EN_VOWEL_SHIFTS = [1, 2, 3]
EN_SHORT_WORD_LIMITS = [2, 3, 4]

RULE_FACTORIES_EN = [
    lambda: wrap2(rule_en_title_case_skip_stops()),
    lambda: wrap2(rule_en_replace_and_amp()),
    lambda: (rule_en_date_to_iso()[0], "en: date → ISO", lambda n_tokens=6: sample_dates_en(n=2)),
    lambda: wrap2(rule_en_pluralize_simple()),
    lambda: wrap2(rule_en_past_tense_simple()),
    lambda: wrap2(rule_en_hyphenate_vcv()),
    lambda: wrap2(rule_en_american_to_british()),
] + [
    (lambda prefix=prefix, suffix=suffix: wrap2(rule_en_mark_ing_words(prefix=prefix, suffix=suffix)))
    for prefix, suffix in EN_ING_MARKERS
] + [
    (lambda shift=shift: wrap2(rule_en_vowel_shift(shift=shift)))
    for shift in EN_VOWEL_SHIFTS
] + [
    (lambda limit=limit: wrap2(rule_en_uppercase_short_words(max_len=limit)))
    for limit in EN_SHORT_WORD_LIMITS
]


# ===== External Dataset / KoBERT helpers =====
DATASET_MIT_DEFAULT = Path(__file__).resolve().parent / "dataset-mit" / "mmlu_KO-KR.csv"


class DatasetMiTSampler:
    """Sample supervised QA pairs from dataset-mit (KMMLU-style) with optional KoBERT hints."""

    OPTION_KEYS = ("A", "B", "C", "D")

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        mix_prob: float = 0.0,
        use_kobert_hint: bool = False,
        kobert_dir: Optional[Path] = None,
        kobert_device: str = "cpu",
        kobert_max_seq_len: int = 384,
    ):
        self.csv_path = Path(csv_path) if csv_path else DATASET_MIT_DEFAULT
        self.mix_prob = max(0.0, min(1.0, float(mix_prob)))
        self.rows = self._load_rows(self.csv_path)
        if not self.rows:
            raise ValueError(f"dataset-mit CSV ({self.csv_path})에서 유효한 행을 찾지 못했습니다.")
        self.teacher = None
        if use_kobert_hint:
            try:
                self.teacher = KoBERTTeacher(
                    model_dir=kobert_dir if kobert_dir is not None else Path(__file__).resolve().parent / "kobert",
                    device=kobert_device,
                    max_seq_len=kobert_max_seq_len,
                )
            except Exception as exc:
                # Disable hints but keep training running.
                print(f"[DatasetMiTSampler] KoBERT 힌트 비활성화: {exc}")
                self.teacher = None

    def should_use(self) -> bool:
        return self.mix_prob > 0.0 and bool(self.rows) and random.random() < self.mix_prob

    def sample(
        self, n_context: int, n_query: int
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
        if not self.rows:
            raise RuntimeError("dataset-mit rows are empty; cannot sample.")
        total = max(0, int(n_context) + int(n_query))
        if total == 0:
            return [], [], "dataset-mit: empty request"
        picks = self._pick_rows(total)
        C_rows = picks[:n_context]
        Q_rows = picks[n_context:]
        desc_subject = (C_rows[0]["subject"] if C_rows else picks[0]["subject"]) or "mmlu-ko"
        C_pairs = [self._row_to_pair(row) for row in C_rows]
        Q_pairs = [self._row_to_pair(row) for row in Q_rows]
        return C_pairs, Q_pairs, f"dataset-mit:{desc_subject}"

    def _pick_rows(self, total: int) -> List[Dict[str, str]]:
        if total <= 0:
            return []
        if len(self.rows) >= total:
            return random.sample(self.rows, total)
        return [random.choice(self.rows) for _ in range(total)]

    def _row_to_pair(self, row: Dict[str, str]) -> Tuple[str, str]:
        prompt_lines = [row["question"].strip(), "선택지:"]
        for key in self.OPTION_KEYS:
            opt = row["options"].get(key)
            if opt:
                prompt_lines.append(f"{key}. {opt}")
        if row["subject"]:
            prompt_lines.append(f"[과목] {row['subject']}")
        hint = self._teacher_hint(row)
        if hint:
            prompt_lines.append(f"KoBERT 예측: {hint}")
        prompt_lines.append("정답을 A/B/C/D 중 하나와 설명으로 답하세요.")
        target = self._format_target(row)
        return "\n".join(prompt_lines), target

    def _format_target(self, row: Dict[str, str]) -> str:
        letter = row["answer"]
        text = row["options"].get(letter, "")
        if text:
            return f"{letter}. {text}"
        return letter

    def _teacher_hint(self, row: Dict[str, str]) -> Optional[str]:
        if self.teacher is None:
            return None
        pred = self.teacher.predict_letter(row["question"], row["options"])
        if pred is None:
            return None
        letter, prob = pred
        opt = row["options"].get(letter, "")
        if opt:
            return f"{letter} ({prob:.2f}) → {opt}"
        return f"{letter} ({prob:.2f})"

    @staticmethod
    def _load_rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"dataset-mit CSV를 찾을 수 없습니다: {path}")
        rows: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                question = (raw.get("Question") or "").strip()
                if not question:
                    continue
                options = {}
                for key in DatasetMiTSampler.OPTION_KEYS:
                    val = raw.get(key)
                    if val is not None:
                        text = str(val).strip()
                        if text:
                            options[key] = text
                answer = (raw.get("Answer") or "").strip().upper()
                if answer not in options:
                    continue
                rows.append({
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "subject": (raw.get("Subject") or "").strip(),
                })
        return rows


class KoBERTTeacher:
    """Lightweight masked-LM helper that scores options via KoBERT."""

    OPTION_KEYS = ("A", "B", "C", "D")

    def __init__(self, model_dir: Path, device: str = "cpu", max_seq_len: int = 384):
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers 패키지가 필요합니다 (KoBERTTeacher).") from exc
        import torch

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model.eval()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA가 지원되지 않는 환경에서 KoBERTTeacher에 cuda를 요청했습니다.")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.max_seq_len = max(32, int(max_seq_len))
        self.option_token_ids = self._prepare_option_token_ids()
        self._cache: Dict[str, Optional[Tuple[str, float]]] = {}

    def _prepare_option_token_ids(self) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        unk_id = self.tokenizer.unk_token_id
        for label in self.OPTION_KEYS:
            tid = self.tokenizer.convert_tokens_to_ids(label)
            if tid is None or (unk_id is not None and tid == unk_id):
                continue
            ids[label] = tid
        return ids

    def predict_letter(self, question: str, options: Dict[str, str]) -> Optional[Tuple[str, float]]:
        cache_key = self._cache_key(question, options)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.option_token_ids:
            self._cache[cache_key] = None
            return None
        prompt = self._build_prompt(question, options)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )
        mask_positions = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            self._cache[cache_key] = None
            return None
        mask_index = int(mask_positions[0].item())
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        mask_logits = logits[0, mask_index]
        scored = []
        for label, token_id in self.option_token_ids.items():
            if not options.get(label):
                continue
            scored.append((label, float(mask_logits[token_id].item())))
        if not scored:
            self._cache[cache_key] = None
            return None
        max_score = max(score for _, score in scored)
        exp_scores = [math.exp(score - max_score) for _, score in scored]
        denom = sum(exp_scores) or 1.0
        probs = [val / denom for val in exp_scores]
        best_idx = max(range(len(scored)), key=lambda idx: scored[idx][1])
        result = (scored[best_idx][0], probs[best_idx])
        self._cache[cache_key] = result
        return result

    def _build_prompt(self, question: str, options: Dict[str, str]) -> str:
        lines = [question.strip(), "선택지:"]
        for key in self.OPTION_KEYS:
            opt = options.get(key)
            if opt:
                lines.append(f"{key}. {opt}")
        lines.append("정답: [MASK]")
        return "\n".join(lines)

    def _cache_key(self, question: str, options: Dict[str, str]) -> str:
        # Simple deterministic key (question + concatenated options)
        opt_tuple = "|".join(f"{k}:{options.get(k,'')}" for k in self.OPTION_KEYS)
        return f"{question}##{opt_tuple}"


# ===== Sample Task =====
def sample_task(
    n_context=6,
    n_query=8,
    n_tokens=5,
    dataset_sampler: Optional[DatasetMiTSampler] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """Build few-shot contexts and queries with description."""
    if dataset_sampler is not None and dataset_sampler.should_use():
        return dataset_sampler.sample(n_context=n_context, n_query=n_query)

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
