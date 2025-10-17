# tasks.py — GPT-3.5급 난이도 태스크용 드롭-인 교체본
import random, re, unicodedata
from typing import List, Tuple, Callable, Optional

# ===== Utilities / Alphabets =====
ALPH = "abcdefghijklmnopqrstuvwxyz"
ALPH_UP = ALPH.upper()
DIG = "0123456789"

KO_JOSA = ["은","는","이","가","을","를","에","에서","에게","와","과","으로","로","도","만"]
EN_STOP = {"the","a","an","and","or","but","to","of","in","on","for","with"}

def rand_word(min_len=3, max_len=8, alph=ALPH):
    L = random.randint(min_len, max_len)
    return "".join(random.choice(alph) for _ in range(L))

def rand_token():
    r = random.random()
    if r < 0.5:
        return rand_word()
    elif r < 0.8:
        return "".join(random.choice(DIG) for _ in range(random.randint(1,5)))
    else:
        return random.choice(["-", "_", ":", "/", ".", ","])

def make_sequence(n_tokens=5):
    return " ".join(rand_token() for _ in range(n_tokens))

# ===== Base Generators =====
def sample_en(n_tokens=5):
    words = ["time","data","system","model","agent","rule","space","music",
             "river","node","graph","token","future","night","dream","light",
             "and","the","of","in","on","to","with"]
    return " ".join(random.choice(words) for _ in range(n_tokens))

def sample_ko(n_tokens=5):
    chunks = ["나는","오늘","데이터","모델","규칙","공간에서","노래를","듣고","있고",
              "그리고","또는","사랑","시간","세계","실험","결과를","본다"]
    return " ".join(random.choice(chunks) for _ in range(n_tokens))

def sample_ja(n_tokens=5):
    chunks = ["わたしは","きょう","データ","モデル","きそく","くうかんで","おんがくを","きいて",
              "いる","そして","または","あい","じかん","せかい","じっけん","けっかを","みる"]
    return " ".join(random.choice(chunks) for _ in range(n_tokens))

# ===== Structured Generators for “3.5-level” tasks =====
MONTHS_EN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def sample_dates_en(n=1):
    outs = []
    for _ in range(n):
        y = random.randint(2018, 2031); m = random.randint(1,12); d = random.randint(1,28)
        f = random.choice([
           f"{m}/{d}/{y}",                       # US
           f"{y}-{m:02d}-{d:02d}",              # ISO
           f"{MONTHS_EN[m-1]} {d}, {y}",        # Mon D, YYYY
        ])
        outs.append(f"meet at {f} with {rand_word()}")
    return " ".join(outs)

def sample_dates_ko(n=1):
    outs = []
    for _ in range(n):
        y = random.randint(2018, 2031); m = random.randint(1,12); d = random.randint(1,28)
        f = random.choice([
           f"{y}년 {m}월 {d}일",
           f"{y}.{m:02d}.{d:02d}",
           f"{m}월{d}일 {y}년",
        ])
        outs.append(f"약속 {f} 확인")
    return " ".join(outs)

def sample_dates_ja(n=1):
    outs = []
    for _ in range(n):
        y = random.randint(2018, 2031); m = random.randint(1,12); d = random.randint(1,28)
        f = random.choice([
           f"{y}年{m}月{d}日",
           f"{y}/{m:02d}/{d:02d}",
        ])
        outs.append(f"予定 {f} チェック")
    return " ".join(outs)

def sample_expr(n_terms=3):
    # e.g., "calc(12 + 3 * 4)"  or "12+5*2"
    ops = ["+","-","*"]
    nums = [str(random.randint(0, 99)) for _ in range(n_terms)]
    expr = nums[0]
    for i in range(1, n_terms):
        expr += random.choice(ops) + nums[i]
    if random.random() < 0.6:
        expr = f"calc({expr})"
    return expr

def sample_codeswitch_ko_en(n_tokens=6):
    # mix Korean and ASCII tokens; good for selective transforms
    ko = ["오늘","회의","시간","확인","요청","모델","규칙"]
    en = ["token","rule","graph","system","light","music","node"]
    seq = []
    for _ in range(n_tokens):
        seq.append(random.choice(ko if random.random()<0.5 else en))
    return " ".join(seq)

def sample_codeswitch_ja_en(n_tokens=6):
    ja = ["予定","時間","確認","規則","モデル","世界","結果"]
    en = ["token","rule","graph","system","light","music","node"]
    seq = []
    for _ in range(n_tokens):
        seq.append(random.choice(ja if random.random()<0.5 else en))
    return " ".join(seq)

# ===== Language-agnostic Rules (simple) =====
def rule_prefix_suffix(prefix="<<", suffix=">>"):
    def f(x: str): return f"{prefix}{x}{suffix}"
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
    def f(x: str):
        out = []
        for tok in x.split():
            if tok.isdigit():
                out.append(f"{l}{tok}{r}")
            else:
                out.append(tok)
        return " ".join(out)
    return f, f"surround numbers with {l} {r}"

def rule_toggle_case_ascii(step=2):
    def f(x: str):
        toks = x.split()
        for i, t in enumerate(toks):
            if t.isascii() and t.isalpha() and (i % step == 0):
                toks[i] = t.swapcase()
        return " ".join(toks)
    return f, f"toggle ASCII case every {step} tokens"

def rule_sort_numeric_tokens(ascending=True):
    def f(x: str):
        toks = x.split()
        idx_num = [(i,int(t)) for i,t in enumerate(toks) if t.isdigit()]
        sorted_vals = sorted([v for _,v in idx_num], reverse=not ascending)
        out = toks[:]
        # put back sorted numbers at numeric positions only
        j=0
        for i,_ in idx_num:
            out[i] = str(sorted_vals[j]); j+=1
        return " ".join(out)
    d = "sort numeric tokens asc" if ascending else "sort numeric tokens desc"
    return f, d

def rule_dedupe_preserve_order():
    def f(x: str):
        seen=set(); out=[]
        for t in x.split():
            if t not in seen:
                seen.add(t); out.append(t)
        return " ".join(out)
    return f, "dedupe tokens (stable)"

def rule_caesar(shift=1):
    def shift_char(c):
        if c in ALPH:
            i = (ALPH.index(c) + shift) % 26; return ALPH[i]
        if c in ALPH_UP:
            i = (ALPH_UP.index(c) + shift) % 26; return ALPH_UP[i]
        return c
    def f(x: str): return "".join(shift_char(c) for c in x)
    return f, f"caesar shift {shift}"

# ===== Korean Rules =====
def rule_ko_emphasize_josa(L="«", R="»"):
    def f(x: str):
        out = x
        for j in KO_JOSA:
            out = out.replace(" "+j+" ", f" {L}{j}{R} ")
        return out
    return f, "ko: emphasize josa"

def rule_ko_append_yo():
    def f(x: str):
        x = x.strip()
        if len(x)==0: return x
        if x.endswith(("요","요.","요!","요?")): return x
        return x + "요"
    return f, "ko: append polite '요'"

def rule_ko_kdigit_box(l="【", r="】"):
    def f(x: str):
        return "".join((l+c+r) if c.isdigit() else c for c in x)
    return f, "ko: box digits"

# Date → ISO (YYYY-MM-DD) for Korean strings
def rule_ko_date_to_iso():
    ko_patterns = [
        re.compile(r"(\d{4})[.]? ?(\d{1,2})[.]? ?(\d{1,2})[일]?", re.U),
        re.compile(r"(\d{4})[년]\s*(\d{1,2})[월]\s*(\d{1,2})[일]?", re.U),
        re.compile(r"(\d{1,2})[월]\s*(\d{1,2})[일]\s*(\d{4})[년]", re.U),
    ]
    def norm(y,m,d): return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    def repl(m):
        g=m.groups()
        if len(g)==3: 
            # handle 2 patterns
            if "년" in m.re.pattern:
                return norm(g[0],g[1],g[2])
            if "월]" in m.re.pattern or "월]" not in m.re.pattern:
                return norm(g[0],g[1],g[2])
        return m.group(0)
    def f(x:str):
        y=x
        for pat in ko_patterns:
            y = pat.sub(lambda m: norm(*m.groups()[0:3]) if len(m.groups())>=3 else m.group(0), y)
        return y
    return f, "ko: date → ISO"

# ===== English Rules =====
def rule_en_title_case_skip_stops():
    def title_word(w):
        if w.lower() in EN_STOP: return w.lower()
        return w[:1].upper() + w[1:]
    def f(x: str):
        toks = x.split()
        return " ".join(title_word(w) for w in toks)
    return f, "en: title case (skip stops)"

def rule_en_replace_and_amp():
    def f(x: str): return x.replace(" and ", " & ")
    return f, "en: replace 'and' with '&'"

def rule_en_date_to_iso():
    # support: M/D/YYYY , Mon D, YYYY
    mon_map = {m:i+1 for i,m in enumerate(MONTHS_EN)}
    re_us = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
    re_mon = re.compile(r"\b([A-Z][a-z]{2}) (\d{1,2}), (\d{4})\b")
    def f(x:str):
        x = re_us.sub(lambda m: f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}", x)
        x = re_mon.sub(lambda m: f"{int(m.group(3)):04d}-{mon_map[m.group(1)]:02d}-{int(m.group(2)):02d}", x)
        return x
    return f, "en: date → ISO"

# ===== Japanese Rules =====
def hira_to_kata(s: str):
    out=[]
    for ch in s:
        code=ord(ch)
        if 0x3041 <= code <= 0x3096: out.append(chr(code+0x60))
        else: out.append(ch)
    return "".join(out)

def kata_to_hira(s: str):
    out=[]
    for ch in s:
        code=ord(ch)
        if 0x30A1 <= code <= 0x30F6: out.append(chr(code-0x60))
        else: out.append(ch)
    return "".join(out)

def rule_ja_hira2kata(): return (lambda x: hira_to_kata(x), "ja: hiragana→katakana")
def rule_ja_kata2hira(): return (lambda x: kata_to_hira(x), "ja: katakana→hiragana")

def rule_ja_fullwidth_digits():
    def f(x:str):
        out=[]
        for ch in x:
            if ch.isdigit(): out.append(chr(ord(ch)+0xFF10-ord('0')))
            else: out.append(ch)
        return "".join(out)
    return f, "ja: fullwidth digits"

def rule_ja_date_to_iso():
    re_ja = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
    re_sl = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")
    def f(x:str):
        x = re_ja.sub(lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", x)
        x = re_sl.sub(lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", x)
        return x
    return f, "ja: date → ISO"

# ===== Math Rule (language-agnostic but “3.5-level”) =====
SAFE_EXPR = re.compile(r"^[0-9+\-*\s()]+$")

def rule_eval_simple_math():
    # evaluate expressions like "12+3*4" or "calc(12+3*4)"
    def eval_expr(s:str)->str:
        s = s.strip()
        if s.startswith("calc(") and s.endswith(")"):
            s = s[5:-1]
        s = s.replace(" ", "")
        if not SAFE_EXPR.match(s): return s
        # very restricted eval
        return str(eval(s, {"__builtins__":None}, {}))
    def f(x:str): return eval_expr(x)
    return f, "math: eval simple expression"

# ===== Rule Composition =====
def compose(fs: List[Callable[[str], str]]):
    def f(x:str):
        y=x
        for g in fs:
            y = g(y)
        return y
    return f

# For factories that may override generator
FactoryRet = Tuple[Callable[[str],str], str, Optional[Callable[..., str]]]

def wrap2(f_desc):
    f, desc = f_desc
    return (f, desc, None)

# ===== Rule Factories =====
RULE_FACTORIES_BASE = [
    lambda: wrap2(rule_prefix_suffix(prefix=random.choice(["<<","[[","("]), 
                                     suffix=random.choice([">>","]]",")"]))),
    lambda: wrap2(rule_uppercase_every_k(k=random.randint(2,4))),
    lambda: wrap2(rule_reverse_tokens()),
    lambda: wrap2(rule_surround_numbers(l=random.choice(["{","<","["]), r=random.choice(["}",">","]"]))),
    lambda: wrap2(rule_toggle_case_ascii(step=random.randint(2,3))),
    lambda: wrap2(rule_sort_numeric_tokens(ascending=random.random()<0.5)),
    lambda: wrap2(rule_dedupe_preserve_order()),
    lambda: wrap2(rule_caesar(shift=random.randint(1,3))),
]

RULE_FACTORIES_KO = [
    lambda: wrap2(rule_ko_emphasize_josa()),
    lambda: wrap2(rule_ko_append_yo()),
    lambda: wrap2(rule_ko_kdigit_box()),
    # Advanced KO (generator override)
    lambda: (rule_ko_date_to_iso()[0], "ko: date → ISO",
             lambda n_tokens=6: sample_dates_ko(n=2)),
]

RULE_FACTORIES_EN = [
    lambda: wrap2(rule_en_title_case_skip_stops()),
    lambda: wrap2(rule_en_replace_and_amp()),
    lambda: (rule_en_date_to_iso()[0], "en: date → ISO",
             lambda n_tokens=6: sample_dates_en(n=2)),
]

RULE_FACTORIES_JA = [
    lambda: wrap2(rule_ja_hira2kata()),
    lambda: wrap2(rule_ja_kata2hira()),
    lambda: wrap2(rule_ja_fullwidth_digits()),
    lambda: (rule_ja_date_to_iso()[0], "ja: date → ISO",
             lambda n_tokens=6: sample_dates_ja(n=2)),
]

# Optional hard task: expression evaluation (used across langs/base)
def FACTORY_MATH():
    return (rule_eval_simple_math()[0], "math: eval", lambda n_tokens=6: sample_expr(n_terms=random.randint(2,4)))

# ===== Sample Task =====
def sample_task(n_context=6, n_query=8, n_tokens=5) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]], str]:
    """
    Few-shot function induction:
      - pick lang bucket
      - pick either: (a) advanced single rule, or (b) 2~3 rule composition from safe set
      - optionally use generator override for structured inputs (dates/math)
    """
    lang = random.choice(["en","ko","ja","base"])
    if lang == "en":
        pool = RULE_FACTORIES_EN + RULE_FACTORIES_BASE[:2] + [FACTORY_MATH]
        gen_default = sample_en
    elif lang == "ko":
        pool = RULE_FACTORIES_KO + RULE_FACTORIES_BASE[:2] + [FACTORY_MATH]
        gen_default = sample_ko
    elif lang == "ja":
        pool = RULE_FACTORIES_JA + RULE_FACTORIES_BASE[:1] + [FACTORY_MATH]
        gen_default = sample_ja
    else:
        pool = RULE_FACTORIES_BASE + [FACTORY_MATH]
        gen_default = make_sequence

    # Choose advanced vs composition
    advanced_pick = random.random() < 0.55  # bias to harder cases
    gen = gen_default
    desc = ""

    if advanced_pick:
        f1, d1, g1 = random.choice(pool)()
        # allow mild composition with one safe post-step
        if random.random() < 0.35:
            f2, d2, _ = random.choice(RULE_FACTORIES_BASE)()
            f = compose([f1, f2])
            desc = f"{d1} ∘ {d2}"
        else:
            f = f1; desc = d1
        if g1 is not None:
            gen = g1
    else:
        # 2~3 rule composition from safe set
        k = random.choice([2,3])
        picks = [random.choice(RULE_FACTORIES_BASE)() for _ in range(k)]
        fs = [p[0] for p in picks]
        f = compose(fs)
        desc = " ∘ ".join(p[1] for p in picks)

    # Build contexts/queries
    C, Q = [], []
    for _ in range(n_context):
        s = gen(n_tokens=n_tokens)
        C.append((s, f(s)))
    for _ in range(n_query):
        s = gen(n_tokens=n_tokens)
        Q.append((s, f(s)))

    return C, Q, f"{lang}: {desc}"
