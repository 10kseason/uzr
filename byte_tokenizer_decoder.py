#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ByteTokenizer decoder (vocab 258: 0..255 bytes, 256=<BOS>, 257=<EOS>).
Usage:
    # decode a space/comma separated line of ints
    echo "256 72 101 108 108 111 44 32 234 149 176 235 139 164 33 257" | python byte_tokenizer_decoder.py

    # decode from a file with one sequence per line
    python byte_tokenizer_decoder.py --in ids.txt --out decoded.txt

    # strict UTF-8 (error if invalid)
    python byte_tokenizer_decoder.py --strict
"""
import argparse, sys, ast

BOS_ID = 256
EOS_ID = 257

def decode_ids(seq, strict=False):
    b = bytearray()
    for t in seq:
        t = int(t)
        if t == BOS_ID:
            continue
        if t == EOS_ID:
            break
        if 0 <= t <= 255:
            b.append(t)
    if strict:
        return bytes(b).decode("utf-8", errors="strict")
    try:
        return bytes(b).decode("utf-8", errors="ignore")
    except Exception:
        return bytes(b).decode("latin-1", errors="ignore")

def parse_seq(s):
    s = s.strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [int(x) for x in obj]
        except Exception:
            pass
    # fallback: split on spaces/commas
    parts = s.replace(",", " ").split()
    return [int(p) for p in parts]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", type=str, default=None, help="input file (one token sequence per line)")
    ap.add_argument("--out", dest="outfile", type=str, default=None, help="output file for decoded text")
    ap.add_argument("--strict", action="store_true", help="strict UTF-8 (raise on invalid)")
    args = ap.parse_args()

    fin = open(args.infile, "r", encoding="utf-8") if args.infile else sys.stdin
    fout = open(args.outfile, "w", encoding="utf-8") if args.outfile else sys.stdout

    try:
        for line in fin:
            seq = parse_seq(line)
            text = decode_ids(seq, strict=args.strict)
            fout.write(text + "\n")
    finally:
        if args.infile:
            fin.close()
        if args.outfile:
            fout.close()

if __name__ == "__main__":
    main()
