#!/usr/bin/env python3
"""
make_degraded_tests.py  --  masked, normalized, randomized

Builds degraded copies of the clean test set. The first HEADER_TOKENS tokens of
text_a are the residual transport header (same region masked in training); payload
is never touched. Three modes:
  mask   -> replace selected header tokens with [MASK]            (header ABSENT)
  random -> replace them with random valid byte-bigrams from vocab (header WRONG)
  zero   -> replace them with the canonical "0000" token          (header NORMALIZED/zeroed)
Fixed seeds make every file reproducible, and the masked POSITIONS are identical
across the two modes (only the replacement value differs) for a clean comparison.
"""
import os, random

CLEAN_TEST    = "datasets/ustc-full-tsv/test_dataset.tsv"
VOCAB_PATH    = "models/encryptd_vocab.txt"
OUT_ROOT      = "datasets/ustc-degraded"
HEADER_TOKENS = 16
LEVELS        = [0, 25, 50, 75, 100]
MODES         = ["mask", "random", "zero"]

# valid byte-bigram tokens from the vocab (4 lowercase-hex chars) -> used for random mode
HEXSET  = set("0123456789abcdef")
bigrams = []
with open(VOCAB_PATH) as f:
    for line in f:
        t = line.strip()
        if len(t) == 4 and all(c in HEXSET for c in t):
            bigrams.append(t)

with open(CLEAN_TEST) as f:
    head = f.readline().rstrip("\n")
    rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]

for mode in MODES:
    out_dir = os.path.join(OUT_ROOT, mode)
    os.makedirs(out_dir, exist_ok=True)
    for level in LEVELS:
        pos_rng = random.Random(1000 + level)     # which positions -> SAME across modes
        val_rng = random.Random(5000 + level)     # replacement values -> random mode only
        n_mask  = round(HEADER_TOKENS * level / 100)
        out     = os.path.join(out_dir, "test_deg%03d.tsv" % level)
        with open(out, "w") as w:
            w.write(head + "\n")
            for label, text_a in rows:
                toks = text_a.split(" ")
                span = min(HEADER_TOKENS, len(toks))
                k    = min(n_mask, span)
                if k > 0:
                    for i in pos_rng.sample(range(span), k):
                        if mode == "mask":
                            toks[i] = "[MASK]"
                        elif mode == "zero":
                            toks[i] = "0000"
                        else:  # random
                            toks[i] = val_rng.choice(bigrams)
                w.write("%s\t%s\n" % (label, " ".join(toks)))
        print("wrote %-42s  (%d/%d header tokens corrupted)" % (out, n_mask, HEADER_TOKENS))
