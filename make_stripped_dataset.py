#!/usr/bin/env python3
"""
make_stripped_dataset.py  --  Reviewer-1 baseline: deterministic header removal.

ET-BERT's preprocessing drops a 38-byte prefix (Eth + IPv4 + ports), leaving ~16
residual TCP-header bigrams at the head of every sequence. This script extends
that strip to 54 bytes by deleting those 16 leading tokens from train/valid/test,
so the header region is absent by construction at BOTH training and inference.
Payload is untouched; because the clean sequences run to ~129 tokens against a
128-token window, stripping pulls 16 further payload bigrams into view.

Reads  datasets/ustc-full-tsv/{train,valid,test}_dataset.tsv
Writes datasets/ustc-strip54/{train,valid,test}_dataset.tsv
"""
import os

SRC_DIR       = "datasets/ustc-full-tsv"
OUT_DIR       = "datasets/ustc-strip54"
HEADER_TOKENS = 16
SPLITS        = ["train_dataset.tsv", "valid_dataset.tsv", "test_dataset.tsv"]

os.makedirs(OUT_DIR, exist_ok=True)
for split in SPLITS:
    src, dst = os.path.join(SRC_DIR, split), os.path.join(OUT_DIR, split)
    n, dropped = 0, 0
    with open(src) as f, open(dst, "w") as w:
        w.write(f.readline())                       # header line: label\ttext_a
        for ln in f:
            if not ln.strip():
                continue
            label, text_a = ln.rstrip("\n").split("\t")
            toks = text_a.split(" ")
            if len(toks) <= HEADER_TOKENS:          # nothing left after the strip
                dropped += 1
                continue
            w.write("%s\t%s\n" % (label, " ".join(toks[HEADER_TOKENS:])))
            n += 1
    print("wrote %-44s  %6d rows  (%d too short, skipped)" % (dst, n, dropped))
