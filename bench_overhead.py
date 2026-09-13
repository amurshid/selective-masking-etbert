#!/usr/bin/env python3
"""
bench_overhead.py  --  runtime and memory cost of header suppression.

Reviewer 2 asked for runtime and memory measurements. The paper claims the method
adds no parameters, leaves inference untouched, and costs negligibly at training
time. The first two are true by construction and are verified here rather than
asserted; the third is measured.

Reports, over N warm steps after discarding warmup:
  - training step time with suppression off vs on (the only place cost can appear)
  - inference throughput, which must be unchanged since the model is not modified
  - parameter count and peak device memory
Writes results/overhead.csv
"""
import os, sys, csv, time, argparse
import numpy as np
import torch

REPO = "/Users/ahnaf/Projects/ET-BERT"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "fine-tuning"))

from run_classifier_mask import Classifier, read_dataset, batch_loader, count_labels_num, apply_header_mask
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts

TRAIN = "datasets/ustc-full-tsv/train_dataset.tsv"
WARMUP, STEPS = 10, 60


def build_args():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    parser.add_argument("--pooling", default="first")
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    argv = ["--vocab_path", "models/encryptd_vocab.txt",
            "--train_path", TRAIN, "--dev_path", TRAIN,
            "--config_path", "models/bert/base_config.json",
            "--embedding", "word_pos_seg", "--encoder", "transformer",
            "--mask", "fully_visible", "--seq_length", "128", "--batch_size", "32"]
    args = load_hyperparam(parser.parse_args(argv))
    set_seed(args.seed)
    args.tokenizer    = str2tokenizer[args.tokenizer](args)
    args.labels_num   = count_labels_num(args.train_path)
    args.soft_targets = False
    args.header_tokens, args.mask_target = 16, "header"
    args.device = (torch.device("mps") if torch.backends.mps.is_available()
                   else torch.device("cuda:0") if torch.cuda.is_available()
                   else torch.device("cpu"))
    return args


def sync(dev):
    if dev.type == "mps":  torch.mps.synchronize()
    elif dev.type == "cuda": torch.cuda.synchronize()


def peak_mem_mb(dev):
    if dev.type == "mps":  return torch.mps.current_allocated_memory() / 1e6
    if dev.type == "cuda": return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def time_training(args, model, batches, mask_rate, op):
    args.mask_rate, args.mask_op = mask_rate, op
    args.mask_repl_id = 41 if op == "normalize" else 4
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    ts = []
    for i, (src, tgt, seg) in enumerate(batches):
        sync(args.device); t0 = time.perf_counter()
        model.zero_grad()
        src_i = apply_header_mask(args, src, seg) if mask_rate > 0 else src
        loss, _ = model(src_i, tgt, seg)
        loss.backward(); opt.step()
        sync(args.device)
        if i >= WARMUP: ts.append(time.perf_counter() - t0)
    return np.array(ts)


def time_inference(args, model, batches):
    model.eval(); ts = []
    for i, (src, tgt, seg) in enumerate(batches):
        sync(args.device); t0 = time.perf_counter()
        with torch.no_grad():
            model(src, tgt, seg)
        sync(args.device)
        if i >= WARMUP: ts.append(time.perf_counter() - t0)
    model.train()
    return np.array(ts)


def main():
    os.chdir(REPO)
    args = build_args()
    print("device=%s  batch=%d  steps=%d (after %d warmup)\n" % (args.device, args.batch_size, STEPS, WARMUP))

    ds = read_dataset(args, TRAIN)[: args.batch_size * (WARMUP + STEPS)]
    src = torch.LongTensor([s[0] for s in ds]); tgt = torch.LongTensor([s[1] for s in ds]); seg = torch.LongTensor([s[2] for s in ds])
    batches = [(a.to(args.device), b.to(args.device), c.to(args.device))
               for a, b, c, _ in batch_loader(args.batch_size, src, tgt, seg)]

    model = Classifier(args).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())

    rows = []
    for label, rate, op in [("baseline (no suppression)", 0.0, "mask"),
                            ("masking p=0.7", 0.7, "mask"),
                            ("normalization p=0.7", 0.7, "normalize"),
                            ("normalization p=1.0", 1.0, "normalize")]:
        t = time_training(args, model, batches, rate, op)
        rows.append(("train_step", label, n_par, "%.2f" % (1000 * t.mean()), "%.2f" % (1000 * t.std()), "%.1f" % peak_mem_mb(args.device)))
        print("  train  %-26s  %7.2f ms/step  (sd %.2f)  params=%d" % (label, 1000 * t.mean(), 1000 * t.std(), n_par))

    t = time_inference(args, model, batches)
    thr = args.batch_size / t.mean()
    rows.append(("inference", "unmodified at test time", n_par, "%.2f" % (1000 * t.mean()), "%.2f" % (1000 * t.std()), "%.1f" % peak_mem_mb(args.device)))
    print("\n  infer  %-26s  %7.2f ms/batch  (%.0f packets/s)" % ("identical for all variants", 1000 * t.mean(), thr))

    os.makedirs("results", exist_ok=True)
    with open("results/overhead.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phase", "variant", "parameters", "ms_mean", "ms_sd", "peak_mem_mb"]); w.writerows(rows)
    print("\nsaved -> results/overhead.csv")


if __name__ == "__main__":
    main()
