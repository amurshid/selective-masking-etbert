#!/usr/bin/env python3
"""
eval_preds.py  --  dump per-sample predictions for every (model, mode, level) cell.

eval_degradation.py reports only aggregates, which is enough for point estimates
but not for confidence intervals. This script re-runs the same evaluation and
saves the raw (gold, pred) vector for each cell so bootstrap_ci.py can resample
them. Predictions are produced with the identical Classifier / read_dataset used
in training, so the point estimates reproduce degradation_results.csv exactly.

Writes results/preds.npz  (one int16 array of shape [n,2] per cell key
"model|mode|level"), plus results/preds_index.csv listing the cells.
"""
import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn as nn

REPO = "/Users/ahnaf/Projects/ET-BERT"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "fine-tuning"))

from run_classifier import Classifier, read_dataset, batch_loader, count_labels_num
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts

MODELS = [("baseline",       "models/finetuned_ustc_full.bin"),
          ("ours_p50",       "models/finetuned_ustc_masked.bin"),
          ("ours_p70",       "models/finetuned_ustc_masked_p70.bin"),
          ("normalize_p70",  "models/finetuned_ustc_normalize_p70.bin"),
          ("normalize_p100", "models/finetuned_ustc_normalize_p100.bin"),
          ("masked_p100",    "models/finetuned_ustc_masked_p100.bin"),
          ("normalize_h8",   "models/finetuned_ustc_normalize_p70_h8.bin"),
          ("normalize_h24",  "models/finetuned_ustc_normalize_p70_h24.bin"),
          ("normalize_h32",  "models/finetuned_ustc_normalize_p70_h32.bin"),
          ("baseline_s17",   "models/finetuned_ustc_full_s17.bin"),
          ("baseline_s27",   "models/finetuned_ustc_full_s27.bin"),
          ("normalize_s17",  "models/finetuned_ustc_normalize_p70_s17.bin"),
          ("normalize_s27",  "models/finetuned_ustc_normalize_p70_s27.bin")]
MODES  = ["mask", "random", "zero"]
LEVELS = [0, 25, 50, 75, 100]


def predict(args, dataset):
    """Same forward pass as run_classifier.evaluate, but keeps per-sample output."""
    src = torch.LongTensor([s[0] for s in dataset])
    tgt = torch.LongTensor([s[1] for s in dataset])
    seg = torch.LongTensor([s[2] for s in dataset])
    golds, preds = [], []
    args.model.eval()
    for src_b, tgt_b, seg_b, _ in batch_loader(args.batch_size, src, tgt, seg):
        src_b, tgt_b, seg_b = src_b.to(args.device), tgt_b.to(args.device), seg_b.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_b, tgt_b, seg_b)
        preds.append(torch.argmax(nn.Softmax(dim=1)(logits), dim=1).cpu().numpy())
        golds.append(tgt_b.cpu().numpy())
    return np.stack([np.concatenate(golds), np.concatenate(preds)], axis=1).astype(np.int16)


def macro_f1_from_pairs(gp, n_cls):
    conf = np.bincount(gp[:, 0] * n_cls + gp[:, 1], minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    tp = np.diag(conf).astype(float)
    p = tp / (conf.sum(axis=0) + 1e-9)      # column = predicted
    r = tp / (conf.sum(axis=1) + 1e-9)      # row    = gold
    return float(np.mean(np.where(p + r == 0, 0.0, 2 * p * r / (p + r + 1e-9))))


def build_args():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    parser.add_argument("--pooling", default="first")
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    argv = ["--vocab_path", "models/encryptd_vocab.txt",
            "--train_path", "datasets/ustc-full-tsv/test_dataset.tsv",
            "--dev_path",   "datasets/ustc-full-tsv/test_dataset.tsv",
            "--config_path", "models/bert/base_config.json",
            "--embedding", "word_pos_seg", "--encoder", "transformer",
            "--mask", "fully_visible", "--seq_length", "128", "--batch_size", "32"]
    args = load_hyperparam(parser.parse_args(argv))
    set_seed(args.seed)
    args.tokenizer    = str2tokenizer[args.tokenizer](args)
    args.labels_num   = count_labels_num(args.train_path)
    args.soft_targets = False
    args.device = (torch.device("mps") if torch.backends.mps.is_available()
                   else torch.device("cuda:0") if torch.cuda.is_available()
                   else torch.device("cpu"))
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated subset of model names (default: all)")
    ap.add_argument("--out", default="results/preds.npz")
    cli, _ = ap.parse_known_args()
    os.chdir(REPO)
    args = build_args()
    print("labels_num=%d  device=%s\n" % (args.labels_num, args.device))
    model = Classifier(args)

    out, index = {}, []
    want = set(cli.models.split(",")) if cli.models else None
    for mname, mpath in MODELS:
        if want and mname not in want:
            continue
        if not os.path.exists(mpath):
            print("SKIP %-16s (not trained yet: %s)" % (mname, mpath)); continue
        model.load_state_dict(torch.load(mpath, map_location="cpu"), strict=False)
        model.to(args.device); args.model = model
        for mode in MODES:
            for lvl in LEVELS:
                gp = predict(args, read_dataset(args, "datasets/ustc-degraded/%s/test_deg%03d.tsv" % (mode, lvl)))
                key = "%s|%s|%d" % (mname, mode, lvl)
                out[key] = gp
                f1  = macro_f1_from_pairs(gp, args.labels_num)
                acc = float((gp[:, 0] == gp[:, 1]).mean())
                index.append((mname, mode, lvl, len(gp), "%.4f" % acc, "%.4f" % f1))
                print("  %-15s %-6s deg=%3d%%  n=%5d  acc=%.4f  macroF1=%.4f" % (mname, mode, lvl, len(gp), acc, f1))

    os.makedirs("results", exist_ok=True)
    np.savez_compressed(cli.out, **out)
    with open("results/preds_index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "mode", "degradation_pct", "n", "accuracy", "macro_f1"]); w.writerows(index)
    print("\nsaved -> %s (%d cells) and results/preds_index.csv" % (cli.out, len(out)))


if __name__ == "__main__":
    main()
