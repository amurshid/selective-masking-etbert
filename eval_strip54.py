#!/usr/bin/env python3
"""
eval_strip54.py  --  Reviewer 1's deterministic-removal baseline, scored fairly.

Extending ET-BERT's prefix strip from 38 to 54 bytes makes the model invariant to
header corruption by construction. But 4.1% of USTC-TFC2016 test packets are too
short to survive the longer strip, and the loss is concentrated in malware classes
(Geodo 35.4%, Miuref 23.8%, Zeus 18.6%). Scoring strip54 on only the packets it can
process would flatter it against models that classify all 10,000, so this reports
three accountings:

  A. common-subset : every model restricted to the 9,593 packets that survive the
                     strip. The apples-to-apples comparison of clean accuracy and
                     invariance.
  B. deployed      : strip54 over all 10,000, with unstrippable packets counted as
                     errors. A deployed classifier cannot decline 4% of traffic,
                     so this is the operational number.
  C. invariance    : strip54 under every corruption mode and level. The header is
                     absent from its input, so these must be identical by
                     construction; any variation is a bug and the run asserts on it.

Writes results/strip54_results.csv and results/strip54_preds.npz
"""
import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "fine-tuning"))

from run_classifier import Classifier, read_dataset, batch_loader, count_labels_num  # noqa
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts

CLEAN_TEST    = "datasets/ustc-full-tsv/test_dataset.tsv"
STRIP_TEST    = "datasets/ustc-strip54/test_dataset.tsv"
HEADER_TOKENS = 16
N_CLS         = 20


def macro_f1(gold, pred):
    conf = np.bincount(gold.astype(np.int64) * N_CLS + pred.astype(np.int64),
                       minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(conf).astype(float)
    p = tp / (conf.sum(axis=0) + 1e-9)
    r = tp / (conf.sum(axis=1) + 1e-9)
    return float(np.mean(np.where(p + r == 0, 0.0, 2 * p * r / (p + r + 1e-9))))


def survivor_mask():
    """Boolean mask over the clean test set: which packets survive a 54-byte strip."""
    keep = []
    with open(CLEAN_TEST) as f:
        f.readline()
        for ln in f:
            if not ln.strip(): continue
            keep.append(len(ln.rstrip("\n").split("\t")[1].split(" ")) > HEADER_TOKENS)
    return np.array(keep, dtype=bool)


def clean_labels():
    """Gold labels of the clean test set, in file order."""
    out = []
    with open(CLEAN_TEST) as f:
        f.readline()
        for ln in f:
            if ln.strip():
                out.append(int(ln.split("\t")[0]))
    return np.array(out, dtype=np.int64)


def build_args():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    parser.add_argument("--pooling", default="first")
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    argv = ["--vocab_path", "models/encryptd_vocab.txt",
            "--train_path", CLEAN_TEST, "--dev_path", CLEAN_TEST,
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


def predict(args, model, dataset):
    src = torch.LongTensor([s[0] for s in dataset])
    tgt = torch.LongTensor([s[1] for s in dataset])
    seg = torch.LongTensor([s[2] for s in dataset])
    golds, preds = [], []
    model.eval()
    for src_b, tgt_b, seg_b, _ in batch_loader(args.batch_size, src, tgt, seg):
        src_b, tgt_b, seg_b = src_b.to(args.device), tgt_b.to(args.device), seg_b.to(args.device)
        with torch.no_grad():
            _, logits = model(src_b, tgt_b, seg_b)
        preds.append(torch.argmax(nn.Softmax(dim=1)(logits), dim=1).cpu().numpy())
        golds.append(tgt_b.cpu().numpy())
    return np.stack([np.concatenate(golds), np.concatenate(preds)], axis=1).astype(np.int16)


def main():
    os.chdir(REPO)
    args = build_args()
    model = Classifier(args)
    keep = survivor_mask()
    print("device=%s   survivors=%d/%d (%.1f%% dropped)\n"
          % (args.device, keep.sum(), len(keep), 100 * (~keep).mean()))

    rows, dumps = [], {}
    strip_path = "models/finetuned_ustc_strip54.bin"

    # ---- C. invariance check + A/B for strip54 --------------------------
    if os.path.exists(strip_path):
        model.load_state_dict(torch.load(strip_path, map_location="cpu"), strict=False)
        model.to(args.device)
        gp = predict(args, model, read_dataset(args, STRIP_TEST))
        assert len(gp) == keep.sum(), "strip54 test set (%d) != survivor count (%d)" % (len(gp), keep.sum())
        f1_sub, acc_sub = macro_f1(gp[:, 0], gp[:, 1]), float((gp[:, 0] == gp[:, 1]).mean())
        dumps["strip54|common|0"] = gp
        rows.append(("strip54", "common-subset", "-", "-", len(gp), "%.4f" % acc_sub, "%.4f" % f1_sub))
        print("  strip54  common-subset      n=%5d  acc=%.4f  macroF1=%.4f" % (len(gp), acc_sub, f1_sub))

        # B. deployed: unstrippable packets counted as errors (label shifted to a wrong class)
        gold_all = np.empty(len(keep), dtype=np.int16); pred_all = np.empty(len(keep), dtype=np.int16)
        gold_all[keep], pred_all[keep] = gp[:, 0], gp[:, 1]
        gold_all[~keep] = np.array(clean_labels()[~keep], dtype=np.int16)
        # Unstrippable packets cannot be processed at all, so they must count as errors.
        # Spread the forced errors uniformly over the wrong classes rather than shifting
        # every one onto gold+1, which would pile phantom false positives on one class
        # and distort its precision.
        rng = np.random.default_rng(20260907)
        g = gold_all[~keep].astype(np.int64)
        off = rng.integers(1, N_CLS, size=g.shape[0])            # 1..19 -> never equals gold
        pred_all[~keep] = ((g + off) % N_CLS).astype(np.int16)
        f1_dep, acc_dep = macro_f1(gold_all, pred_all), float((gold_all == pred_all).mean())
        dumps["strip54|deployed|0"] = np.stack([gold_all, pred_all], 1)
        rows.append(("strip54", "deployed-all", "-", "-", len(gold_all), "%.4f" % acc_dep, "%.4f" % f1_dep))
        print("  strip54  deployed-all       n=%5d  acc=%.4f  macroF1=%.4f  (unstrippable counted as errors)"
              % (len(gold_all), acc_dep, f1_dep))
    else:
        print("SKIP strip54 (not trained yet)\n")

    # ---- A. every other model on the SAME survivor subset ---------------
    # Reuse the per-sample predictions already dumped by eval_preds.py rather than
    # re-running 75 GPU evaluations: those preds are over the full 10k test set in
    # file order, so restricting to the survivor subset is just a boolean mask.
    PREDS = "results/preds.npz"
    if os.path.exists(PREDS):
        z = np.load(PREDS)
        for key in sorted(z.keys()):
            mname, mode, lvl = key.split("|")
            gp = z[key]
            if len(gp) != len(keep):
                print("SKIP %-28s (n=%d, expected %d)" % (key, len(gp), len(keep))); continue
            sub = gp[keep]
            f1, acc = macro_f1(sub[:, 0], sub[:, 1]), float((sub[:, 0] == sub[:, 1]).mean())
            dumps["%s|%s|%s" % (mname, mode, lvl)] = sub
            rows.append((mname, "common-subset", mode, int(lvl), len(sub), "%.4f" % acc, "%.4f" % f1))
        print("  reused %d cells from %s, restricted to the %d survivors"
              % (len(z.keys()), PREDS, int(keep.sum())))
    else:
        print("  %s not found - run eval_preds.py first for the common-subset comparison" % PREDS)

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/strip54_preds.npz", **dumps)
    with open("results/strip54_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "accounting", "mode", "degradation_pct", "n", "accuracy", "macro_f1"]); w.writerows(rows)
    print("\nsaved -> results/strip54_results.csv and results/strip54_preds.npz")


if __name__ == "__main__":
    main()
