#!/usr/bin/env python3
"""
adversarial_header.py  --  worst-case header manipulation.

Both reviewers noted that omission/randomization/zeroing model NON-adversarial
corruption, and that a deliberate adversary is untested. This adds two threat
models, each confined to the 16 residual transport-header tokens; the payload is
never touched, so any drop is attributable to header reliance alone.

  transplant : replace the header with the header of a real packet drawn from a
               DIFFERENT class. Requires no model access (black-box) and is
               operationally trivial - an attacker rewrites their own header
               fields to imitate benign traffic.
  hotflip    : gradient-guided substitution (Ebrahimi et al.). For each header
               position, score every candidate byte-bigram by its first-order
               effect on the loss, (E[v] - E[orig]) . dL/d(word_emb), and take
               the maximiser. White-box, and a strict upper bound on transplant.

Candidates are restricted to valid 4-hex-char bigrams, so every attack produces
a byte sequence the attacker could actually put on the wire.

Writes results/adversarial_results.csv and results/adversarial_preds.npz
(the latter feeds bootstrap_ci.py for intervals on the attacked numbers).
"""
import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.abspath(__file__))
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
          ("masked_p100",    "models/finetuned_ustc_masked_p100.bin")]

CLEAN_TEST    = "datasets/ustc-full-tsv/test_dataset.tsv"
HEADER_TOKENS = 16
N_CAND        = 1024        # sampled candidate bigrams for hotflip scoring
HOTFLIP_ITERS = 2           # re-scoring passes
SEED          = 20260907


def macro_f1(gold, pred, n_cls):
    conf = np.bincount(gold.astype(np.int64) * n_cls + pred.astype(np.int64),
                       minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    tp = np.diag(conf).astype(float)
    p = tp / (conf.sum(axis=0) + 1e-9)
    r = tp / (conf.sum(axis=1) + 1e-9)
    return float(np.mean(np.where(p + r == 0, 0.0, 2 * p * r / (p + r + 1e-9))))


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


def hex_bigram_ids(vocab):
    hexset = set("0123456789abcdef")
    return sorted(i for t, i in vocab.items() if len(t) == 4 and all(c in hexset for c in t))


def transplant_headers(src, tgt, rng):
    """Swap in the header of a real packet whose label differs."""
    out = src.clone()
    n = src.size(0)
    tgt_np = tgt.numpy()
    for i in range(n):
        for _ in range(20):                       # rejection-sample a different class
            j = rng.integers(0, n)
            if tgt_np[j] != tgt_np[i]:
                out[i, 1:1 + HEADER_TOKENS] = src[j, 1:1 + HEADER_TOKENS]
                break
    return out


def hotflip_batch(args, model, src_b, tgt_b, seg_b, cand_ids, emb_table):
    """One-shot gradient-guided substitution over the header region, iterated."""
    src_adv = src_b.clone()
    end = min(1 + HEADER_TOKENS, src_adv.size(1))
    cand_emb = emb_table[cand_ids]                                  # [C, H]
    for _ in range(HOTFLIP_ITERS):
        captured = {}
        def hook(_m, _in, out):
            out.retain_grad(); captured["e"] = out
        h = model.embedding.word_embedding.register_forward_hook(hook)
        model.zero_grad(set_to_none=True)
        loss, _ = model(src_adv, tgt_b, seg_b)
        loss.backward()
        h.remove()
        g = captured["e"].grad[:, 1:end, :]                         # [B, R, H]
        orig = emb_table[src_adv[:, 1:end]]                         # [B, R, H]
        # first-order change in loss for each candidate substitution
        delta = torch.einsum("brh,ch->brc", g, cand_emb) - (g * orig).sum(-1, keepdim=True)
        best = cand_ids[delta.argmax(dim=-1)]                       # [B, R]
        valid = seg_b[:, 1:end] != 0                                # never touch padding
        src_adv[:, 1:end] = torch.where(valid, best, src_adv[:, 1:end])
    return src_adv.detach()


def run_attack(args, model, dataset, attack, cand_ids, emb_table, rng):
    src = torch.LongTensor([s[0] for s in dataset])
    tgt = torch.LongTensor([s[1] for s in dataset])
    seg = torch.LongTensor([s[2] for s in dataset])
    if attack == "transplant":
        src = transplant_headers(src, tgt, rng)
    golds, preds = [], []
    for src_b, tgt_b, seg_b, _ in batch_loader(args.batch_size, src, tgt, seg):
        src_b, tgt_b, seg_b = src_b.to(args.device), tgt_b.to(args.device), seg_b.to(args.device)
        if attack == "hotflip":
            model.train()                     # grads needed; dropout disabled below
            for m in model.modules():
                if isinstance(m, nn.Dropout): m.eval()
            src_b = hotflip_batch(args, model, src_b, tgt_b, seg_b, cand_ids, emb_table)
        model.eval()
        with torch.no_grad():
            _, logits = model(src_b, tgt_b, seg_b)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        golds.append(tgt_b.cpu().numpy())
    return np.stack([np.concatenate(golds), np.concatenate(preds)], axis=1).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", default="clean,transplant,hotflip")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test packets (0 = all)")
    ap.add_argument("--device", default="", help="force a device (cpu) instead of auto-selecting MPS/CUDA")
    cli = ap.parse_args()

    os.chdir(REPO)
    args = build_args()
    if cli.device:
        args.device = torch.device(cli.device)
    print("labels_num=%d  device=%s" % (args.labels_num, args.device))
    model = Classifier(args)

    dataset = read_dataset(args, CLEAN_TEST)
    if cli.limit:
        dataset = dataset[:cli.limit]
    print("test packets: %d\n" % len(dataset))

    all_bigrams = hex_bigram_ids(args.tokenizer.vocab)
    sub_rng = np.random.default_rng(SEED)
    cand = sorted(sub_rng.choice(all_bigrams, size=min(N_CAND, len(all_bigrams)), replace=False).tolist())
    cand_ids = torch.LongTensor(cand).to(args.device)
    print("candidate bigrams: %d of %d valid\n" % (len(cand), len(all_bigrams)))

    rows, dumps = [], {}
    for mname, mpath in MODELS:
        if not os.path.exists(mpath):
            print("SKIP %-16s (not trained yet)" % mname); continue
        model.load_state_dict(torch.load(mpath, map_location="cpu"), strict=False)
        model.to(args.device)
        emb_table = model.embedding.word_embedding.weight.detach()
        for attack in cli.attacks.split(","):
            rng = np.random.default_rng(SEED)
            gp = run_attack(args, model, dataset, attack, cand_ids, emb_table, rng)
            acc = float((gp[:, 0] == gp[:, 1]).mean())
            f1  = macro_f1(gp[:, 0], gp[:, 1], args.labels_num)
            rows.append((mname, attack, len(gp), "%.4f" % acc, "%.4f" % f1))
            dumps["%s|%s|0" % (mname, attack)] = gp
            print("  %-15s %-11s  acc=%.4f  macroF1=%.4f" % (mname, attack, acc, f1))

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/adversarial_preds.npz", **dumps)
    with open("results/adversarial_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "attack", "n", "accuracy", "macro_f1"]); w.writerows(rows)
    print("\nsaved -> results/adversarial_results.csv and results/adversarial_preds.npz")


if __name__ == "__main__":
    main()
