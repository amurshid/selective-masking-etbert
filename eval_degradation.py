#!/usr/bin/env python3
"""
eval_degradation.py  --  the robustness matrix.

For each trained model {baseline, ours}, load it once and evaluate on every
degraded test set {mask,random} x {0,25,50,75,100}, reporting accuracy + macro-F1
AND full per-class precision/recall/F1 (all 20 classes). Reuses the SAME
Classifier/read_dataset/evaluate as training, so metrics match.
Writes:
  results/degradation_results.csv    (overall: accuracy + macro-F1)
  results/degradation_perclass.csv   (per-class P/R/F1 for all 20 classes)
"""
import os, sys, csv, argparse
import torch

REPO = "/Users/ahnaf/Projects/ET-BERT"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "fine-tuning"))

from run_classifier import Classifier, read_dataset, evaluate, count_labels_num
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts

MODELS = [("baseline",     "models/finetuned_ustc_full.bin"),
          ("ours_p50",     "models/finetuned_ustc_masked.bin"),
          ("ours_p70",     "models/finetuned_ustc_masked_p70.bin"),
          ("normalize_p70","models/finetuned_ustc_normalize_p70.bin")]
MODES  = ["mask", "random", "zero"]
LEVELS = [0, 25, 50, 75, 100]

# label id -> class name (order confirmed from preprocessing output)
CLASS_NAMES = ["Shifu","Geodo","BitTorrent","Gmail","Neris","SMB","Nsis-ay","Cridex",
               "Zeus","Miuref","MySQL","Virut","Weibo","WorldOfWarcraft","Outlook",
               "Facetime","Tinba","FTP","Htbot","Skype"]

def per_class_prf(conf):                 # conf[pred, gold]
    n, eps, out = conf.size(0), 1e-9, []
    for i in range(n):
        tp = conf[i, i].item()
        p  = tp / (conf[i, :].sum().item() + eps)
        r  = tp / (conf[:, i].sum().item() + eps)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        out.append((p, r, f1))
    return out

def build_args():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    parser.add_argument("--pooling", default="first")
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    argv = ["--vocab_path","models/encryptd_vocab.txt",
            "--train_path","datasets/ustc-full-tsv/test_dataset.tsv",  # dummy: label count only
            "--dev_path",  "datasets/ustc-full-tsv/test_dataset.tsv",  # dummy: unused
            "--config_path","models/bert/base_config.json",
            "--embedding","word_pos_seg","--encoder","transformer",
            "--mask","fully_visible","--seq_length","128","--batch_size","32"]
    args = parser.parse_args(argv)
    args = load_hyperparam(args)
    set_seed(args.seed)
    args.tokenizer    = str2tokenizer[args.tokenizer](args)
    args.labels_num   = count_labels_num(args.train_path)
    args.soft_targets = False
    args.device = (torch.device("mps") if torch.backends.mps.is_available()
                   else torch.device("cuda:0") if torch.cuda.is_available()
                   else torch.device("cpu"))
    return args

def main():
    os.chdir(REPO)
    args = build_args()
    print("labels_num=%d  device=%s\n" % (args.labels_num, args.device))
    model = Classifier(args)

    overall, perclass = [], []
    for mname, mpath in MODELS:
        model.load_state_dict(torch.load(mpath, map_location="cpu"), strict=False)
        model.to(args.device); args.model = model
        for mode in MODES:
            for lvl in LEVELS:
                ds = read_dataset(args, "datasets/ustc-degraded/%s/test_deg%03d.tsv" % (mode, lvl))
                acc, conf = evaluate(args, ds)
                prf = per_class_prf(conf)
                f1  = sum(x[2] for x in prf) / len(prf)             # macro-F1
                overall.append((mname, mode, lvl, "%.4f"%acc, "%.4f"%f1))
                for cid, (p, r, f1c) in enumerate(prf):
                    perclass.append((mname, mode, lvl, cid, CLASS_NAMES[cid],
                                     "%.4f"%p, "%.4f"%r, "%.4f"%f1c))
                print("  %-8s %-6s deg=%3d%%   acc=%.4f  macroF1=%.4f" % (mname, mode, lvl, acc, f1))

    os.makedirs("results", exist_ok=True)
    with open("results/degradation_results.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["model","mode","degradation_pct","accuracy","macro_f1"]); w.writerows(overall)
    with open("results/degradation_perclass.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["model","mode","degradation_pct","label","class_name","precision","recall","f1"]); w.writerows(perclass)
    print("\nsaved -> results/degradation_results.csv  and  results/degradation_perclass.csv")

if __name__ == "__main__":
    main()
