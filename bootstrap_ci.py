#!/usr/bin/env python3
"""
bootstrap_ci.py  --  confidence intervals for the robustness matrix.

Reviewer 2 asked for statistical confidence measures. This computes
percentile-bootstrap 95% intervals over the test set by resampling packets with
replacement (B=2000). Note what this does and does not cover: it quantifies
test-set sampling variance for a FIXED trained model. It says nothing about
training variance across random seeds, which is a separate experiment.

Three families of statistic, all resampled on a COMMON set of packet indices so
that comparisons are properly paired:
  1. per-cell macro-F1 CI            -> results/bootstrap_ci.csv
  2. paired model-vs-baseline delta  -> results/bootstrap_paired.csv
  3. degradation drop (0% -> 100%) per model, and the reduction RATIO
     baseline_drop / model_drop, i.e. the "fivefold" claim  -> results/bootstrap_headline.csv
Pairing matters: the same packets appear at every degradation level, so a paired
resample cancels packet-difficulty noise that separate intervals would leave in.
"""
import csv, itertools, os
import numpy as np

B       = int(os.environ.get("BOOT_B", "2000"))
ALPHA   = 5.0                      # -> 95% interval
N_CLS   = 20
SEED    = 20260907
BASELINE = "baseline"
PREDS   = os.environ.get("PREDS_NPZ", "results/preds.npz")
OUTDIR  = os.environ.get("BOOT_OUTDIR", "results")


def macro_f1(gold, pred, w=None):
    """Macro-F1 from label vectors; w = per-sample multiplicity (bootstrap counts)."""
    idx = gold.astype(np.int64) * N_CLS + pred.astype(np.int64)
    conf = np.bincount(idx, weights=w, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(conf).astype(float)
    p  = tp / (conf.sum(axis=0) + 1e-9)
    r  = tp / (conf.sum(axis=1) + 1e-9)
    return float(np.mean(np.where(p + r == 0, 0.0, 2 * p * r / (p + r + 1e-9))))


def ci(samples):
    a = np.asarray(samples, dtype=float)
    return float(np.percentile(a, ALPHA / 2)), float(np.percentile(a, 100 - ALPHA / 2))


def main():
    z = np.load(PREDS)
    keys = list(z.keys())
    models = sorted({k.split("|")[0] for k in keys}, key=lambda m: (m != BASELINE, m))
    modes  = sorted({k.split("|")[1] for k in keys})
    levels = sorted({int(k.split("|")[2]) for k in keys})
    n = z[keys[0]].shape[0]
    print("cells=%d  models=%s  n=%d  B=%d" % (len(keys), models, n, B))

    # every cell shares the same packet order, so one index draw serves all of them
    rng = np.random.default_rng(SEED)
    counts = np.empty((B, n), dtype=np.int32)
    for b in range(B):
        counts[b] = np.bincount(rng.integers(0, n, n), minlength=n)

    gold = {k: z[k][:, 0] for k in keys}
    pred = {k: z[k][:, 1] for k in keys}

    def boot_f1(key):
        g, p = gold[key], pred[key]
        return np.array([macro_f1(g, p, counts[b].astype(float)) for b in range(B)])

    cache = {}
    def F(key):
        if key not in cache:
            cache[key] = boot_f1(key)
        return cache[key]

    # ---- 1. per-cell CI -------------------------------------------------
    rows = []
    for m, mode, lvl in itertools.product(models, modes, levels):
        k = "%s|%s|%d" % (m, mode, lvl)
        if k not in gold: continue
        pt = macro_f1(gold[k], pred[k])
        lo, hi = ci(F(k))
        rows.append((m, mode, lvl, "%.4f" % pt, "%.4f" % lo, "%.4f" % hi, "%.4f" % ((hi - lo) / 2)))
        print("  %-15s %-6s deg=%3d%%  F1=%.4f  [%.4f, %.4f]" % (m, mode, lvl, pt, lo, hi))
    with open(os.path.join(OUTDIR, "bootstrap_ci.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model","mode","degradation_pct","macro_f1","ci_lo","ci_hi","half_width"]); w.writerows(rows)

    # ---- 2. paired delta vs baseline ------------------------------------
    rows = []
    for m, mode, lvl in itertools.product(models, modes, levels):
        if m == BASELINE: continue
        k, kb = "%s|%s|%d" % (m, mode, lvl), "%s|%s|%d" % (BASELINE, mode, lvl)
        if k not in gold or kb not in gold: continue
        d  = F(k) - F(kb)                                   # paired: same resamples
        pt = macro_f1(gold[k], pred[k]) - macro_f1(gold[kb], pred[kb])
        lo, hi = ci(d)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        rows.append((m, mode, lvl, "%.4f" % pt, "%.4f" % lo, "%.4f" % hi, sig))
        print("  %-15s %-6s deg=%3d%%  d=%+.4f [%+.4f, %+.4f] excludes_zero=%s" % (m, mode, lvl, pt, lo, hi, sig))
    with open(os.path.join(OUTDIR, "bootstrap_paired.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model","mode","degradation_pct","delta_vs_baseline","ci_lo","ci_hi","excludes_zero"]); w.writerows(rows)

    # ---- 3. headline: drop 0->100 and the reduction ratio ----------------
    rows = []
    for mode in modes:
        kb0, kb1 = "%s|%s|0" % (BASELINE, mode), "%s|%s|100" % (BASELINE, mode)
        if kb0 not in gold or kb1 not in gold: continue   # no 0->100 pair (e.g. adversarial file)
        base_drop = F(kb0) - F(kb1)
        base_pt   = macro_f1(gold[kb0], pred[kb0]) - macro_f1(gold[kb1], pred[kb1])
        for m in models:
            k0, k1 = "%s|%s|0" % (m, mode), "%s|%s|100" % (m, mode)
            if k0 not in gold: continue
            drop = F(k0) - F(k1)
            pt   = macro_f1(gold[k0], pred[k0]) - macro_f1(gold[k1], pred[k1])
            lo, hi = ci(drop)
            if m == BASELINE:
                rows.append((m, mode, "%.4f" % pt, "%.4f" % lo, "%.4f" % hi, "", "", ""))
            else:
                ratio = base_drop / np.maximum(drop, 1e-6)   # paired ratio
                rlo, rhi = ci(ratio)
                rows.append((m, mode, "%.4f" % pt, "%.4f" % lo, "%.4f" % hi,
                             "%.2f" % (base_pt / max(pt, 1e-6)), "%.2f" % rlo, "%.2f" % rhi))
            print("  %-15s %-6s drop=%.4f [%.4f, %.4f]%s" % (m, mode, pt, lo, hi,
                  "" if m == BASELINE else "  ratio=%.2f [%.2f, %.2f]" % (base_pt / max(pt, 1e-6), rlo, rhi)))
    with open(os.path.join(OUTDIR, "bootstrap_headline.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model","mode","drop_0_to_100","drop_ci_lo","drop_ci_hi",
                                       "reduction_ratio_vs_baseline","ratio_ci_lo","ratio_ci_hi"]); w.writerows(rows)
    print("\nsaved -> %s/{bootstrap_ci,bootstrap_paired,bootstrap_headline}.csv" % OUTDIR)


if __name__ == "__main__":
    main()
