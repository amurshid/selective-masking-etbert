#!/usr/bin/env bash
# Regenerate the reported tables and figures.
#
# Stage 1 runs from a fresh clone and rebuilds both figures from the result
# tables tracked in results/. Stages 2 onward reproduce those tables from
# scratch and additionally require the USTC-TFC2016 dataset and the ET-BERT
# pre-trained checkpoint, neither of which is redistributable here; see the
# README for how to obtain them. Fine-tuning the ten model variants takes
# roughly 50 GPU-hours.
#
# Usage:  ./reproduce.sh figures     regenerate figures only (default)
#         ./reproduce.sh all         full pipeline, needs data + checkpoint

set -euo pipefail
cd "$(dirname "$0")"
STAGE="${1:-figures}"

need() {
  [ -e "$1" ] || { echo "missing: $1"; echo "$2"; exit 1; }
}

echo "== Stage 1: figures from tracked result tables =="
need results/degradation_results.csv "run './reproduce.sh all' to regenerate it"
need results/degradation_perclass.csv "run './reproduce.sh all' to regenerate it"
python plot_degradation.py
python plot_perclass_bar.py

if [ "$STAGE" != "all" ]; then
  echo
  echo "Done. Pass 'all' to rebuild the result tables from the models."
  exit 0
fi

need datasets/ustc-full-tsv/train_dataset.tsv "see README: Data"
need models/pre-trained_model.bin            "see README: Pre-trained model"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo
echo "== Stage 2: degraded and stripped evaluation sets =="
python make_degraded_tests.py
python make_stripped_dataset.py

echo
echo "== Stage 3: fine-tuning =="
echo "Not scripted. Each variant is a separate run of"
echo "  fine-tuning/run_classifier_mask.py (or run_classifier.py for the baseline"
echo "  and the 38->54 byte strip), with the flags listed in the README."
echo "Expected artifacts in models/ before continuing:"
for m in finetuned_ustc_full finetuned_ustc_masked finetuned_ustc_masked_p70 \
         finetuned_ustc_normalize_p70 finetuned_ustc_normalize_p100 \
         finetuned_ustc_masked_p100 finetuned_ustc_strip54; do
  printf '  %-40s %s\n' "$m.bin" "$([ -f "models/$m.bin" ] && echo present || echo MISSING)"
done
[ -f models/finetuned_ustc_strip54.bin ] || { echo "fine-tune the variants above, then re-run"; exit 1; }

echo
echo "== Stage 4: evaluation sweeps =="
python eval_degradation.py
python eval_preds.py
python eval_strip54.py
python adversarial_header.py
python bench_overhead.py

echo
echo "== Stage 5: statistical analysis =="
python bootstrap_ci.py

echo
echo "== Stage 6: figures =="
python plot_degradation.py
python plot_perclass_bar.py
echo
echo "Done."
