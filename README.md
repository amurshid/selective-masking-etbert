# Selective Header Masking for Robust Encrypted Traffic Classification

Encrypted traffic classifiers built on pre-trained transformers reach near-perfect
accuracy on clean benchmarks, but much of that accuracy rests on a handful of
transport-layer header bytes. When those bytes are missing or corrupted in the
field — partial captures, middlebox rewriting, truncation — accuracy can collapse
without warning.

This project introduces **selective header masking**: a lightweight change to the
fine-tuning recipe that randomly hides the residual TCP-header tokens during
training, forcing the model to rely on the rest of the packet. The result is a
classifier that degrades *gracefully* instead of catastrophically.

> Built on [ET-BERT](https://github.com/linwhitehat/ET-BERT) (Lin et al., WWW '22).
> See [Acknowledgements](#acknowledgements).

## Headline result

Under progressive header degradation on **USTC-TFC2016** (20 classes), our masked
model's macro-F1 drops by only **0.15** across full header masking, versus **0.46**
for a conventionally fine-tuned baseline — a roughly **3× smaller** degradation.

| Macro-F1 (header masked) | Clean | 50% | 100% |
|--------------------------|:-----:|:---:|:----:|
| Baseline                 | 0.976 | 0.652 | 0.514 |
| **Ours (p = 0.7)**       | 0.960 | 0.836 | **0.807** |

![Robustness under header degradation](results/fig_degradation.png)

The gains hold even under a corruption pattern never seen during training
(randomized rather than masked headers), indicating genuine robustness rather than
memorization of a single failure mode. Full numbers are in
[`results/degradation_results.csv`](results/degradation_results.csv).

## What's new in this repo

The contribution is implemented in these files (everything else is the ET-BERT base):

| File | Purpose |
|------|---------|
| [`fine-tuning/run_classifier_mask.py`](fine-tuning/run_classifier_mask.py) | Fine-tuning with selective header masking (`--mask_rate`, `--header_tokens`, `--mask_op`) |
| [`make_degraded_tests.py`](make_degraded_tests.py) | Builds degraded test sets (mask / random corruption, 0–100%) |
| [`eval_degradation.py`](eval_degradation.py) | Evaluation-only sweep producing the degradation matrix |
| [`plot_degradation.py`](plot_degradation.py) | Generates the IEEE robustness figure |

## Reproducing the experiments

1. **Environment** — Python 3.10, PyTorch 2.x. Works on CUDA or Apple MPS.
2. **Data** — Obtain [USTC-TFC2016](https://github.com/yungshenglu/USTC-TFC2016) and
   preprocess with `data_process/` into train/valid/test TSVs. See that script for
   the pipeline (byte-bigram tokenization; the first 38 bytes — Ethernet + IPv4 +
   ports — are stripped, leaving the 16-byte residual TCP header as the masking target).
3. **Pre-trained model** — download `pre-trained_model.bin` from the
   [ET-BERT release](https://github.com/linwhitehat/ET-BERT).
4. **Fine-tune with masking:**
   ```bash
   PYTHONPATH=. python -u fine-tuning/run_classifier_mask.py \
     --pretrained_model_path models/pre-trained_model.bin \
     --vocab_path models/encryptd_vocab.txt \
     --train_path datasets/<your>/train_dataset.tsv \
     --dev_path   datasets/<your>/valid_dataset.tsv \
     --test_path  datasets/<your>/test_dataset.tsv \
     --epochs_num 10 --batch_size 32 --seq_length 128 --learning_rate 2e-5 \
     --embedding word_pos_seg --encoder transformer --mask fully_visible \
     --mask_rate 0.7 --header_tokens 16 --mask_op mask \
     --output_model_path models/finetuned_masked_p70.bin
   ```
5. **Build degraded tests, evaluate, and plot:**
   ```bash
   python make_degraded_tests.py
   python eval_degradation.py
   python plot_degradation.py   # -> results/fig_degradation.pdf
   ```

> Trained weights (`*.bin`) and datasets are not tracked here due to size; the steps
> above reproduce them from the public ET-BERT base model and USTC-TFC2016.

## Acknowledgements

This work builds directly on **ET-BERT**:

> Xinjie Lin, Gang Xiong, Gaopeng Gou, Zhen Li, Junzheng Shi, Jing Yu.
> *ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers
> for Encrypted Traffic Classification.* WWW 2022.
> [Paper](https://dl.acm.org/doi/10.1145/3485447.3512217) ·
> [Code](https://github.com/linwhitehat/ET-BERT)

The base model architecture, tokenization, and pre-training code originate from the
ET-BERT repository (MIT License). The selective masking strategy, degradation
protocol, evaluation sweep, and analysis in this repository are my own contribution.
