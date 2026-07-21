#!/usr/bin/env bash
# Run each task end-to-end (eval -> train -> eval). Results are written to
# results/<task>/<task>.txt. Defaults in arguments.py are now set for a working
# run (--embed-dim 768, --n-epochs 10, etc.), so the flags below are just the
# knobs most worth tuning. Run tasks individually by commenting out the others.

# Task 1: baseline BERT + classifier head, cross-entropy loss.
python main.py --task baseline

# Task 2: custom fine-tuning technique (e.g. reinitialize the last N encoder layers).
python main.py --task custom --reinit_n_layers 2

# Task 3: contrastive learning (SupContrast / SimCLR). Larger batches usually help;
# batch size 64 is the graded configuration.
python main.py --task supcon --batch-size 64

# [AI Assisted: Claude Code] Variants of task 3:
#   SimCLR (unsupervised) loss instead of SupContrast:
# python main.py --task supcon --batch-size 32 --contrast-loss simclr
#   Different batch sizes (recommended exploration in the writeup):
# python main.py --task supcon --batch-size 16
# python main.py --task supcon --batch-size 64

# [AI Assisted: Claude Code] Quick sanity check on a tiny data slice (CPU-friendly):
# python main.py --task baseline --debug --n-epochs 1
