# PA3 - Transformers for Amazon Scenario classification

Fine-tunes `bert-base-uncased` on the Amazon MASSIVE scenario dataset (18 classes)
three ways: a cross-entropy baseline, a custom fine-tuning technique (top-layer
re-initialization), and contrastive learning (SupContrast / SimCLR).

<!-- [AI Assisted: Claude Code] README instructions written with Claude Code -->

## Setup

```bash
# 1. Install PyTorch for your platform first (see https://pytorch.org/get-started/locally/)
pip install torch

# 2. Install the remaining dependencies
pip install -r requirements.txt
```

The dataset (`mteb/amazon_massive_scenario`, English split) downloads automatically
into `assets/` on first run; tokenized features are cached under `assets/cache/`.

## How to run

Each command runs eval (untrained sanity check) -> training -> final test eval, and
writes the training log to `results/<task>/<task>.txt`.

```bash
# Task 1: baseline — BERT [CLS] -> dropout -> 2-layer classifier, cross-entropy, 10 epochs
python main.py --task baseline

# Task 2: custom fine-tuning — re-initialize the last N encoder layers + pooler
# (technique from "Advanced Techniques for Fine-tuning Transformers")
python main.py --task custom --reinit_n_layers 2

# Task 3: contrastive learning — SupContrast loss + jointly-trained classifier head
python main.py --task supcon --batch-size 32

# Task 3 variant: SimCLR (unsupervised) loss instead of SupContrast
python main.py --task supcon --batch-size 32 --contrast-loss simclr
```

Or run all three via the shell script:

```bash
bash run.sh
```

### Useful flags (see `arguments.py` for the full list)

| flag | default | meaning |
|---|---|---|
| `--task` | `baseline` | `baseline` / `custom` / `supcon` |
| `--batch-size` | 16 | batch size (larger helps contrastive learning) |
| `--learning-rate` | 2e-5 | AdamW learning rate (linear warmup + decay) |
| `--n-epochs` | 10 | training epochs |
| `--drop-rate` | 0.1 | dropout on the [CLS] embedding |
| `--max-len` | 20 | tokenizer max sequence length |
| `--reinit_n_layers` | 0 | (custom) # of top encoder layers to re-initialize |
| `--contrast-loss` | `supcon` | (supcon) `supcon` or `simclr` |
| `--temperature` | 0.07 | (supcon) contrastive loss temperature |
| `--ignore-cache` | off | re-tokenize instead of loading cached features |
| `--debug` | off | slice every split tiny for a fast CPU sanity check |

## Model / implementation notes

- **Baseline** (`ScenarioModel`): pretrained BERT encoder; the `last_hidden_state`
  [CLS] token goes through dropout and then the provided `Classifier` head.
  AdamW + linear schedule with 10% warmup.
- **Custom** (`CustomModel`): inherits the baseline and re-initializes the pooler
  and the last `--reinit_n_layers` encoder layers with BERT's pretraining
  initialization (normal(0, initializer_range)), which helps the top layers shed
  their pretraining specialization.
- **SupCon/SimCLR** (`SupConModel`): adds a linear projection head on top of the
  dropped [CLS] embedding. Data augmentation is SimCSE-style: the same batch is
  passed through the encoder twice, and the two different dropout masks yield two
  views per sentence for the contrastive loss (`loss.py`, from the SupContrast
  repo). Since evaluation takes an argmax over class logits, the classifier head
  is trained jointly with cross-entropy; `forward()` returns those logits so the
  shared `run_eval` works unchanged.

## Results (UCSD Datahub, NVIDIA A30, 10 epochs each)

| experiment | test accuracy |
|---|---|
| baseline (bs16) | 0.9156 |
| custom: re-init top 2 layers (bs16) | 0.9159 |
| **supcon: SupContrast (bs64)** — graded run | **0.9206** |
| supcon: SupContrast (bs16) | 0.9203 |
| supcon: SupContrast (bs128) | 0.9176 |
| supcon: SimCLR (bs64) | 0.9193 |

Observations: SupCon slightly outperforms SimCLR, as its label-aware positives pull
whole classes together while SimCLR's only positives are the two dropout views of
the same sentence. Batch size has a modest, non-monotonic effect on SupCon — larger
batches provide more negatives per step, but with epochs fixed they also mean fewer
optimizer steps, and the two effects roughly cancel.

## Results files

Do not modify the code writing `results/[task]/[task].txt` — the autograder uses it.
Each file records the argparse config, per-epoch training loss, validation accuracy
per epoch, and test accuracy before/after training.
