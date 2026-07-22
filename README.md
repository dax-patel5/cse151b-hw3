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

**UCSD DSMLP/datahub note:** the stock conda environment already has everything
needed (torch, transformers, datasets) — no installs required. If imports fail with
`operator torchvision::nms does not exist`, a stale torch in `~/.local` is shadowing
the conda one; fix with `export PYTHONNOUSERSITE=1` before running.

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
# (batch size 64 is the graded configuration in the results table below)
python main.py --task supcon --batch-size 64

# Task 3 variant: SimCLR (unsupervised) loss instead of SupContrast
python main.py --task supcon --batch-size 64 --contrast-loss simclr
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

Full training logs: the graded runs are `results/<task>/<task>.txt`; the exploration
runs are kept alongside as `results/supcon/supcon_bs16.txt`, `supcon_bs128.txt`,
and `supcon_simclr_bs64.txt`.

## Results files

Do not modify the code writing `results/[task]/[task].txt` — the autograder uses it.
Each file records the argparse config, per-epoch training loss, validation accuracy
per epoch, and test accuracy before/after training.

## AI usage disclosure

Code sections marked with `[AI Assisted: Claude Code]` comments were written with
the help of Claude Code (Fable 5), as permitted by the course policy. Every added
block carries the tag, so `grep -rn "AI Assisted" *.py` shows exactly what was
AI-written; the untagged starter code is unchanged.

Here are some of the prompts I used:

"Do this homework entirely. Any code you add must be tagged with [AI Assisted:
Claude Code]. Once you are done writing the models, start training only to ensure
that it works. I will then wire this up to a git repo and train models on ucsd
datahub."

"push the current repo to this remote. after that, i will go to datahub. give me a
bunch of diagnostic commands (what version, what venv, etc) commands to run after
cloning this repo at root of datahub and cding into it. based on the response from
datahub here, you will construct the commands for the actual training runs."

My workflow: Claude wrote the implementation task-by-task (tokenizer, feature prep,
the three model classes, the three training loops) and I checked each code addition
step by step against the assignment PDF before moving on. Claude smoke-tested
everything locally in a debug mode before any GPU time was spent. For the real
training I ran the commands myself on UCSD datahub, pasting outputs back: Claude
built the environment diagnostics, diagnosed a broken torch import on the pod (a
stale `~/.local` torch shadowing the conda one, fixed with `PYTHONNOUSERSITE=1`),
and constructed the exact training commands. When the graded runs cleared the
thresholds, Claude suggested the optional exploration runs (SimCLR variant and the
batch-size sweep), which I then ran; the observations in the Results section came
out of discussing those numbers. There were also smaller prompts around git/GitHub
setup, log bookkeeping, and this README. I had no code-quality concerns and did not
manually edit the code.
