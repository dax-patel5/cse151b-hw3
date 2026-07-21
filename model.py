import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
# Use torch's AdamW: transformers.AdamW is deprecated and was removed in newer versions.
from torch.optim import AdamW

class ScenarioModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size

    # task1: add necessary class variables as you wish.
    # [AI Assisted: Claude Code]
    self.args = args

    # task2: initilize the dropout and classify layers
    # [AI Assisted: Claude Code]
    self.dropout = nn.Dropout(args.drop_rate)
    self.classify = Classifier(args, target_size)

  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    # [AI Assisted: Claude Code]
    self.encoder = BertModel.from_pretrained('bert-base-uncased')

    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  # [AI Assisted: Claude Code]
  def setup_optimizer_scheduler(self, args, total_steps):
    """Attach an AdamW optimizer and a linear-warmup LR schedule to the model.

    The training loops in main.py call model.optimizer.step() and
    model.scheduler.step(), so both must exist as attributes on the model.
    """
    self.optimizer = AdamW(self.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # 10% of training used for linear LR warmup, then linear decay to 0
    warmup_steps = int(0.1 * total_steps)
    self.scheduler = get_linear_schedule_with_warmup(
        self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

  def forward(self, inputs, targets):
    """
    task1:
        feeding the input to the encoder,
    task2:
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument,
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    # [AI Assisted: Claude Code]
    outputs = self.encoder(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] is the first token
    logits = self.classify(self.dropout(cls_token))
    return logits


class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)

    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model
    # [AI Assisted: Claude Code]
    # Technique from "Advanced Techniques for Fine-tuning Transformers":
    # re-initializing the top N encoder layers (plus the pooler). The top layers of a
    # pretrained BERT are the most specialized to the pretraining objective, so
    # re-initializing them can help the model adapt to the downstream task.
    if args.reinit_n_layers > 0:
      self._reinit_top_layers(args.reinit_n_layers)

  # [AI Assisted: Claude Code]
  def _reinit_top_layers(self, n_layers):
    """Re-initialize the pooler and the last `n_layers` encoder layers with the
    same scheme BERT uses at pretraining time (normal(0, initializer_range))."""
    print(f"Re-initializing the last {n_layers} encoder layers and the pooler")
    std = self.encoder.config.initializer_range
    self.encoder.pooler.dense.weight.data.normal_(mean=0.0, std=std)
    self.encoder.pooler.dense.bias.data.zero_()
    for layer in self.encoder.encoder.layer[-n_layers:]:
      for module in layer.modules():
        if isinstance(module, nn.Linear):
          module.weight.data.normal_(mean=0.0, std=std)
          if module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
          module.weight.data.fill_(1.0)
          module.bias.data.zero_()

class SupConModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
    # [AI Assisted: Claude Code]
    # Linear projection head mapping the [CLS] embedding into the space where the
    # contrastive (SupCon / SimCLR) loss is computed.
    self.head = nn.Linear(args.embed_dim, feat_dim)

  def forward(self, inputs, targets):

    """
    task1:
        feeding the input to the encoder,
    task2:
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument,
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding.

    NOTE: the shared run_eval scores accuracy by argmax over class logits, so the model must
    be able to produce classification logits at eval time (see the note in supcon_train). How
    you reconcile that with this contrastive forward is your design choice.
    """
    # [AI Assisted: Claude Code]
    # Design choice: forward() returns classification logits so the shared run_eval
    # (argmax over logits) works unchanged. The contrastive embedding used for
    # training is produced by contrast() below; supcon_train trains the classifier
    # head jointly with the contrastive loss so these logits are meaningful.
    outputs = self.encoder(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :]
    logits = self.classify(self.dropout(cls_token))
    return logits

  # [AI Assisted: Claude Code]
  def contrast(self, inputs):
    """One stochastic pass through the model; returns (embedding, logits).

    The [CLS] token goes through the dropout layer (SimCSE-style: in train mode,
    two calls on the same batch yield two different "views" of each sentence
    because dropout masks differ), then the normalized dropout output goes
    through the linear head. The head output is L2-normalized so the SupCon /
    SimCLR loss operates on unit vectors. The logits from the same dropped
    [CLS] are returned so the classifier head can be trained jointly.
    """
    outputs = self.encoder(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :]
    dropped = self.dropout(cls_token)
    embedding = self.head(F.normalize(dropped, dim=1))
    embedding = F.normalize(embedding, dim=1)
    logits = self.classify(dropped)
    return embedding, logits
