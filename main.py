import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import ScenarioModel, SupConModel, CustomModel
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer, out_file):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    # [AI Assisted: Claude Code]
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: attach an optimizer AND an lr scheduler to the model as model.optimizer
    #        and model.scheduler (the loop below calls model.optimizer.step() and
    #        model.scheduler.step(), so both must exist as attributes on the model).
    # [AI Assisted: Claude Code]
    total_steps = len(train_dataloader) * args.n_epochs
    model.setup_optimizer_scheduler(args, total_steps)

    # task3: write a training loop
    # [AI Assisted: Claude Code]
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        run_eval(args, model, datasets, tokenizer, out_file=out_file, split='validation')
        print(f'epoch {epoch_count} | losses {losses}')
        out_file.write(f'epoch {epoch_count} | losses {losses}\n') #IMPORTANT FOR AUTOGRADING - DO NOT CHANGE
  
def custom_train(args, model, datasets, tokenizer, out_file):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    # [AI Assisted: Claude Code]
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: attach an optimizer AND an lr scheduler to the model as model.optimizer
    #        and model.scheduler (the loop below calls model.optimizer.step() and
    #        model.scheduler.step(), so both must exist as attributes on the model).
    # [AI Assisted: Claude Code]
    # The custom technique (re-initializing the top N encoder layers) lives in
    # CustomModel.__init__; the training loop itself matches the baseline so the
    # comparison between baseline and custom isolates the technique.
    total_steps = len(train_dataloader) * args.n_epochs
    model.setup_optimizer_scheduler(args, total_steps)

    # task3: write a training loop
    # [AI Assisted: Claude Code]
    for epoch_count in range(args.n_epochs):
      losses = 0
      model.train()

      for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
          inputs, labels = prepare_inputs(batch)
          logits = model(inputs, labels)
          loss = criterion(logits, labels)
          loss.backward()

          model.optimizer.step()
          model.scheduler.step()
          model.zero_grad()
          losses += loss.item()

      run_eval(args, model, datasets, tokenizer, out_file=out_file, split='validation')
      print(f'epoch {epoch_count} | losses {losses}')
      out_file.write(f'epoch {epoch_count} | losses {losses}\n') #IMPORTANT FOR AUTOGRADING - DO NOT CHANGE

def run_eval(args, model, datasets, tokenizer, out_file, split='validation'): 
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs, labels)
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} accuracy: {acc/len(datasets[split])}, | dataset split {split} size: {len(datasets[split])}')
    out_file.write(f'{split} accuracy: {acc/len(datasets[split])}, | dataset split {split} size: {len(datasets[split])}\n') #IMPORTANT FOR AUTOGRADING - DO NOT CHANGE

def supcon_train(args, model, datasets, tokenizer, out_file):
    from loss import SupConLoss
    criterion = SupConLoss()

    # NOTE: the provided run_eval scores accuracy by argmax over class logits, while the
    #       contrastive loss trains on embeddings. Per 5(a) you also "fine tune ... for our
    #       classification objective", so a meaningful supcon accuracy needs a trained
    #       classifier whose logits run_eval can read at eval time. The writeup does not
    #       mandate how -- any approach that produces eval-time logits is fine. (The inherited
    #       classifier head is available as self.classify if you choose to reuse it.)

    # task1: load training split of the dataset
    # [AI Assisted: Claude Code]
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup optimizer_scheduler in your model
    # [AI Assisted: Claude Code]
    total_steps = len(train_dataloader) * args.n_epochs
    model.setup_optimizer_scheduler(args, total_steps)

    # task3: write a training loop for SupConLoss function
    # [AI Assisted: Claude Code]
    # Two dropout-augmented "views" of each sentence (SimCSE-style) feed the
    # contrastive loss: with labels it is SupContrast, without labels it
    # degenerates to SimCLR (--contrast-loss simclr). The classifier head is
    # trained jointly with cross-entropy so run_eval's argmax over logits works.
    ce_criterion = nn.CrossEntropyLoss()
    for epoch_count in range(args.n_epochs):
      losses = 0
      model.train()

      for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
          inputs, labels = prepare_inputs(batch)

          feat1, logits = model.contrast(inputs)
          feat2, _ = model.contrast(inputs)  # second pass -> different dropout mask
          features = torch.stack([feat1, feat2], dim=1)  # [bsz, n_views=2, feat_dim]

          if args.contrast_loss == 'simclr':
              loss = criterion(features)          # unsupervised: SimCLR
          else:
              loss = criterion(features, labels)  # supervised: SupContrast
          loss = loss + ce_criterion(logits, labels)  # joint classifier training
          loss.backward()

          model.optimizer.step()
          model.scheduler.step()
          model.zero_grad()
          losses += loss.item()

      run_eval(args, model, datasets, tokenizer, out_file=out_file, split='validation')
      print(f'epoch {epoch_count} | losses {losses}')
      out_file.write(f'epoch {epoch_count} | losses {losses}\n') #IMPORTANT FOR AUTOGRADING - DO NOT CHANGE


if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  #IMPORTANT FOR AUTOGRADING - DO NOT CHANGE
  results_dir = f"results/{args.task}"
  if not os.path.exists(results_dir):
    os.mkdir(results_dir)
  f = open(f"{results_dir}/{args.task}.txt", "w")
  f.write(f"{str(args)}\n")
  #IMPORTANT FOR AUTOGRADING - DO NOT CHANGE

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)

  # Derive the number of output classes from the data instead of hard-coding it.
  # (Computed from the FULL feature set, before any debug slicing below.)
  target_size = len({inst.scenario_label for split_feats in features.values() for inst in split_feats})
  print(f'target_size (number of scenario classes): {target_size}')

  # [AI Assisted: Claude Code]
  # Debug mode: train/eval on a small slice of each split so a full pass through
  # main.py finishes in minutes on CPU. For sanity-checking the code only.
  if args.debug:
    features = {split: feats[:args.batch_size * 8] for split, feats in features.items()}
    print(f'DEBUG MODE: sliced every split to {args.batch_size * 8} examples')

  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
 
  if args.task == 'baseline':
    model = ScenarioModel(args, tokenizer, target_size=target_size).to(device)
    run_eval(args, model, datasets, tokenizer, f, split='validation') 
    run_eval(args, model, datasets, tokenizer, f, split='test') 
    baseline_train(args, model, datasets, tokenizer, f) 
    run_eval(args, model, datasets, tokenizer, f, split='test') 
  elif args.task == 'custom': # you can have multiple custom tasks for different techniques
    model = CustomModel(args, tokenizer, target_size=target_size).to(device)
    run_eval(args, model, datasets, tokenizer, f, split='validation') 
    run_eval(args, model, datasets, tokenizer, f, split='test') 
    custom_train(args, model, datasets, tokenizer, f) 
    run_eval(args, model, datasets, tokenizer, f, split='test') 
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=target_size).to(device)
    run_eval(args, model, datasets, tokenizer, f, split='validation') 
    run_eval(args, model, datasets, tokenizer, f, split='test') 
    supcon_train(args, model, datasets, tokenizer, f) 
    run_eval(args, model, datasets, tokenizer, f, split='test')

  f.close()