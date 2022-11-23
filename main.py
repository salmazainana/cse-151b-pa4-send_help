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
from model import IntentModel, SupConModel, CustomModel
from torch import nn
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

device = 'cuda'

def baseline_train(args, model, datasets):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, dataset=datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.scheduler = get_linear_schedule_with_warmup(model.optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.n_epochs)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        print(epoch_count)
        for step, batch in progress_bar(enumerate(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model.forward(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
    
        run_eval(args, model, datasets, split='validation')
        print('epoch', epoch_count, '| losses:', losses)
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, dataset=datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(len(train_dataloader)*args.n_epochs)
    model.scheduler = get_linear_schedule_with_warmup(model.optimizer, num_warmup_steps=65, num_training_steps=len(train_dataloader)*args.n_epochs)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        print(epoch_count)
        for step, batch in progress_bar(enumerate(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model.forward(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
    
        run_eval(args, model, datasets, split='validation')
        print('epoch', epoch_count, '| losses:', losses)

def run_eval(args, model, datasets, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)

    # task1: load training split of the dataset
    

    # task2: setup optimizer_scheduler in your model

    # task3: write a training loop for SupConLoss function 

if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
    
    print(features)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
 
  if args.task == 'baseline':
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, split='validation')
    run_eval(args, model, datasets, split='test')
    baseline_train(args, model, datasets)
    run_eval(args, model, datasets, split='test')
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    model = CustomModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, split='validation')
    run_eval(args, model, datasets, split='test')
    custom_train(args, model, datasets)
    run_eval(args, model, datasets, split='test')
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets)
   
