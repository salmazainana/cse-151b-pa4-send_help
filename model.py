import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class IntentModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size

    # task1: add necessary class variables as you wish.
    
    # task2: initilize the dropout and classify layers
    self.dropout = nn.Dropout(p = args.drop_rate)
    self.classify = Classifier(args, self.target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained('bert-base-uncased')
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

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
    x = self.encoder.forward(**inputs)
    x = self.dropout(x[0][:,0,:])
    x = self.classify.forward(x)
    return x
  
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


class CustomModel(IntentModel):
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model
    def __init__(self, args, tokenizer, target_size):
        super().__init__(args, tokenizer, target_size)
        self._do_reinit(args.reinit_n_layers)
    
    
    def _do_reinit(self, layers):
        # Re-init pooler.
        self.encoder.pooler.dense.weight.data.normal_(mean=0.0, std=0.02)
        self.encoder.pooler.dense.bias.data.zero_()
        for param in self.encoder.pooler.parameters():
            param.requires_grad = True 
        # Re-init last n layers.
        for n in range(layers):
            self.encoder.encoder.layer[-(n+1)].apply(self._init_weight)

    def _init_weight(self,module):                       
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std= 0.02)
            module.bias.data.zero_()
            
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
class SupConModel(IntentModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
    self.norm = nn.BatchNorm1d(feat_dim)
    self.fc = nn.Linear(feat_dim, target_size)
 
  def forward(self, inputs, targets):

    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    x = self.encoder.forward(**inputs)
    
    x = self.dropout(x[0][:, 0, :]) 
    x= self.fc(x)
    # have also replaced last seld.norm and self.fc with code below and still dont work
    x = F.normalize(x, dim=1)
    
    return x
