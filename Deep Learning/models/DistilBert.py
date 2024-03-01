# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 19:56:29 2021

@author: Pablo González
"""
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

class DistilBert(nn.Module):
    # define model elements
    def __init__(self):
        super(DistilBert, self).__init__()

        # Parameters:
        num_classes = 2
        hidden_dim = (768, 128)

        # Pre-trained model
        self.distilbert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', 
                      num_labels= num_classes ,output_hidden_states=True,output_attentions=False)
        
        # For the loader:
        self.num_epochs = 5
        self.learning_rate, self.betas = 5e-5, (0.999, 0.9999)

        # Initial Hidden
        self.leaky = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear( hidden_dim[0] , hidden_dim[1] )
        self.fc2 = nn.Linear( hidden_dim[1] , num_classes )
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
      x = x.transpose(0,1)
      x, mask = x[0].to(int), x[1].to(int)
      x = self.distilbert( input_ids = x , attention_mask = mask)
      x = x.hidden_states[1]
      x = x[:,-1,:]
      x = self.leaky( self.fc1( x ) )
      x = self.fc2( x )
      x = self.softmax(x)
      return x