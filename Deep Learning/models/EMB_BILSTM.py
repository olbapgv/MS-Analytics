# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:34:41 2021

@author: Pablo González
"""
import torch
import torch.nn as nn

class EMB_BILSTM(nn.Module):
    # define model elements
    def __init__(self, batch_size, weights):
        super(EMB_BILSTM, self).__init__()

        # Embedding
        self.embeddings = nn.Embedding.from_pretrained(weights)
        self.embeddings.requires_grad = False

        # Parameters:
        embedding_dim = self.embeddings.embedding_dim
        hidden_dim = (128, 64)
        num_layers = 2
        num_classes = 2

        # For the loader:
        self.num_epochs = 10
        self.learning_rate, self.betas = 5e-4, (0.999, 0.9999)


        # Initial Hidden
        self.hidden = (torch.zeros(num_layers, batch_size, hidden_dim[0]).double(),
                       torch.zeros(num_layers, batch_size, hidden_dim[0]).double())
        self.lstm = nn.LSTM(embedding_dim , hidden_dim[0], num_layers,
                            bidirectional = True, batch_first = True )
        self.leaky = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(2*hidden_dim[0], hidden_dim[1] )
        self.fl = nn.Linear( hidden_dim[1], num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
      x = self.embeddings( x.to(int) )
      x, self.hidden = self.lstm(x)
      x = x[:,-1,:]
      x = self.leaky( self.fc(x) )
      x = self.softmax(self.fl(x))
      return x