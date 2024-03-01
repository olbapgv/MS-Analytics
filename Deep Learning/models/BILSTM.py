# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:42:14 2021

@author: Pablo González
"""
import torch
import torch.nn as nn

class BILSTM(nn.Module):
    # define model elements
    def __init__(self, input_dim, batch_size = 128):
        super(BILSTM, self).__init__()

        # Parameters:
        hidden_dim  = 64
        num_layers = 2
        num_classes = 2

        # For the loader:
        self.num_epochs = 15
        self.learning_rate, self.betas = 5e-4, (0.999, 0.99990)

        # Initial Hidden

        self.hidden = (torch.zeros(num_layers, batch_size, hidden_dim).double(),
                       torch.zeros(num_layers, batch_size, hidden_dim).double())
        
        self.relu = nn.LeakyReLU(0.1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout = 0.25,
                            bidirectional = True, batch_first = True )
        self.fc1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
      x = x.unsqueeze(0)
      x, self.hidden = self.lstm(x)
      x = self.relu(self.fc1(x))
      x = self.softmax(self.fc2(x))
      x = x.squeeze(0)
      return x