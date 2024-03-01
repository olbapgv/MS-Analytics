# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 01:24:46 2021

@author: Pablo González
"""

import torch
import torch.nn as nn

class BILSTM2(nn.Module):
    # define model elements
    def __init__(self, input_dim, batch_size = 128):
        super(BILSTM2, self).__init__()

        # Parameters:
        hidden_dim0 = 4096
        hidden_dim1 = 1024
        hidden_dim2 = 128
        hidden_dim3 = 64
        num_layers = 1
        num_classes = 2

        # For the loader:
        self.num_epochs = 15
        self.learning_rate, self.betas = 1e-4, (0.99, 0.999)

        # Initial Hidden

        self.hidden = (torch.zeros(num_layers, batch_size, hidden_dim2).double(),
                       torch.zeros(num_layers, batch_size, hidden_dim2).double())
        
        self.relu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.6)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim0)
        self.fc2 = nn.Linear(hidden_dim0, hidden_dim1)  
        self.norm1 = nn.LayerNorm(hidden_dim1)

        self.lstm = nn.LSTM(hidden_dim1, hidden_dim2, num_layers,
                            bidirectional = True, batch_first = True )
        self.norm2 = nn.LayerNorm(2*hidden_dim2)
        self.fc3 = nn.Linear(2*hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
      x = x.unsqueeze(0)
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.norm1(x)
      x, self.hidden = self.lstm(x)
      x = self.norm2(x)
      x = self.relu(self.fc3(x))
      x = self.relu(self.fc4(x))
      x = self.drop(x)
      x = self.softmax(x)
      x = x.squeeze(0)
      return x