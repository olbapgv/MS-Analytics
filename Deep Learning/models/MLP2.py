# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:26:16 2021

@author: Pablo González
"""

import torch.nn as nn

class MLP2(nn.Module):
    # define model elements
    def __init__(self, input_dim):
        super(MLP2, self).__init__()
        
        # Parameters
        hidden_dim5 = 16
        hidden_dim4 = 64
        hidden_dim3 = 256
        hidden_dim2 = 1024
        hidden_dim1 = 4096
        num_classes = 2
        
        # For the loader
        self.num_epochs = 50
        self.learning_rate = 5e-4
        self.betas = (0.999, 0.9999) #(0.9, 0.999)
        
        self.relu = nn.LeakyReLU(0.3)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.norm = nn.BatchNorm1d(num_classes)
        self.fc1, self.norm1 = nn.Linear(input_dim, hidden_dim1), nn.BatchNorm1d(hidden_dim1)
        self.fc2, self.norm2 = nn.Linear(hidden_dim1, hidden_dim2), nn.BatchNorm1d(hidden_dim2)
        self.fc3, self.norm3 = nn.Linear(hidden_dim2, hidden_dim3), nn.BatchNorm1d(hidden_dim3)
        self.fc4, self.norm4 = nn.Linear(hidden_dim3, hidden_dim4), nn.BatchNorm1d(hidden_dim4)
        self.fc5, self.norm5 = nn.Linear(hidden_dim4, hidden_dim5), nn.BatchNorm1d(hidden_dim5)
        self.fc6 = nn.Linear(hidden_dim5, num_classes)
        self.sigmoid = nn.Sigmoid()

    # forward propagate input
    def forward(self, x):
        x = self.norm1(self.relu(self.fc1(x)))
        x = self.norm2(self.relu(self.fc2(x)))
        x = self.norm3(self.relu(self.fc3(x)))
        x = self.drop1(x)
        x = self.norm4(self.relu(self.fc4(x)))
        x = self.norm5(self.relu(self.fc5(x)))
        x = self.relu(self.fc6(x))
        x = self.norm(x)
        x = self.sigmoid(self.drop2(x))
        return x