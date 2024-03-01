# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 12:03:01 2021

@author: pablo
"""
import torch.nn as nn

class EMB_CNN(nn.Module): # In Progress
    # define model elements
    def __init__(self, weights):
        super(EMB_CNN, self).__init__()

        # Embedding
        self.embeddings = nn.Embedding.from_pretrained(weights)
        self.embeddings.requires_grad = False

        # Parameters:
        in_channels = 700
        out_channels = (512, 256, 128, 64, 32, 16)
        num_classes = 2
        kernel_size = 8
        stride = 1
        padding = 3
        hidden = 1024

        # For the loader:
        self.num_epochs = 10
        self.learning_rate, self.betas = 5e-5, (0.999, 0.999999)


        # Initial Hidden
        self.leaky = nn.LeakyReLU(0.35)
        self.conv1 = nn.Conv1d(in_channels, out_channels[0], kernel_size, stride, padding)
        self.conv2 = nn.Conv1d( out_channels[0] , out_channels[1], kernel_size, stride, padding)
        self.conv3 = nn.Conv1d( out_channels[1], out_channels[2], kernel_size, stride, padding)
        self.conv4 = nn.Conv1d( out_channels[2] , out_channels[3], kernel_size, stride, padding)
        self.conv5 = nn.Conv1d( out_channels[3] , out_channels[4], kernel_size, stride, padding)
        self.conv6 = nn.Conv1d( out_channels[4] , out_channels[5], kernel_size, stride, padding)
        self.pool = nn.MaxPool1d(kernel_size)

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear( out_channels[5] , hidden)
        self.fc2 = nn.Linear( hidden , num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):

      x = self.embeddings( x.to(int) )
      x = self.leaky( self.conv1(x) )
      x = self.leaky( self.conv2(x) )
      x = self.leaky( self.conv3(x) )
      x = self.pool( x )
      x = self.leaky( self.conv4(x) )
      x = self.leaky( self.conv5(x) )
      x = self.leaky( self.conv6(x) )
      x = self.pool( x )
      x = x[:,:,-1]
      x = self.leaky( self.fc1(x) )
      x = self.drop(x)
      x = self.fc2(x)
      x = self.softmax(x)

      return x