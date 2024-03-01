# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 22:01:07 2021

@author: Pablo González
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Loader():
    
    def __init__(self, train_dataset, valid_dataset, batch_size = 128):
        # create data loader
        torch.cuda.empty_cache()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_loader = torch.utils.data.DataLoader( dataset=train_dataset, 
                                                   batch_size=batch_size, shuffle=True)
        
        self.valid_loader = torch.utils.data.DataLoader( dataset=valid_dataset, 
                                                   batch_size=batch_size, shuffle=True)
        
        self.loss_func = nn.BCELoss()
        
        
    def train(self, model, verbose = True):
        
        self.model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, 
                                     betas = model.betas)
        # start training
        # lists for storing loss at each epoch
        best_val_loss = 99999999 # store best validation loss
        epoch_list = [] # list of epochs (for plotting later)
        train_loss_list = [] # training loss at each epoch
        val_loss_list = [] # validation loss at each epoch
        
        for epoch in range(model.num_epochs): # loop through epoc
            self.model.train() # set model to training mode
            batch_losses = []
            for batch_x, batch_y in self.train_loader: # for each training step
                
                # get batch of data
                train = torch.autograd.Variable(batch_x).float().to(self.device)
                label = torch.autograd.Variable(batch_y).float().to(self.device)
        
                prediction = self.model(train).to(self.device) # forward propogration
                label = torch.nn.functional.one_hot(label.to(torch.int64), 2)
                loss = self.loss_func(prediction, label.float().detach()) # calculate loss
                loss.backward() # calculate gradients
                optimizer.step() # update parameters based on caluclated gradients
                optimizer.zero_grad() # clear gradients for next train
        
                batch_losses.append(loss.item()) # record loss for this mini batch
        
            # record average mini batch loss over epoch
            training_loss = np.mean(batch_losses)
            train_loss_list.append(training_loss)
        
            with torch.no_grad(): # set all requires_grad to false
                val_losses = []
                for batch_x, batch_y in self.valid_loader:
                    self.model.eval() # ensure batchnorm and dropout work properly in evaluation mode
        
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
        
                    yhat = self.model(batch_x).to(self.device)
        
                    # record loss for this mini batch
                    batch_y = torch.nn.functional.one_hot(batch_y.to(torch.int64), 2)
                    val_loss = self.loss_func(yhat, batch_y.float().detach()).item()
                    val_losses.append(val_loss)
                
                # record average mini batch loss for validation
                validation_loss = np.mean(val_losses)
                if validation_loss < best_val_loss: best_val_loss = validation_loss # record best validation loss
                val_loss_list.append(validation_loss)
        
            if verbose == True:        
                print(f"[{epoch+1}] Training loss: {training_loss:.5f}\t Validation loss: {validation_loss:.5f}\t Best Validation loss: {best_val_loss:.5f}")
            epoch_list.append(epoch)
        self.epoch_list = epoch_list
        self.train_loss_list = train_loss_list
        self.val_loss_list = val_loss_list
            
    def eval(self, test_dataset):
        
        test_loader = torch.utils.data.DataLoader( dataset= test_dataset, shuffle=False,
                                                   batch_size = self.train_loader.batch_size)
                
        # Plot loss curve
        plt.plot(self.epoch_list, self.val_loss_list, label="validation")
        plt.plot(self.epoch_list, self.train_loss_list, label="train")
        plt.ylim(bottom=0)
        plt.xlabel("Number of epochs")
        plt.ylabel("MSE")
        plt.title("BCELoss vs Number of iterations")
        plt.legend()
        plt.show()
        
        with torch.no_grad():
            self.model.eval()
            # Compute accuracy on training set
            train_acc, train_len = [], []
            for batch_x, batch_y in self.train_loader:
              output = self.model(batch_x.float().to(self.device))
              output = torch.argmax(output, dim=1).cpu()
              acc = accuracy_score(batch_y.numpy(), output.numpy() )
              train_acc = np.append(train_acc, acc)
              train_len = np.append(train_len, len(output) )
            total = sum(train_len)
            train_len = np.divide(train_len , total)
            print("Accuracy on train set: " + str( np.average(train_acc, weights=train_len )))
            
            # Compute accuracy on validation set
            valid_acc, valid_len = [], []
            for batch_x, batch_y in self.valid_loader:
              output = self.model(batch_x.float().to(self.device))
              output = torch.argmax(output, dim=1).cpu()
              acc = accuracy_score(batch_y.numpy(), output.numpy() )
              valid_acc = np.append(valid_acc, acc)
              valid_len = np.append(valid_len, len(output) )
            total = sum(valid_len)
            valid_len = np.divide(valid_len , total)
            print("Accuracy on validation set: " + str( np.average(valid_acc, weights=valid_len )))
            
            # Compute accuracy on test set
            test_acc, test_len = [], []
            for batch_x, batch_y in test_loader:
              output = self.model(batch_x.float().to(self.device))
              output = torch.argmax(output, dim=1).cpu()
              acc = accuracy_score(batch_y.numpy(), output.numpy() )
              test_acc = np.append(test_acc, acc)
              test_len = np.append(test_len, len(output) )
            total = sum(test_len)
            test_len = np.divide(test_len , total)
            print("Accuracy on test set: " + str( np.average(test_acc, weights=test_len )))
            
