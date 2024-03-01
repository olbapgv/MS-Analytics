

import torch.nn as nn

class MLP(nn.Module):
    # define model elements
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        
        # Parameters
        hidden_dim = 128
        num_classes = 2
        
        # For the loader
        self.num_epochs = 50
        self.learning_rate = 0.0005
        self.betas = (0.9, 0.999)
        
        # Optimized version
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(-1)

    # forward propagate input
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x