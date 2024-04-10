import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        # print("Before FC1:", x.shape) # Test
        x = F.relu(self.fc1(x))
        # print("After FC1:", x.shape) #Test
        x = F.relu(self.fc2(x))
        # print("After FC2:", x.shape) #Test
        x = self.fc3(x)
        # print("After FC3:", x.shape) #Test
        return x