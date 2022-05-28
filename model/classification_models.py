import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision



class FCNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=2, bn_momentum= 0.9):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim,momentum=bn_momentum)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim,momentum=bn_momentum)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        return out



def simple_FCNet(input_dim = 9):
    model = FCNet(input_dim=input_dim,output_dim=2)
    return model