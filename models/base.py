import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


# DQN CNN Architecture
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Original DQN architecture from the paper
        self.convLayers = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        self.fcLayers = nn.Sequential(
            nn.Linear(self._get_conv_output_size(), 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def _get_conv_output_size(self):
        # Create a sample input tensor to get the size of the flattened output
        sample_input = torch.zeros(1, *self.input_shape)
        output = self.convLayers(sample_input)
        return int(np.prod(output.size()))
    
    def forward(self, x):
        x = self.convLayers(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fcLayers(x)
        return x