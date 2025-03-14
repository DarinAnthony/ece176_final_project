from base import DQN
import torch
import torch.nn as nn

class DQN2(DQN):
    def __init__(self, input_shape, num_actions):
        super().__init__(input_shape, num_actions)
        
        self.convLayers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(self._get_conv_output_size(), 512)
        self.fc2 = nn.Linear(512, num_actions)
        