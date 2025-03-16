import torch
import torch.nn as nn
import torch.nn.functional as F

class FlappyBirdDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(FlappyBirdDQN, self).__init__()
        
        # Input shape should be (4, 80, 80)
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Get the actual output size by doing a forward pass
        temp_input = torch.zeros(1, *input_shape)
        conv_out = self.conv(temp_input)
        self.fc_input_size = conv_out.numel() // conv_out.size(0)
        
        print(f"Convolutional output size: {self.fc_input_size}")
        
        # Now create the fully connected layers with the correct input size
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        # Ensure input is normalized
        if x.max() > 1.0:
            x = x / 255.0
            
        # Forward pass through conv layers
        x = self.conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Forward pass through fc layers
        x = self.fc(x)
        
        return x