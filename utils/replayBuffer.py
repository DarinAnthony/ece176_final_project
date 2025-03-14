import random
import torch
import numpy as np
from collections import deque

# Replay Memory for experience replay
class ReplayBuffer:
    def __init__(self, capacity, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to torch tensors and ensure all are on the same device
        # Check if states are already tensors (they might be from get_state)
        if isinstance(states[0], torch.Tensor):
            states = torch.cat(states).to(self.device)
        else:
            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        if isinstance(next_states[0], torch.Tensor):
            next_states = torch.cat(next_states).to(self.device)
        else:
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)