import random
import torch
import numpy as np
from collections import deque

# Replay Memory for experience replay
# Modified ReplayBuffer for CPU storage with efficient GPU transfer
class ReplayBuffer:
    def __init__(self, capacity, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        # Always store states on CPU to save GPU memory
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process states on CPU first
        if isinstance(states[0], torch.Tensor):
            # Convert to numpy then back to tensor for consistent handling
            states_np = np.stack([s.cpu().numpy() for s in states])
        else:
            states_np = np.stack([s if s.shape == (4,84,84) else s.squeeze(0) for s in states])
        
        # Same for next_states
        if isinstance(next_states[0], torch.Tensor):
            next_states_np = np.stack([s.cpu().numpy() for s in next_states])
        else:
            next_states_np = np.stack([s if s.shape == (4,84,84) else s.squeeze(0) for s in next_states])
            
        # Convert to tensors and transfer to device just before returning
        states = torch.tensor(states_np, dtype=torch.float32, device=self.device).squeeze(1)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=self.device).squeeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)