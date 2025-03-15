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
            state = state.contiguous().cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.contiguous().cpu()
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process states on CPU first
        if isinstance(states[0], torch.Tensor):
            # Convert tensors to numpy arrays with consistent layout
            states_np = np.stack([s.cpu().numpy() for s in states])
        else:
            # Process numpy arrays directly
            states_np = np.stack([s if s.shape == (4,84,84) else s.squeeze(0) for s in states])
        
        # Same for next_states
        if isinstance(next_states[0], torch.Tensor):
            next_states_np = np.stack([s.cpu().numpy() for s in next_states])
        else:
            next_states_np = np.stack([s if s.shape == (4,84,84) else s.squeeze(0) for s in next_states])
        
        # Ensure correct shape (batch_size, channels, height, width)
        if states_np.ndim > 4:  # Extra dimension
            states_np = states_np.squeeze(1)
        if next_states_np.ndim > 4:
            next_states_np = next_states_np.squeeze(1)
        
        # If states are in NHWC format instead of NCHW, transpose them
        if states_np.shape[-1] == 4:  # Last dim is channels
            states_np = np.transpose(states_np, (0, 3, 1, 2))
        if next_states_np.shape[-1] == 4:
            next_states_np = np.transpose(next_states_np, (0, 3, 1, 2))
            
        # Convert to tensors - use torch.as_tensor which avoids copy when possible
        states = torch.from_numpy(states_np.copy()).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(next_states_np.copy()).float().to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)